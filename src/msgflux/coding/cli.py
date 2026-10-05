"""Entry point for the optional Vulcano coding interface."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import importlib
import inspect
import os
import sqlite3
import sys
from contextlib import aclosing
from dataclasses import replace
from pathlib import Path

import msgspec

from msgflux.coding.accounts import AccountStorage
from msgflux.coding.approval import CodingApprovalController
from msgflux.coding.config import ToolSelection, load_config, select_tools
from msgflux.coding.extensions import CodingExtensions
from msgflux.coding.host import CodingHost
from msgflux.coding.prompts import system_prompt
from msgflux.coding.session import CodingSession
from msgflux.coding.storage import ThreadStorage
from msgflux.coding.tools import resolve_tools, validate_tool_selection
from msgflux.coding.workspace import open_coding_workspace
from msgflux.data.stores.providers.sqlite import SQLiteCheckpointStore
from msgflux.models import Model
from msgflux.models.gateway import ModelGateway
from msgflux.nn import Agent
from msgflux.runtime import (
    AgentApprovals,
    SQLiteApprovalStore,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the msgFlux coding TUI")
    parser.add_argument("--model", help="Override the configured default model")
    parser.add_argument("--profile", help="Coding profile from config.toml")
    parser.add_argument(
        "--tools",
        metavar="NAMES",
        help="Comma-separated active tools; tool flags replace profile tool selection",
    )
    parser.add_argument(
        "--deferred-tools",
        metavar="NAMES",
        help="Comma-separated deferred tools; omitted tool list is empty on override",
    )
    parser.add_argument("--reasoning-effort", help="Override reasoning effort")
    parser.add_argument(
        "--account",
        action="append",
        default=[],
        metavar="PROVIDER:ALIAS",
        help="Use a named account for this run without changing config.toml",
    )
    parser.add_argument("--config", type=Path, help="TOML configuration file")
    parser.add_argument(
        "-c",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one TOML setting using a dotted key",
    )
    parser.add_argument(
        "--workspace", type=Path, default=Path.cwd(), help="Project directory"
    )
    parser.add_argument(
        "-p", "--print", dest="prompt", help="Run one prompt without the TUI"
    )
    parser.add_argument(
        "--read-only", action="store_true", help="Disable edits and Bash"
    )
    parser.add_argument(
        "--review-edits",
        action="store_true",
        help="Review file tool edits before applying",
    )
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--thread", help="Existing thread ID to reopen")
    choice.add_argument(
        "--resume",
        nargs="?",
        const="",
        metavar="THREAD_ID",
        help="Reopen a thread, or choose one interactively",
    )
    choice.add_argument(
        "--continue",
        dest="continue_last",
        action="store_true",
        help="Reopen the most recent thread for this workspace",
    )
    parser.add_argument(
        "--extension",
        action="append",
        default=[],
        metavar="MODULE:FUNCTION",
        help="Call a trusted Python function to register UI extensions",
    )
    parser.add_argument(
        "--allow-edits",
        action="store_true",
        help="Enable edits (the default; retained for compatibility)",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path.home() / ".msgflux",
        help="Root directory for configuration, accounts and threads",
    )
    return parser


def _load_extensions(specs: list[str]) -> CodingExtensions:
    extensions = CodingExtensions()
    for spec in specs:
        module_name, separator, function_name = spec.partition(":")
        if not separator or not module_name or not function_name:
            raise ValueError("Extension must be MODULE:FUNCTION")
        register = getattr(importlib.import_module(module_name), function_name)
        if not callable(register):
            raise TypeError(f"Extension entry point {spec!r} is not callable")
        try:
            extensions.load(register)
        except Exception as error:
            error.add_note(f"Coding extension entry point: {spec}")
            raise
    return extensions


def _account_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="vulcano account")
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".msgflux")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="Store one named provider API key")
    add.add_argument("provider")
    add.add_argument("alias")
    add.add_argument("--key-env", help="Read the key from this environment variable")
    add.add_argument("--model", action="append", default=[])
    listed = commands.add_parser("list", help="List named accounts without secrets")
    listed.add_argument("provider", nargs="?")
    args = parser.parse_args(argv)
    accounts = AccountStorage(args.state_dir)
    if args.command == "list":
        for account in accounts.list(args.provider):
            models = ", ".join(account.models)
            sys.stdout.write(f"{account.provider}:{account.alias}\t{models}\n")
        return
    key = os.environ.get(args.key_env) if args.key_env else getpass.getpass("API key: ")
    if not key:
        raise ValueError("The selected API key is empty or unavailable")
    accounts.add(args.provider, args.alias, key, models=tuple(args.model))
    sys.stdout.write(f"Stored {args.provider}:{args.alias}\n")


def _make_model(model_path: str, effort: str | None, config, accounts):
    provider, separator, _ = model_path.partition("/")
    if not separator:
        raise ValueError("Model must be provider/model-id")
    kwargs = {}
    account_alias = config.active_accounts.get(provider)
    if account_alias is not None:
        kwargs["credential_resolver"] = accounts.credential_resolver(
            provider, account_alias
        )
    try:
        model = Model.chat_completion(model_path, **kwargs)
    except TypeError as error:
        if account_alias is None:
            raise
        raise ValueError(
            f"Provider {provider!r} cannot use a selected coding account"
        ) from error
    reasoning = getattr(
        getattr(getattr(model, "profile", None), "capabilities", None),
        "reasoning",
        True,
    )
    if effort is None and reasoning and model.supports_reasoning_effort():
        effort = "medium"
    if effort is not None:
        if not model.supports_reasoning_effort():
            raise ValueError(f"Model {model_path!r} does not support reasoning effort")
        model.set_reasoning_effort(effort)
    if account_alias is not None:
        accounts.record_model(provider, account_alias, model_path)
    return model


def _make_subagents(config, accounts, checkpoint_store) -> tuple[Agent, ...]:
    agents = []
    for name, spec in config.agents.items():
        deployments = []
        for model_path in spec.models:
            model = _make_model(model_path, config.reasoning_effort, config, accounts)
            deployments.append(
                {
                    "model_name": model_path,
                    "model": model,
                    "description": model_path,
                }
            )
        gateway = ModelGateway(deployments, fallback=False)
        agents.append(
            Agent(
                name=name,
                model=gateway,
                system_prompt=(
                    (
                        spec.description
                        or "Explore the workspace and report relevant findings."
                    )
                    + " "
                    "Use read tools to verify claims. Do not edit files."
                ),
                tools=resolve_tools(
                    ToolSelection(active=("workspace",)),
                    model=gateway,
                    allow_edits=False,
                    has_executor=False,
                ),
                checkpoint_store=checkpoint_store,
            )
        )
    return tuple(agents)


def _legacy_thread_stores(root: Path, thread_id: str) -> tuple[Path, Path] | None:
    """Find an existing shared-store thread without moving its history."""
    for directory in (root, root / "coding"):
        checkpoint_path = directory / "checkpoints.sqlite3"
        if not checkpoint_path.is_file():
            continue
        with sqlite3.connect(checkpoint_path) as connection:
            try:
                row = connection.execute(
                    "SELECT 1 FROM checkpoints WHERE thread_id=? LIMIT 1",
                    (thread_id,),
                ).fetchone()
            except sqlite3.OperationalError:
                continue
        if row is not None:
            return checkpoint_path, directory / "approvals.sqlite3"
    return None


def _agent_namespace(store, thread_id):
    if store.list_runs("coding_agent", thread_id) and not store.list_runs(
        "main", thread_id
    ):
        return "coding_agent"
    return "main"


async def _close_resources(resources):
    errors = []
    for resource in resources:
        if resource is None:
            continue
        close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception as error:
                errors.append(error)
    if errors:
        for error in errors[1:]:
            errors[0].add_note(f"Additional cleanup failure: {error}")
        raise errors[0]


def _approval_controller(agent, policy):
    return (
        CodingApprovalController(agent, policy, principal=lambda: "local_user")
        if policy is not None
        else None
    )


def _approval_policy(store):
    return (
        AgentApprovals(
            store, {"write": "v1", "edit": "v1", "apply_patch": "v1"}, "vulcano-v1"
        )
        if store is not None
        else None
    )


async def _run(args: argparse.Namespace) -> None:  # noqa: C901
    if args.prompt is not None and args.review_edits:
        raise ValueError("--review-edits requires an interactive session")
    root = args.workspace.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Workspace must be a directory")
    state_dir = args.state_dir.expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)
    state_dir = state_dir.resolve()
    if root == state_dir or root in state_dir.parents:
        raise ValueError("State directory must be outside the workspace")
    config_path = args.config.expanduser() if args.config else state_dir / "config.toml"
    if args.config is not None and not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = load_config(config_path, tuple(args.c))
    if args.account:
        active_accounts = dict(config.active_accounts)
        for selected in args.account:
            provider, separator, alias = selected.partition(":")
            if not separator or not provider or not alias:
                raise ValueError("--account must be PROVIDER:ALIAS")
            active_accounts[provider] = alias
        config = msgspec.structs.replace(config, active_accounts=active_accounts)
    profile = config.profile(args.profile)
    selection = select_tools(
        profile.tools, active=args.tools, deferred=args.deferred_tools
    )
    extensions = _load_extensions(args.extension)
    validate_tool_selection(selection, tool_specs=extensions.tools())
    model_path = args.model or profile.model or config.default_model
    if model_path is None:
        raise ValueError("Set default_model in config.toml or pass --model")
    effort = (
        args.reasoning_effort or profile.reasoning_effort or config.reasoning_effort
    )
    accounts = AccountStorage(state_dir)
    thread_storage = ThreadStorage(state_dir)
    selected_thread = args.thread or args.resume or None
    if args.continue_last or args.resume == "":
        threads = thread_storage.list_threads(workspace=str(root))
        if not threads:
            raise ValueError("No saved sessions for this workspace")
        selected_thread = threads[0].thread_id
    if args.prompt is not None and args.resume == "":
        raise ValueError("--resume requires a thread ID in print mode")

    async def create_session(thread_id):
        legacy = (
            _legacy_thread_stores(state_dir, thread_id)
            if not thread_storage.thread_dir(thread_id).exists()
            else None
        )
        store = registry = workspace = approval_store = model = None
        subagents = ()
        custom_tools = []

        async def close():
            models = [model] if model is not None else []
            for subagent in subagents:
                models.extend(subagent.model.models)
            await _close_resources(
                [
                    *reversed(custom_tools),
                    *models,
                    approval_store,
                    workspace,
                    registry,
                    store,
                ]
            )

        try:
            store = (
                SQLiteCheckpointStore(str(legacy[0]))
                if legacy
                else thread_storage.open_checkpoint_store(thread_id)
            )
            checkpoint_path = (
                legacy[0] if legacy else thread_storage.checkpoint_path(thread_id)
            )
            workspace, registry = await open_coding_workspace(
                root, checkpoint_path, read_only=args.read_only
            )
            if args.review_edits and not args.read_only:
                approval_store = (
                    SQLiteApprovalStore(str(legacy[1]))
                    if legacy
                    else thread_storage.open_approval_store(thread_id)
                )
            approval_policy = _approval_policy(approval_store)
            model = _make_model(model_path, effort, config, accounts)
            subagents = (
                _make_subagents(config, accounts, store)
                if "agents" in (*selection.active, *selection.deferred)
                else ()
            )
            tools = resolve_tools(
                selection,
                tool_specs=extensions.tools(),
                on_tool_created=custom_tools.append,
                model=model,
                allow_edits=not args.read_only,
                has_executor=workspace.supports_execution and not args.read_only,
                agents=subagents,
                interactive=args.prompt is None,
            )
            agent = Agent(
                name=_agent_namespace(store, thread_id),
                model=model,
                workspace=workspace,
                system_prompt=system_prompt(
                    str(root), interactive=args.prompt is None, read_only=args.read_only
                ),
                config={"stream": True},
                tools=tools,
                checkpoint_store=store,
                approvals=approval_policy,
            )
            session = CodingSession(
                agent,
                thread_id=thread_id,
                checkpoint_store=store,
                scope_factory=lambda scope: replace(
                    scope,
                    workspace=workspace,
                    permissions=workspace.permissions,
                    principal="local_user",
                ),
            )
            # Validate history before replacing the current session/resources.
            await session.snapshot()
            controller = _approval_controller(agent, approval_policy)
            return session, controller, close
        except BaseException as error:
            try:
                await close()
            except Exception as cleanup_error:
                error.add_note(f"Session cleanup also failed: {cleanup_error}")
            raise

    host = CodingHost(
        thread_storage,
        str(root),
        create_session,
        existing_thread=lambda thread_id: (
            _legacy_thread_stores(state_dir, thread_id) is not None
        ),
    )
    try:
        session, approval_controller = await host.select(selected_thread)
        if args.prompt is not None:
            streamed = False
            async with aclosing(session.stream(args.prompt)) as events:
                async for event in events:
                    kind = (
                        event.type.value
                        if hasattr(event.type, "value")
                        else str(event.type)
                    )
                    if kind == "message.delta":
                        text = event.data.get("delta") or event.data.get("text") or ""
                        if isinstance(text, str):
                            sys.stdout.write(text)
                            sys.stdout.flush()
                            streamed = True
                    elif kind == "message.end" and not streamed:
                        sys.stdout.write(str(event.data.get("content", "")))
            sys.stdout.write("\n")
            if thread_storage.thread_dir(session.thread_id).exists():
                thread_storage.touch(session.thread_id)
        else:
            try:
                from msgflux.coding.tui import CodingApp  # noqa: PLC0415
            except ModuleNotFoundError as error:
                if error.name and error.name.startswith(("textual", "rich")):
                    raise SystemExit(
                        "Run the TUI with: uv run --extra coding vulcano ..."
                    ) from error
                raise
            await CodingApp(
                session,
                workspace=str(root),
                approval_controller=approval_controller,
                extensions=extensions,
                host=host,
                open_session_picker=args.resume == "",
            ).run_async()
    finally:
        await host.aclose()


def main(argv: list[str] | None = None) -> None:
    """Start the interactive coding application."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "account":
        _account_command(argv[1:])
        return
    args = _parser().parse_args(argv)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
