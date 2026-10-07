"""Trusted backend construction for the experimental remote coding frontend."""

from __future__ import annotations

import asyncio
import inspect
import re
from concurrent.futures import Future
from functools import partial
from pathlib import Path

from msgflux.coding.accounts import AccountStorage
from msgflux.coding.checkpoints import CodingCheckpointExtension
from msgflux.coding.config import load_config
from msgflux.coding.prompts import system_prompt
from msgflux.coding.tools import resolve_tools
from msgflux.coding.workspace import open_coding_workspace
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models import Model
from msgflux.nn import Agent
from msgflux.runtime import (
    AgentApprovals,
    AgentInbox,
    SQLiteAgentInboxStore,
    SQLiteApprovalStore,
)
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.tasks import SQLiteTaskStore


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
        raise ExceptionGroup("Coding backend cleanup failed", errors)


async def _stop_background(agent, tasks):
    if agent is None or tasks is None:
        return
    dispatcher = agent.tool_library.get_background_dispatcher()
    futures = []
    for task in tasks.list():
        future = dispatcher.get_task_future(task.task_id)
        if future is not None:
            tasks.request_interrupt(task.task_id)
            futures.append(
                asyncio.wrap_future(future) if isinstance(future, Future) else future
            )
    if futures:
        await asyncio.gather(*futures, return_exceptions=True)


def _configure_approvals(agent, profile, folder, *, read_only):
    names = profile.approvals if not read_only else ()
    if not names:
        return None, None

    library = agent.tool_library
    missing = sorted(set(names) - set(library.get_tool_names()))
    if missing:
        raise ValueError(f"Approval tool is not available in profile {missing[0]!r}")
    for name in names:
        definition = library.get_tool_definition(name)
        if (
            definition.dispatch.name != "foreground"
            or definition.feedback.name == "call_as_response"
        ):
            raise ValueError(
                f"Approval tool {name!r} must be executable and foreground"
            )

    store = SQLiteApprovalStore(folder / "approvals.sqlite3")
    policy = AgentApprovals(
        store,
        dict.fromkeys(names, "coding-profile-v1"),
        "coding-profile-v1",
    )
    agent.approvals = policy
    return store, policy


async def _create_session(thread, *, state_dir, config, accounts, profile, read_only):
    model_path = profile.model or config.default_model or "openai-codex/gpt-6-luna"
    effort = profile.reasoning_effort or config.reasoning_effort or "medium"
    if thread.cwd is None:
        raise ValueError("A coding conversation requires a project directory")
    if re.fullmatch(r"[A-Za-z0-9_-]+", thread.thread_id) is None:
        raise ValueError("Unsupported coding thread ID")
    folder = state_dir / "threads" / thread.thread_id
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = folder / "checkpoints.sqlite3"
    model = workspace = registry = None
    checkpoints = tasks = inbox_store = None
    approvals = agent = None

    async def close():
        try:
            await _stop_background(agent, tasks)
        finally:
            await _close_resources(
                (model, workspace, registry, checkpoints, tasks, inbox_store, approvals)
            )

    try:
        checkpoints = SQLiteCheckpointStore(checkpoint_path)
        tasks = SQLiteTaskStore(str(checkpoint_path))
        inbox_store = SQLiteAgentInboxStore(str(checkpoint_path))
        inbox = AgentInbox(store=inbox_store, owner="main")
        workspace, registry = await open_coding_workspace(
            Path(thread.cwd), checkpoint_path, read_only=read_only
        )
        provider = model_path.partition("/")[0]
        alias = config.active_accounts.get(provider)
        kwargs = (
            {"credential_resolver": accounts.credential_resolver(provider, alias)}
            if alias is not None
            else {}
        )
        model = Model.chat_completion(model_path, **kwargs)
        if model.supports_reasoning_effort():
            model.set_reasoning_effort(effort)
        elif profile.reasoning_effort or config.reasoning_effort:
            raise ValueError("The selected model does not support reasoning effort")
        tools = resolve_tools(
            profile.tools,
            model=model,
            allow_edits=not read_only,
            has_executor=workspace.supports_execution and not read_only,
            interactive=True,
        )
        agent = Agent(
            name="main",
            model=model,
            workspace=workspace,
            tools=tools,
            system_prompt=system_prompt(
                thread.cwd, interactive=True, read_only=read_only
            ),
            checkpoint_store=checkpoints,
            agent_inbox=inbox,
            config={"stream": True},
        )
        approvals, approval_policy = _configure_approvals(
            agent, profile, folder, read_only=read_only
        )
        agent.tool_library.set_task_store(tasks)
        agent.register_extension("coding_checkpoints", CodingCheckpointExtension())
        return AgentSession(
            agent,
            task_store=tasks,
            agent_inbox=inbox,
            scope_factory=(lambda scope: scope.with_overrides(principal="local-user"))
            if approval_policy is not None
            else None,
            approval_reviewer="local-user" if approval_policy is not None else None,
            on_close=close,
        )
    except BaseException:
        await close()
        raise


def create_service(runtime_dir: Path) -> AgentService:
    """Build a service from server-owned config; model creation stays lazy."""
    state_dir = runtime_dir.parent
    config = load_config(state_dir / "config.toml")
    accounts = AccountStorage(state_dir)
    journal = SQLiteServiceStore(runtime_dir / "service.sqlite3")
    service = AgentService(store=journal)

    try:
        for name in dict.fromkeys((config.default_profile, *config.profiles)):
            profile = config.profile(name)
            for read_only in (False, True):
                agent_id = f"{name}:read-only" if read_only else name
                service.register(
                    agent_id,
                    partial(
                        _create_session,
                        state_dir=state_dir,
                        config=config,
                        accounts=accounts,
                        profile=profile,
                        read_only=read_only,
                    ),
                )
    except BaseException:
        journal.close()
        raise
    return service
