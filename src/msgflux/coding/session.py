"""A small, headless session wrapper for interactive coding clients."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing

from msgflux.coding.checkpoints import CodingCheckpointExtension
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.abort import AbortSignal
from msgflux.runtime.context import (
    ExecutionScope,
    execution_context,
    get_execution_scope,
    new_thread_id,
)
from msgflux.runtime.event_hub import ThreadSnapshot
from msgflux.runtime.events import ExecutionEvent
from msgflux.runtime.permissions import require_permissions

TERMINAL_RUN_STATUSES = frozenset({"completed", "interrupted"})


class CodingSession:
    """Keep a durable thread identity while streaming Agent events.

    ``scope_factory`` can add application-specific execution settings, such as
    a workspace environment. It receives the base scope and must return a
    complete :class:`ExecutionScope`.

    An ``Agent`` owns its durable namespace and uses its module name for
    checkpoint keys. For that case, ``namespace`` is replaced by the agent's
    module name so the session reports the same namespace used for persistence.
    """

    def __init__(
        self,
        agent,
        *,
        thread_id: str | None = None,
        namespace: str = "coding",
        checkpoint_store=None,
        scope_factory: Callable[[ExecutionScope], ExecutionScope] | None = None,
    ) -> None:
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("`namespace` must be a non-empty string")
        self.agent = agent
        if isinstance(agent, Agent) and not agent.has_extension("coding_checkpoints"):
            agent.register_extension("coding_checkpoints", CodingCheckpointExtension())
        if thread_id is not None and (not isinstance(thread_id, str) or not thread_id):
            raise ValueError("`thread_id` must be a non-empty string or None")
        self._thread_id = thread_id if thread_id is not None else new_thread_id()
        get_module_name = getattr(agent, "get_module_name", None)
        agent_namespace = get_module_name() if callable(get_module_name) else None
        self.namespace = agent_namespace or namespace
        self.checkpoint_store = checkpoint_store
        self.scope_factory = scope_factory
        self._abort_signal: AbortSignal | None = None

    @property
    def thread_id(self) -> str:
        """Stable durable conversation identity for this session."""
        return self._thread_id

    def cancel(self) -> None:
        """Request cooperative cancellation of the currently streamed run."""
        if self._abort_signal is not None:
            self._abort_signal.abort("coding session cancelled")

    async def snapshot(self) -> ThreadSnapshot:
        """Return the Agent's current durable and live snapshot for this thread.

        ``snapshot.messages`` contains restored conversation history when the
        checkpoint store has a run for this thread.
        """
        watch = getattr(self.agent, "watch", None)
        if not callable(watch):
            raise TypeError("The session agent must support watch(thread_id)")
        with execution_context(
            scope=self._create_scope(), checkpoint_store=self.checkpoint_store
        ):
            async with watch(self._thread_id) as watcher:
                return watcher.snapshot

    async def workspace_files(self, limit: int = 500) -> tuple[str, ...]:
        """List bounded virtual file paths from the session's live workspace.

        Directories that the current scope cannot list are skipped. Both the
        number of returned files and the total number of entries inspected are
        bounded; paths are virtual POSIX paths, never host filesystem paths.
        """
        if type(limit) is not int or not 0 < limit <= 10_000:
            raise ValueError("`limit` must be an integer from 1 to 10000")
        scope = self._create_scope()
        filesystem = scope.workspace
        if filesystem is None:
            return ()

        with execution_context(
            scope=scope,
            checkpoint_store=self.checkpoint_store,
        ):
            return await _enumerate_workspace_files(filesystem, limit)

    @staticmethod
    async def _scan_workspace_directory(filesystem, path, max_entries):
        try:
            return await filesystem.ascandir(path, max_entries=max_entries)
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            return ()
        except ValueError:
            return None

    @staticmethod
    def _can_read_workspace_file(filesystem, path):
        try:
            if not (
                get_execution_scope().permissions or filesystem.permissions
            ).missing(("filesystem.read",)):
                return True
            require_permissions((), (filesystem.permission(path, "filesystem.read"),))
        except PermissionError:
            return False
        return True

    def runs(self) -> tuple[dict, ...]:
        """Discover saved turns; checkpoint state remains authoritative."""
        if not callable(getattr(self.checkpoint_store, "list_runs", None)):
            return ()
        return tuple(
            dict(item)
            for item in self.checkpoint_store.list_runs(self.namespace, self.thread_id)
        )

    def latest_run(self) -> dict | None:
        runs = self.runs()
        return runs[0] if runs else None

    def saved_state(self, run_id: str) -> dict:
        if self.checkpoint_store is None:
            raise ValueError("This session has no checkpoint store")
        state = self.checkpoint_store.load_state(self.namespace, self.thread_id, run_id)
        if state is None:
            raise ValueError(f"Unknown run: {run_id}")
        return state

    async def stream(self, prompt: str) -> AsyncIterator[ExecutionEvent]:
        """Run ``prompt`` and yield the Agent's ordered execution events."""
        if not isinstance(prompt, str):
            raise TypeError("`prompt` must be a string")
        latest = self.latest_run()
        if latest:
            validator = getattr(
                self.agent, "_validate_checkpoint_command_receipts", None
            )
            if callable(validator):
                validator(self.saved_state(latest["run_id"]))
        if latest and latest["status"] not in TERMINAL_RUN_STATUSES:
            raise ValueError(
                "An unfinished run exists. Use /continue to resume it explicitly."
            )
        async with aclosing(self._stream(prompt)) as events:
            async for event in events:
                yield event

    async def resume(self, run_id: str) -> AsyncIterator[ExecutionEvent]:
        """Resume a paused durable run and yield its execution events."""
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("`run_id` must be a non-empty string")
        state = self.saved_state(run_id)
        if state.get("status") in TERMINAL_RUN_STATUSES:
            raise ValueError(
                "This run is terminal. Send a new prompt to continue the conversation."
            )
        latest = self.latest_run()
        if latest and latest["run_id"] != run_id:
            raise ValueError(
                "Only the latest unfinished run can continue; "
                "older checkpoints remain unchanged"
            )
        async with aclosing(self._stream(None, run_id=run_id)) as events:
            async for event in events:
                yield event

    async def _stream(
        self, prompt: str | None, *, run_id: str | None = None
    ) -> AsyncIterator[ExecutionEvent]:
        if self._abort_signal is not None:
            raise RuntimeError("A CodingSession can stream only one prompt at a time")

        signal = AbortSignal()
        self._abort_signal = signal
        try:
            scope = self._create_scope(run_id=run_id, abort_signal=signal)
            with execution_context(
                scope=scope,
                checkpoint_store=self.checkpoint_store,
            ):
                async with aclosing(
                    self.agent.stream_events(prompt, scope=scope)
                ) as events:
                    async for event in events:
                        yield event
        finally:
            self._abort_signal = None

    def _create_scope(
        self,
        *,
        run_id: str | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> ExecutionScope:
        scope = ExecutionScope(
            thread_id=self._thread_id,
            namespace=self.namespace,
            run_id=run_id,
            abort_signal=abort_signal,
        )
        if self.scope_factory is None:
            return scope
        scope = self.scope_factory(scope)
        if not isinstance(scope, ExecutionScope):
            raise TypeError("`scope_factory` must return an ExecutionScope")
        if scope.thread_id != self._thread_id:
            raise ValueError("`scope_factory` must preserve the session thread_id")
        return scope.with_overrides(run_id=run_id, abort_signal=abort_signal)


async def _enumerate_workspace_files(filesystem, limit: int) -> tuple[str, ...]:
    files: list[str] = []
    directories = deque([filesystem.cwd])
    inspected = 0
    inspection_limit = limit * 4
    while directories and inspected < inspection_limit and len(files) < limit:
        directory = directories.popleft()
        remaining = inspection_limit - inspected
        entries = await CodingSession._scan_workspace_directory(
            filesystem, directory, remaining
        )
        if entries is None:
            # An over-budget directory cannot be paged through by scandir.
            break
        if not entries:
            continue

        inspected += len(entries)
        for entry in entries:
            path = f"{directory.rstrip('/')}/{entry.name}"
            if entry.kind == "file":
                if CodingSession._can_read_workspace_file(filesystem, path):
                    files.append(path)
                    if len(files) == limit:
                        break
            elif entry.kind == "directory" and entry.name not in {
                ".git",
                ".venv",
                "__pycache__",
                "node_modules",
                ".pytest_cache",
                ".ruff_cache",
                "build",
                "dist",
                "site",
            }:
                directories.append(path)
    return tuple(sorted(files))
