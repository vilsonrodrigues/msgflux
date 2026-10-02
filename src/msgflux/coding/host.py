"""Resource ownership and session selection, independent of Textual."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from msgflux.coding.storage import ThreadStorage, validate_thread_id
from msgflux.runtime.context import new_thread_id


class CodingHost:
    """Open a replacement before closing the old session's resources.

    The factory returns ``(session, approval_controller, async_close)``. Failed
    factory calls must release resources they acquired. History is never copied.
    """

    def __init__(
        self,
        storage: ThreadStorage,
        workspace: str,
        factory: Callable[..., Awaitable],
        *,
        existing_thread: Callable[[str], bool] | None = None,
    ):
        self.storage = storage
        self.workspace = str(Path(workspace).resolve())
        self._factory = factory
        self._existing_thread = existing_thread
        self._close = None
        self._lock = asyncio.Lock()
        self.cleanup_error = None
        self._pending_closes = []
        self.session = None
        self.approval_controller = None

    def threads(self):
        return self.storage.list_threads(workspace=self.workspace)

    def _require_idle_tasks(self):
        if self.session is not None:
            library = getattr(self.session.agent, "tool_library", None)
            if library is not None:
                tasks = library.get_task_store().list()
                if any(task.status in {"queued", "running"} for task in tasks):
                    raise RuntimeError(
                        "Finish or interrupt background tasks before switching sessions"
                    )

    async def select(self, thread_id: str | None = None):
        async with self._lock:
            return await self._select(thread_id)

    async def _select(self, thread_id):
        if self.session is not None and self.session._abort_signal is not None:
            raise RuntimeError("Cancel the current run before switching sessions")
        self._require_idle_tasks()
        if thread_id is None:
            thread_id = new_thread_id()
            self.storage.create_thread(thread_id, workspace=self.workspace)
        else:
            validate_thread_id(thread_id)
            try:
                metadata = self.storage.read_metadata(thread_id)
            except FileNotFoundError:
                if self._existing_thread is None or not self._existing_thread(
                    thread_id
                ):
                    raise
                metadata = None
            if (
                metadata is not None
                and metadata.workspace is not None
                and metadata.workspace != self.workspace
            ):
                raise ValueError(
                    "This thread belongs to another workspace. "
                    "Reopen with its --workspace path."
                )
            checkpoint = self.storage.checkpoint_path(thread_id)
            if metadata is not None and (
                not checkpoint.is_file() or checkpoint.is_symlink()
            ):
                raise ValueError("The thread has no valid checkpoint database")
        session, controller, close = await self._factory(thread_id)
        previous_close = self._close
        self.session, self.approval_controller, self._close = session, controller, close
        self.cleanup_error = None
        if previous_close is not None:
            try:
                await previous_close()
            except Exception as error:
                self.cleanup_error = error
                self._pending_closes.append(previous_close)
        return session, controller

    async def aclose(self):
        async with self._lock:
            closes = self._pending_closes[:]
            self._pending_closes.clear()
            if self._close is not None:
                closes.append(self._close)
                self._close = None
            errors = []
            for close in closes:
                try:
                    await close()
                except Exception as error:
                    errors.append(error)
            if errors:
                raise errors[0]
