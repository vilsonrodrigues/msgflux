import asyncio
from contextlib import suppress
from typing import Any, Awaitable, Callable, Dict, List, Optional

DoneCallback = Callable[[str, asyncio.Task[Any]], None]


class RunManager:
    """Tracks active and pending async work by session."""

    def __init__(self) -> None:
        self._active_tasks: Dict[str, asyncio.Task[Any]] = {}
        self._pending_tasks: Dict[str, asyncio.Task[Any]] = {}
        self._pending_items: Dict[str, List[Any]] = {}

    def active_task(self, session_id: str) -> Optional[asyncio.Task[Any]]:
        task = self._active_tasks.get(str(session_id))
        if task is None or task.done():
            return None
        return task

    def create_active(
        self,
        session_id: str,
        awaitable: Awaitable[Any],
        *,
        done_callback: Optional[DoneCallback] = None,
    ) -> asyncio.Task[Any]:
        session_key = str(session_id)
        task = asyncio.create_task(awaitable)
        self._active_tasks[session_key] = task
        task.add_done_callback(
            lambda completed, key=session_key: self._active_done(
                key,
                completed,
                done_callback,
            )
        )
        return task

    def forget_active(self, session_id: str, task: asyncio.Task[Any]) -> None:
        session_key = str(session_id)
        if self._active_tasks.get(session_key) is task:
            self._active_tasks.pop(session_key, None)

    def add_pending_item(self, session_id: str, item: Any) -> None:
        self._pending_items.setdefault(str(session_id), []).append(item)

    def pop_pending_items(self, session_id: str) -> List[Any]:
        return self._pending_items.pop(str(session_id), [])

    def replace_pending(
        self,
        session_id: str,
        awaitable: Awaitable[Any],
    ) -> asyncio.Task[Any]:
        session_key = str(session_id)
        task = self._pending_tasks.pop(session_key, None)
        if task is not None:
            task.cancel()

        pending_task = asyncio.create_task(awaitable)
        self._pending_tasks[session_key] = pending_task
        return pending_task

    def forget_pending_if_current(
        self,
        session_id: str,
        task: asyncio.Task[Any],
    ) -> None:
        session_key = str(session_id)
        if self._pending_tasks.get(session_key) is task:
            self._pending_tasks.pop(session_key, None)

    def cancel_session(self, session_id: str) -> bool:
        session_key = str(session_id)
        cancelled = False

        pending_task = self._pending_tasks.pop(session_key, None)
        if pending_task is not None:
            pending_task.cancel()
            self._pending_items.pop(session_key, None)
            cancelled = True

        task = self.active_task(session_key)
        if task is not None:
            task.cancel()
            cancelled = True
        return cancelled

    async def drain(self) -> None:
        pending_tasks = list(self._pending_tasks.values())
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        tasks = list(self._active_tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def cancel_all(self) -> None:
        tasks = [*self._active_tasks.values(), *self._pending_tasks.values()]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._active_tasks.clear()
        self._pending_tasks.clear()
        self._pending_items.clear()

    def _active_done(
        self,
        session_id: str,
        task: asyncio.Task[Any],
        done_callback: Optional[DoneCallback],
    ) -> None:
        self.forget_active(session_id, task)
        if done_callback is not None:
            done_callback(session_id, task)
