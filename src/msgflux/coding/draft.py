"""An unpersisted conversation that opens durable resources on its first prompt."""

import asyncio
from contextlib import aclosing

from msgflux.runtime.event_hub import ThreadSnapshot


class DraftCodingSession:
    """Keep a thread identity without opening models, workspaces or databases."""

    def __init__(self, thread_id, activate):
        self.thread_id = thread_id
        self._activate = activate
        self._session = None
        self._lock = asyncio.Lock()

    @property
    def agent(self):
        return self._session.agent if self._session is not None else None

    @property
    def _abort_signal(self):
        return self._session._abort_signal if self._session is not None else None

    def cancel(self):
        if self._session is not None:
            self._session.cancel()

    async def snapshot(self):
        if self._session is None:
            return ThreadSnapshot(self.thread_id, namespace="main")
        return await self._session.snapshot()

    def runs(self):
        return self._session.runs() if self._session is not None else ()

    def latest_run(self):
        return self._session.latest_run() if self._session is not None else None

    async def stream(self, prompt):
        if not isinstance(prompt, str):
            raise TypeError("`prompt` must be a string")
        async with self._lock:
            if self._session is None:
                self._session = await self._activate(self)
        async with aclosing(self._session.stream(prompt)) as events:
            async for event in events:
                yield event

    async def resume(self, run_id):
        if self._session is None:
            raise ValueError("No saved run to continue")
        async with aclosing(self._session.resume(run_id)) as events:
            async for event in events:
                yield event
