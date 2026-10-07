"""Frontend selection of service-owned coding conversations."""

from pathlib import Path

from msgflux.runtime.service.http import AgentSessionClient


class RemoteCodingHost:
    """Select sessions without owning Agents, stores, or background tasks."""

    def __init__(
        self,
        client,
        *,
        agent_id: str,
        workspace: Path,
        require_agent_id: bool = False,
        connector=None,
    ):
        self.client = client
        self.agent_id = agent_id
        self.require_agent_id = require_agent_id
        self._connector = connector
        self._owns_client = False
        self.workspace = str(workspace.expanduser().resolve(strict=True))
        self.session = None
        self.approval_controller = None
        self.cleanup_error = None
        self.is_new = False

    async def threads(self):
        return tuple(
            thread
            for thread in await self.client.threads()
            if thread.cwd == self.workspace
        )

    async def select(self, thread_id=None):
        if thread_id is None:
            session = await AgentSessionClient.open(
                self.client, agent_id=self.agent_id, cwd=self.workspace
            )
        else:
            thread = next(
                (
                    item
                    for item in await self.client.threads()
                    if item.thread_id == thread_id
                ),
                None,
            )
            if thread is None:
                raise ValueError(f"Unknown coding thread: {thread_id}")
            if thread.cwd is None:
                raise ValueError("This thread has no saved workspace directory")
            if self.require_agent_id and thread.agent_id != self.agent_id:
                raise ValueError(
                    "The saved thread uses another profile or permission mode. "
                    "Start a new conversation for the requested configuration."
                )
            session = AgentSessionClient(self.client, thread)
            self.workspace = thread.cwd
        self.session = session
        self.is_new = thread_id is None
        return session, None

    async def aclose(self):
        if self.session is not None:
            await self.session.aclose()
        if self._owns_client:
            await self.client.aclose()

    async def reconnect(self):
        """Rediscover a daemon endpoint without repeating an Agent input."""
        if self._connector is None:
            return self.session
        replacement = await self._connector()
        previous = self.client
        owned_previous = self._owns_client
        self.client = replacement
        try:
            if self.session is not None:
                session, _controller = await self.select(self.session.thread_id)
            else:
                session = None
        except BaseException:
            self.client = previous
            await replacement.aclose()
            raise
        self._owns_client = True
        if owned_previous:
            await previous.aclose()
        return session
