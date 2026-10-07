"""Project and permission bindings stay service-owned when selecting sessions."""

from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.coding.remote_host import RemoteCodingHost
from msgflux.runtime.service import ServiceThread
from msgflux.runtime.service.http import AgentServiceClient


@pytest.mark.asyncio
async def test_resume_uses_saved_workspace_and_profile(tmp_path):
    current = tmp_path / "current"
    saved = tmp_path / "saved"
    current.mkdir()
    saved.mkdir()
    client = Mock(spec=AgentServiceClient)
    client.threads = AsyncMock(
        return_value=(ServiceThread("saved", "full", str(saved)),)
    )
    host = RemoteCodingHost(client, agent_id="lite", workspace=current)
    session, _controller = await host.select("saved")
    assert session.agent_id == "full"
    assert session.workspace_root == host.workspace == str(saved)
    assert not host.is_new
    assert await host.threads() == (ServiceThread("saved", "full", str(saved)),)


@pytest.mark.asyncio
async def test_explicit_read_only_does_not_resume_a_writable_thread(tmp_path):
    client = Mock(spec=AgentServiceClient)
    client.threads = AsyncMock(
        return_value=(ServiceThread("saved", "lite", str(tmp_path)),)
    )
    host = RemoteCodingHost(
        client, agent_id="lite:read-only", workspace=tmp_path, require_agent_id=True
    )
    with pytest.raises(ValueError, match="permission mode"):
        await host.select("saved")
    assert host.session is None


@pytest.mark.asyncio
async def test_new_binding_uses_project_and_keeps_client_borrowed(tmp_path):
    client = Mock(spec=AgentServiceClient)
    client.open_thread = AsyncMock(
        return_value=ServiceThread("new", "lite", str(tmp_path))
    )
    client.aclose = AsyncMock()
    host = RemoteCodingHost(client, agent_id="lite", workspace=tmp_path)
    session, _controller = await host.select()
    client.open_thread.assert_awaited_once_with(
        "lite", thread_id=None, cwd=str(tmp_path)
    )
    assert session.thread_id == "new"
    assert host.is_new
    await host.aclose()
    client.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_rediscovery_changes_endpoint_without_resubmitting_prompt(tmp_path):
    binding = ServiceThread("saved", "lite", str(tmp_path))
    old = Mock(spec=AgentServiceClient)
    old.threads = AsyncMock(return_value=(binding,))
    old.aclose = AsyncMock()
    new = Mock(spec=AgentServiceClient)
    new.threads = AsyncMock(return_value=(binding,))
    new.prompt = AsyncMock()
    new.aclose = AsyncMock()
    connector = AsyncMock(return_value=new)
    host = RemoteCodingHost(
        old, agent_id="lite", workspace=tmp_path, connector=connector
    )
    await host.select("saved")
    restored = await host.reconnect()
    assert restored.thread_id == binding.thread_id
    assert restored.workspace_root == str(tmp_path)
    assert host.client is new
    new.prompt.assert_not_awaited()
    old.aclose.assert_not_awaited()
    await host.aclose()
    new.aclose.assert_awaited_once()
