"""Opt-in subscription smoke through the real daemon and Textual frontend."""

import asyncio
import os
import signal

import pytest
from textual.widgets import Static, TextArea

from msgflux.runtime.service.http import AgentSessionClient
from msgflux.coding.tui import CodingApp
from msgflux.runtime.service.local import connect_local_service
from msgflux.runtime.service.local.files import read_record

pytestmark = pytest.mark.skipif(
    os.environ.get("MSGFLUX_LIVE_CODING") != "1" or os.name != "posix",
    reason="Opt-in Codex subscription test with a local POSIX daemon",
)


@pytest.mark.asyncio
async def test_live_codex_background_bash_is_rendered_by_remote_tui(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    runtime = tmp_path / "state" / "runtime"
    client = await connect_local_service(
        "msgflux.coding.remote_backend:create_service", runtime_dir=runtime, cwd=root
    )
    record = read_record(runtime)
    try:
        session = await AgentSessionClient.open(client, agent_id="lite", cwd=root)
        app = CodingApp(session, observe_immediately=False)
        async with app.run_test() as pilot:
            app.query_one("#composer", TextArea).load_text(
                "Do not inspect other files. Run bash in background with the exact "
                "command `sleep 0.2; printf ui-smoke`. Retrieve its completed output "
                "with the task tool. Then answer only ui-smoke."
            )
            await pilot.press("enter")
            async with asyncio.timeout(120):
                while not (await session.runs()):
                    await asyncio.sleep(0.1)
                while app._run_active or app._active_run_id is not None:
                    await asyncio.sleep(0.1)
                while "Assistant: ui-smoke" not in _text(app):
                    if "Run failed" in str(app.query_one("#status", Static).content):
                        pytest.fail(_text(app))
                    await asyncio.sleep(0.1)
            text = _text(app)
            assert "bash" in text.lower()
            assert "task" in text.lower()
            assert "ui-smoke" in text
            assert (await session.latest_run()).status == "completed"
    finally:
        await client.aclose()
        os.kill(record.pid, signal.SIGTERM)
        async with asyncio.timeout(10):
            while (runtime / "daemon.json").exists():
                await asyncio.sleep(0.05)


def _text(app):
    return "\n".join(
        str(row.content) for row in app.query_one("#transcript").query(Static)
    )
