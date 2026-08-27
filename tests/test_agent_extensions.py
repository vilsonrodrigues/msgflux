import asyncio
from dataclasses import replace

import pytest

from msgflux.exceptions import AbortRequestedError
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent, AgentExtension
from msgflux.nn.hooks import Hook, ModelContext
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime import AbortSignal, ExecutionScope
from msgflux.runtime.context import execution_context


def _text_response(text: str) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    response.reasoning = None
    response.metadata = {}
    return response


class _RecordingModel:
    model_type = "chat_completion"

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _text_response("ok")

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        return _text_response("ok")


class _PromptExtension(AgentExtension):
    def __init__(self):
        super().__init__("prompt")

    def hooks(self):
        def add_prompt(ctx: ModelContext):
            return replace(ctx, prompt=f"{ctx.prompt}\nextension prompt".strip())

        return (Hook(event="transform_system_prompt", handler=add_prompt),)

    def tools(self):
        def extension_tool(value: str) -> str:
            """Return the supplied value."""
            return value

        return (extension_tool,)


def test_extension_contributes_and_removes_hooks_and_tools():
    model = _RecordingModel()
    agent = Agent(name="agent", model=model)

    handle = agent.register_extension("prompt", _PromptExtension())

    assert handle.active
    assert "extension prompt" in agent.get_system_prompt()
    assert "extension_tool" in agent.tool_library.library

    handle.remove()

    assert not handle.active
    assert "extension prompt" not in agent.get_system_prompt()
    assert "extension_tool" not in agent.tool_library.library


@pytest.mark.asyncio
async def test_extension_removal_preserves_active_run_snapshot():
    started = asyncio.Event()
    release = asyncio.Event()
    model = _RecordingModel()

    class BlockingExtension(_PromptExtension):
        async def add_prompt(self, ctx: ModelContext):
            started.set()
            await release.wait()
            return replace(ctx, prompt=f"{ctx.prompt}\nactive snapshot".strip())

        def hooks(self):
            return (Hook(event="transform_system_prompt", handler=self.add_prompt),)

    agent = Agent(name="agent", model=model)
    handle = agent.register_extension("prompt", BlockingExtension())

    active_run = asyncio.create_task(agent.acall("first"))
    await started.wait()
    handle.remove()

    assert await agent.acall("second") == "ok"
    assert "active snapshot" not in (model.calls[0]["system_prompt"] or "")
    assert model.calls[0]["tool_catalog"] is None

    release.set()
    assert await active_run == "ok"
    assert "active snapshot" in (model.calls[1]["system_prompt"] or "")
    assert "extension_tool" not in agent.tool_library.library


@pytest.mark.asyncio
async def test_async_extension_removal_waits_for_active_run_cleanup():
    started = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()

    class AsyncCleanupExtension(AgentExtension):
        def __init__(self):
            super().__init__("cleanup")

        async def block(self, ctx):
            started.set()
            await release.wait()
            return ctx

        def hooks(self):
            return (Hook(event="transform_system_prompt", handler=self.block),)

        async def aon_remove(self, agent):
            cleaned.set()

    agent = Agent(name="agent", model=_RecordingModel())
    handle = agent.register_extension("cleanup", AsyncCleanupExtension())
    run = asyncio.create_task(agent.acall("first"))
    await started.wait()

    removal = asyncio.create_task(handle.aremove())
    await asyncio.sleep(0)
    assert not cleaned.is_set()

    release.set()
    assert await run == "ok"
    await removal
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_abort_cancels_async_extension_hook():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class NetworkExtension(AgentExtension):
        def __init__(self):
            super().__init__("network")

        async def analyze(self, payload):
            started.set()
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.set()
            return payload

        def hooks(self):
            return (Hook(event="before_run", handler=self.analyze),)

    signal = AbortSignal()
    agent = Agent(
        name="agent",
        model=_RecordingModel(),
        extensions=[NetworkExtension()],
    )
    run = asyncio.create_task(
        agent.acall("hello", scope=ExecutionScope(abort_signal=signal))
    )
    await started.wait()

    signal.abort("test cancellation")

    with pytest.raises(AbortRequestedError, match="test cancellation"):
        await run
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_abort_reaches_transform_output_after_agent_forward():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class OutputExtension(AgentExtension):
        def __init__(self):
            super().__init__("output")

        async def transform(self, output):
            started.set()
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.set()
            return output

        def hooks(self):
            return (Hook(event="transform_output", handler=self.transform),)

    signal = AbortSignal()
    agent = Agent(
        name="agent",
        model=_RecordingModel(),
        extensions=[OutputExtension()],
    )
    run = asyncio.create_task(
        agent.acall("hello", scope=ExecutionScope(abort_signal=signal))
    )
    await started.wait()

    signal.abort("stop output")

    with pytest.raises(AbortRequestedError, match="stop output"):
        await run
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_agent_output_context_exposes_runtime_vars_and_scope():
    signal = AbortSignal()
    scope = ExecutionScope(abort_signal=signal)

    def transform(ctx):
        assert ctx.vars == {"tenant": "acme"}
        assert ctx.scope.abort_signal is signal
        return replace(ctx, output=f"{ctx.output}:{ctx.vars['tenant']}")

    agent = Agent(
        name="agent",
        model=_RecordingModel(),
        hooks=[Hook(event="transform_output", handler=transform)],
    )

    assert await agent.acall("hello", vars={"tenant": "acme"}, scope=scope) == "ok:acme"


@pytest.mark.asyncio
async def test_extension_state_is_shared_within_run_and_isolated_between_runs():
    class StatefulExtension(AgentExtension):
        def __init__(self):
            super().__init__("stateful")

        def remember(self, payload):
            self.state()["message"] = payload.message
            return payload

        def add_prompt(self, ctx: ModelContext):
            message = self.state()["message"]
            return replace(ctx, prompt=f"{ctx.prompt}\nrun:{message}".strip())

        def hooks(self):
            return (
                Hook(event="before_run", handler=self.remember),
                Hook(event="transform_system_prompt", handler=self.add_prompt),
            )

    model = _RecordingModel()
    agent = Agent(
        name="agent",
        model=model,
        extensions=[StatefulExtension()],
    )

    assert await asyncio.gather(agent.acall("one"), agent.acall("two")) == [
        "ok",
        "ok",
    ]
    assert {call["system_prompt"] for call in model.calls} == {"run:one", "run:two"}


@pytest.mark.asyncio
async def test_abort_cancels_foreground_async_tool():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_tool() -> str:
        """Wait until the caller cancels this tool."""
        started.set()
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()
        return "late"

    signal = AbortSignal()
    library = ToolLibrary("agent", [slow_tool])
    with execution_context(scope=ExecutionScope(abort_signal=signal)):
        run = asyncio.create_task(library.acall([("call_1", "slow_tool", {})]))
        await started.wait()
        signal.abort("stop tool")
        with pytest.raises(AbortRequestedError, match="stop tool"):
            await run

    assert cancelled.is_set()
