
import pytest
from msgflux.dotdict import dotdict
from msgflux.dsl.inline import inline, ainline


@pytest.fixture
def modules():

    def prep(msg: dotdict) -> dotdict:
        msg["output"] = {"agent": "xpto", "score": 10, "status": "success"}
        msg["counter"] = 0
        return msg

    def increment(msg: dotdict) -> dotdict:
        msg["counter"] = msg.get("counter", 0) + 1
        return msg

    def feat_a(msg: dotdict) -> dotdict:
        msg["feat_a"] = "result_a"
        return msg

    def feat_b(msg: dotdict) -> dotdict:
        msg["feat_b"] = "result_b"
        return msg

    def final(msg: dotdict) -> dotdict:
        msg["final"] = "done"
        return msg

    return {
        "prep": prep,
        "increment": increment,
        "feat_a": feat_a,
        "feat_b": feat_b,
        "final": final,
    }


@pytest.fixture
def async_modules():

    async def prep(msg: dotdict) -> dotdict:
        msg["output"] = {"agent": "xpto", "score": 10, "status": "success"}
        msg["counter"] = 0
        return msg

    async def increment(msg: dotdict) -> dotdict:
        msg["counter"] = msg.get("counter", 0) + 1
        return msg

    async def feat_a(msg: dotdict) -> dotdict:
        msg["feat_a"] = "result_a"
        return msg

    async def feat_b(msg: dotdict) -> dotdict:
        msg["feat_b"] = "result_b"
        return msg

    async def final(msg: dotdict) -> dotdict:
        msg["final"] = "done"
        return msg

    return {
        "prep": prep,
        "increment": increment,
        "feat_a": feat_a,
        "feat_b": feat_b,
        "final": final,
    }


def test_simple_sequential_flow(modules):
    expression = "prep -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["final"] == "done"
    assert "output" in result


def test_parallel_flow(modules):
    expression = "prep -> [feat_a, feat_b] -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert result["feat_b"] == "result_b"
    assert result["final"] == "done"


def test_conditional_flow_true(modules):
    expression = "prep -> {output.agent == 'xpto'?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result
    assert result["final"] == "done"


def test_conditional_flow_false(modules):
    expression = "prep -> {output.agent == 'unknown'?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert "feat_a" not in result
    assert result["feat_b"] == "result_b"
    assert result["final"] == "done"


def test_while_loop(modules):
    expression = "prep -> @{counter < 5}: increment; -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["counter"] == 5
    assert result["final"] == "done"


def test_nested_while_loop(modules):
    expression = "prep -> @{counter < 3}: increment -> [feat_a, feat_b]; -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["counter"] == 3
    assert result["feat_a"] == "result_a"
    assert result["feat_b"] == "result_b"
    assert result["final"] == "done"


@pytest.mark.asyncio
async def test_async_simple_sequential_flow(async_modules):
    expression = "prep -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["final"] == "done"
    assert "output" in result


@pytest.mark.asyncio
async def test_async_parallel_flow(async_modules):
    expression = "prep -> [feat_a, feat_b] -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert result["feat_b"] == "result_b"
    assert result["final"] == "done"


@pytest.mark.asyncio
async def test_async_conditional_flow_true(async_modules):
    expression = "prep -> {output.agent == 'xpto'?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result
    assert result["final"] == "done"


@pytest.mark.asyncio
async def test_async_conditional_flow_false(async_modules):
    expression = "prep -> {output.agent == 'unknown'?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert "feat_a" not in result
    assert result["feat_b"] == "result_b"
    assert result["final"] == "done"


@pytest.mark.asyncio
async def test_async_while_loop(async_modules):
    expression = "prep -> @{counter < 5}: increment; -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["counter"] == 5
    assert result["final"] == "done"


@pytest.mark.asyncio
async def test_async_nested_while_loop(async_modules):
    expression = "prep -> @{counter < 3}: increment -> [feat_a, feat_b]; -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["counter"] == 3
    assert result["feat_a"] == "result_a"
    assert result["feat_b"] == "result_b"
    assert result["final"] == "done"
