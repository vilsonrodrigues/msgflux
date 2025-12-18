import pytest

from msgflux.dotdict import dotdict
from msgflux.dsl.inline import ainline, inline


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


def test_conditional_with_and_operator(modules):
    """Test conditional with AND operator."""
    expression = "prep -> {output.agent == 'xpto' & output.score >= 10?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


def test_conditional_with_or_operator(modules):
    """Test conditional with OR operator."""
    expression = "prep -> {output.agent == 'unknown' || output.score >= 10?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


def test_conditional_with_not_operator(modules):
    """Test conditional with NOT operator."""
    expression = "prep -> {!(output.agent == 'unknown')?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


def test_conditional_is_none(modules):
    """Test conditional with is None check."""
    def setup(msg: dotdict) -> dotdict:
        msg["value"] = None
        return msg
    
    modules_with_setup = {**modules, "setup": setup}
    expression = "setup -> {value is None?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules_with_setup, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


def test_conditional_is_not_none(modules):
    """Test conditional with is not None check."""
    def setup(msg: dotdict) -> dotdict:
        msg["value"] = "exists"
        return msg
    
    modules_with_setup = {**modules, "setup": setup}
    expression = "setup -> {value is not None?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules_with_setup, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


def test_conditional_with_comparison_operators(modules):
    """Test conditional with different comparison operators."""
    expression = "prep -> {output.score > 5?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"


def test_conditional_less_than(modules):
    """Test conditional with less than operator."""
    expression = "prep -> {output.score < 100?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"


def test_while_loop_with_nested_parallel(modules):
    """Test while loop with nested parallel execution."""
    expression = "prep -> @{counter < 2}: increment -> [feat_a, feat_b]; -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["counter"] == 2
    assert result["feat_a"] == "result_a"
    assert result["feat_b"] == "result_b"


def test_inline_with_dotted_path(modules):
    """Test accessing nested attributes in conditions."""
    expression = "prep -> {output.status == 'success'?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"


def test_parallel_with_multiple_modules(modules):
    """Test parallel execution with more than two modules."""
    def feat_c(msg: dotdict) -> dotdict:
        msg["feat_c"] = "result_c"
        return msg
    
    modules_extended = {**modules, "feat_c": feat_c}
    expression = "prep -> [feat_a, feat_b, feat_c] -> final"
    input_msg = dotdict()
    result = inline(expression, modules_extended, input_msg)
    assert result["feat_a"] == "result_a"
    assert result["feat_b"] == "result_b"
    assert result["feat_c"] == "result_c"


@pytest.mark.asyncio
async def test_async_conditional_with_and_operator(async_modules):
    """Test async conditional with AND operator."""
    expression = "prep -> {output.agent == 'xpto' & output.score >= 10?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


@pytest.mark.asyncio
async def test_async_conditional_with_or_operator(async_modules):
    """Test async conditional with OR operator."""
    expression = "prep -> {output.agent == 'unknown' || output.score >= 10?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


@pytest.mark.asyncio
async def test_async_conditional_is_none(async_modules):
    """Test async conditional with is None check."""
    async def setup(msg: dotdict) -> dotdict:
        msg["value"] = None
        return msg
    
    modules_with_setup = {**async_modules, "setup": setup}
    expression = "setup -> {value is None?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = await ainline(expression, modules_with_setup, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


@pytest.mark.asyncio
async def test_async_parallel_with_multiple_modules(async_modules):
    """Test async parallel execution with multiple modules."""
    async def feat_c(msg: dotdict) -> dotdict:
        msg["feat_c"] = "result_c"
        return msg
    
    modules_extended = {**async_modules, "feat_c": feat_c}
    expression = "prep -> [feat_a, feat_b, feat_c] -> final"
    input_msg = dotdict()
    result = await ainline(expression, modules_extended, input_msg)
    assert result["feat_a"] == "result_a"
    assert result["feat_b"] == "result_b"
    assert result["feat_c"] == "result_c"
