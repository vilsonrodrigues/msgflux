import pytest

from msgflux.nn.modules.tool import ToolLibrary
from msgflux.nn.modules.tool_v2 import ToolRef
from msgflux.tools.config import tool_config
from msgflux.tools.types import ToolBucket, ToolLibraryOperator


def test_handle_executes_stable_tool_ref_through_library_pipeline():
    def double(value: int) -> int:
        """Double a value."""
        return value * 2

    library = ToolLibrary(name="math", tools=[double])
    ref = library.get_tool_ref("double")

    result = library.get_handle()(ref, value=3)

    assert ref == ToolRef(library_id=library.name, tool_id="double")
    assert result == 6


@pytest.mark.asyncio
async def test_handle_rejects_reference_owned_by_another_library():
    def double(value: int) -> int:
        """Double a value."""
        return value * 2

    library = ToolLibrary(name="math", tools=[double])
    foreign = ToolRef(library_id="foreign", tool_id="double")

    with pytest.raises(ValueError, match="belongs to `foreign`"):
        await library.get_handle().acall(foreign, value=3)


def test_bucket_handle_resolves_captured_tool_by_ref_without_accessing_impl():
    class WorkerBucket(ToolBucket, ToolLibraryOperator):
        """Proxy worker tools."""

        name = "worker"
        capture = {"tool_kind": "worker", "defer_loading": False}

        def __call__(self) -> str:
            return "worker"

    @tool_config(tool_kind="worker")
    def double(value: int) -> int:
        """Double a value."""
        return value * 2

    library = ToolLibrary(name="math", tools=[WorkerBucket(), double])
    bucket = library.library["worker"].impl
    ref = bucket.get_ref("double")
    handle = library.get_handle().for_tool(tool_name="worker")

    result = handle(ref, value=4)

    assert ref == library.get_tool_ref("double")
    assert result == 8
