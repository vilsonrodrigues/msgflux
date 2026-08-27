import asyncio
from dataclasses import replace

import pytest

from msgflux.nn import ToolLibrary
from msgflux.nn.extensions import ToolLibraryExtension
from msgflux.nn.hooks import Hook


class MathExtension(ToolLibraryExtension):
    def __init__(self):
        super().__init__("math")

    def tools(self):
        def double(value: int) -> int:
            """Double a value."""
            return value * 2

        return (double,)

    def hooks(self):
        return (
            Hook(
                event="before_tool",
                handler=lambda event: replace(
                    event,
                    arguments={"value": event.arguments["value"] + 1},
                ),
            ),
            Hook(
                event="after_tool",
                handler=lambda event: replace(event, result=event.result + 1),
            ),
        )


def test_tool_library_extension_contributes_tools_and_lifecycle_hooks():
    library = ToolLibrary("math", [], extensions=[MathExtension()])

    response = library([("call_1", "double", {"value": 2})])

    assert response.tool_calls[0].parameters == {"value": 3}
    assert response.tool_calls[0].result == 7


def test_tool_library_extension_handle_removes_owned_contributions():
    library = ToolLibrary("math", [])
    handle = library.register_extension("math", MathExtension())

    assert handle.active
    assert "double" in library.get_tool_names()

    handle.remove()

    assert not handle.active
    assert "double" not in library.get_tool_names()
    response = library([("call_1", "double", {"value": 2})])
    assert response.tool_calls[0].error == "Error: Tool `double` not found."


@pytest.mark.asyncio
async def test_tool_library_extension_supports_async_hooks_and_cleanup():
    cleaned = asyncio.Event()

    class AsyncExtension(MathExtension):
        async def increase(self, event):
            await asyncio.sleep(0)
            return replace(
                event,
                arguments={"value": event.arguments["value"] + 2},
            )

        def hooks(self):
            return (Hook(event="before_tool", handler=self.increase),)

        async def aon_remove(self, _library):
            cleaned.set()

    library = ToolLibrary("math", [])
    handle = library.register_extension("math", AsyncExtension())

    response = await library.acall([("call_1", "double", {"value": 2})])
    assert response.tool_calls[0].result == 8

    await handle.aremove()
    assert cleaned.is_set()


def test_deferred_tools_install_tool_search_as_an_extension():
    def lookup(query: str) -> str:
        """Look up a value."""
        return query

    lookup.tool_config = {"defer_loading": True}
    library = ToolLibrary("search", [lookup])

    assert library.has_extension("tool_search")
    assert library.bucket_has_tool("tool_search", "lookup")

    library.remove("lookup")
    library.add(lookup)

    assert library.bucket_has_tool("tool_search", "lookup")


def test_background_controls_are_managed_by_builtin_extension():
    def long_job() -> str:
        """Run a background job."""
        return "done"

    long_job.tool_config = {"background": True}
    library = ToolLibrary("tasks", [long_job])

    assert library.has_extension("background_tasks")
    assert "task_status" in library.get_tool_names()
