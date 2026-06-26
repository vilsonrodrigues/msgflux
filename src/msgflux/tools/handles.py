from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:
    from msgflux.nn.modules.tool import ToolLibrary


class ToolLibraryHandle:
    """Controlled handle exposed to runtime-aware tools."""

    def __init__(self, library: ToolLibrary):
        self._library = library

    def add(self, tool: Callable) -> str:
        return self._library.add(tool)

    def remove(self, tool_name: str) -> str:
        if tool_name in self._library._runtime_tool_names:
            raise ValueError(f"The runtime tool `{tool_name}` cannot be removed.")
        self._library.remove(tool_name)
        return tool_name

    def list_tools(self) -> List[str]:
        return self._library.get_tool_names()
