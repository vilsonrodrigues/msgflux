from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class ToolSearchRuntime:
    """Runtime tool discovery for on-demand tools."""

    TOOL_NAME = "tool_search"

    def __init__(
        self,
        library: Any,
        *,
        register_tool: Callable[..., None],
        runtime_tool_names: set[str],
    ):
        self.library = library
        self._register_tool = register_tool
        self._runtime_tool_names = runtime_tool_names
        self._enabled = False

    def reset(self) -> None:
        self._enabled = False

    def ensure_tool(self) -> None:
        if self._enabled:
            return
        self._enabled = True
        self._runtime_tool_names.add(self.TOOL_NAME)
        self._register_tool(
            name=self.TOOL_NAME,
            description=(
                "Search registered on-demand tools by keyword or exact "
                "selection. Matching tools become available in the next model "
                "call. Use `select:tool_a,tool_b` for direct selection."
            ),
            annotations={"query": str, "max_results": Optional[int]},
            impl=self.tool_search,
        )

    def sync(self) -> None:
        if self.get_on_demand_tool_names():
            self.ensure_tool()
            return
        if not self._enabled:
            return
        self._enabled = False
        self._runtime_tool_names.discard(self.TOOL_NAME)
        if self.TOOL_NAME in self.library.library:
            self.library.library.pop(self.TOOL_NAME)
        self.library.tool_configs.pop(self.TOOL_NAME, None)

    def get_on_demand_tool_names(self) -> List[str]:
        return list(self.library.on_demand_tools.keys())

    def is_tool_exposed(self, tool_name: str) -> bool:
        return tool_name not in self.library.on_demand_tools

    def load_tools(self, tool_names: List[str]) -> List[str]:
        newly_loaded = []
        for tool_name in tool_names:
            metadata = self.library.on_demand_tools.pop(tool_name, None)
            if metadata is None:
                continue
            metadata.tool_config["on_demand"] = False
            self.library.tool_configs.pop(tool_name, None)
            self.library.add(metadata)
            newly_loaded.append(tool_name)
        self.sync()
        return newly_loaded

    def tool_search(
        self,
        query: str,
        max_results: int | None = 5,
    ) -> Dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("`query` must be a non-empty string.")
        if max_results is not None:
            if isinstance(max_results, bool) or not isinstance(max_results, int):
                raise TypeError(
                    f"`max_results` must be int or None, given `{type(max_results)}`"
                )
            if max_results <= 0:
                raise ValueError("`max_results` must be greater than 0.")

        on_demand_tool_names = self.get_on_demand_tool_names()
        total = len(on_demand_tool_names)
        if total == 0:
            return {
                "query": query,
                "matches": [],
                "loaded": [],
                "already_loaded": [],
                "total_on_demand_tools": 0,
            }

        if query.lower().startswith("select:"):
            requested = [
                item.strip()
                for item in query.split(":", 1)[1].split(",")
                if item.strip()
            ]
            matches = self.select_tools(requested)
        else:
            matches = self.search_tools(
                query=query,
                max_results=max_results or 5,
            )

        newly_loaded = self.load_tools(matches)
        already_loaded = [
            tool_name for tool_name in matches if tool_name not in newly_loaded
        ]
        return {
            "query": query,
            "matches": matches,
            "loaded": newly_loaded,
            "already_loaded": already_loaded,
            "total_on_demand_tools": total,
        }

    def select_tools(self, requested: List[str]) -> List[str]:
        resolved = []
        normalized = {
            tool_name.lower(): tool_name
            for tool_name in self.get_on_demand_tool_names()
        }
        for tool_name in requested:
            match = normalized.get(tool_name.lower())
            if match is not None and match not in resolved:
                resolved.append(match)
        return resolved

    def search_tools(self, *, query: str, max_results: int) -> List[str]:
        query_lower = query.strip().lower()
        terms = [term for term in query_lower.split() if term]
        if not terms:
            return []

        matches = []
        for tool_name in self.get_on_demand_tool_names():
            tool = self.library.on_demand_tools[tool_name]
            name_parts = tool_name.lower().replace("__", " ").replace("_", " ")
            description = (tool.description or "").lower()
            score = 0
            if query_lower == tool_name.lower():
                score += 100
            if query_lower in name_parts:
                score += 40
            for term in terms:
                if term in name_parts:
                    score += 15
                if description and term in description:
                    score += 5
            if score > 0:
                matches.append((score, tool_name))

        matches.sort(key=lambda item: (-item[0], item[1]))
        return [tool_name for _, tool_name in matches[:max_results]]
