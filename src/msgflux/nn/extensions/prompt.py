"""Built-in extensions for optional system-prompt capabilities."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from msgflux.nn.extensions.base import AgentExtension
from msgflux.nn.hooks import Hook, ModelContext
from msgflux.utils.time import utc_current_date

__all__ = ["CurrentDateExtension", "ToolUsageGuidanceExtension"]


def _append_section(ctx: ModelContext, section: str) -> ModelContext:
    prompt = f"{ctx.prompt}\n{section}" if ctx.prompt else section
    return replace(ctx, prompt=prompt)


class CurrentDateExtension(AgentExtension):
    """Add the current UTC date to the rendered system prompt."""

    def __init__(self, date_factory: Callable[[], str] = utc_current_date) -> None:
        super().__init__("current_date")
        self.date_factory = date_factory

    def hooks(self):
        return (Hook(event="transform_system_prompt", handler=self._add_date),)

    def _add_date(self, ctx: ModelContext) -> ModelContext:
        return _append_section(ctx, f"The current date is: {self.date_factory()}")


class ToolUsageGuidanceExtension(AgentExtension):
    """Render guidance owned by tools in the active request catalog."""

    def __init__(self) -> None:
        super().__init__("tool_usage_guidance")

    def hooks(self):
        return (Hook(event="transform_system_prompt", handler=self._add_guidance),)

    def _add_guidance(self, ctx: ModelContext) -> ModelContext:
        guidance = self.agent.tool_library.get_tool_usage_guidance(ctx.tool_names)
        if not guidance:
            return ctx
        lines = ["<tool_usage_guidance>"]
        for tool in guidance:
            lines.extend(
                [
                    f'<tool name="{tool["name"]}">',
                    tool["guidance"],
                    "</tool>",
                ]
            )
        lines.append("</tool_usage_guidance>")
        return _append_section(ctx, "\n".join(lines))
