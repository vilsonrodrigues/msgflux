from __future__ import annotations

from typing import List

from msgflux.runtime.context import (
    ExecutionScope,
    get_execution_context,
    new_thread_id,
)
from msgflux.tools.handles import ToolBucketHandle
from msgflux.tools.types import ToolBucket, ToolLibraryOperator


class AgentTool(ToolBucket, ToolLibraryOperator):
    """Dispatch a message to one of the configured agents."""

    capture = {"tool_kind": "agent", "defer_loading": False}
    task_resume_params = ("name",)
    task_checkpoint_namespace_param = "name"
    name = "agent"
    display_name = "Agent"
    tool_config = {
        "inject_handle": True,
    }
    description = "Available agents:"
    annotations = {"name": str, "message": str, "return": str}

    def __init__(self):
        self._base_description = type(self).description
        self.refresh()

    def refresh(self) -> None:
        self.description = self._build_description()
        self.usage_guidance = self._build_usage_guidance()

    def __call__(
        self,
        name: str,
        message: str,
        *,
        handle: ToolBucketHandle,
        scope: ExecutionScope | None = None,
    ) -> str:
        self._validate_agent(name, handle)
        namespace = handle.get_execution_namespace(name)
        return handle(
            name,
            message=message,
            scope=scope or self._build_scope(namespace),
        )

    async def acall(
        self,
        name: str,
        message: str,
        *,
        handle: ToolBucketHandle,
        scope: ExecutionScope | None = None,
    ) -> str:
        self._validate_agent(name, handle)
        namespace = handle.get_execution_namespace(name)
        return await handle.acall(
            name,
            message=message,
            scope=scope or self._build_scope(namespace),
        )

    @staticmethod
    def _validate_agent(name: str, handle: ToolBucketHandle) -> None:
        if handle.has_tool(name):
            return
        available = ", ".join(handle.list_captured_tools()) or "none"
        raise ValueError(f"Agent `{name}` not found. Available agents: {available}.")

    def _build_scope(self, agent_name: str) -> ExecutionScope:
        context = get_execution_context()
        parent_scope = context["scope"]
        thread_id = context.get("thread_id")
        task_handle = context.get("task_handle")
        run_id = getattr(task_handle, "task_id", None) or context.get("run_id")
        parent_run_id = context.get("parent_run_id")
        if parent_run_id is None and context.get("run_id") != run_id:
            parent_run_id = context.get("run_id")

        return parent_scope.with_overrides(
            namespace=agent_name,
            thread_id=thread_id if isinstance(thread_id, str) else new_thread_id(),
            run_id=run_id if isinstance(run_id, str) else None,
            parent_run_id=parent_run_id if isinstance(parent_run_id, str) else None,
            root_run_id=(
                context.get("root_run_id")
                if isinstance(context.get("root_run_id"), str)
                else None
            ),
        )

    def _build_description(self) -> str:
        agent_lines: List[str] = []
        for agent_name, metadata in sorted(self.tools.items()):
            description = metadata.description
            if description:
                agent_lines.append(f"- {agent_name}: {description}")
            else:
                agent_lines.append(f"- {agent_name}")
        if not agent_lines:
            agent_lines.append("- none")
        return f"{self._base_description}\n" + "\n".join(agent_lines)

    def _build_usage_guidance(self) -> str | None:
        guidance_sections: List[str] = []
        configured_guidance = self.tool_config.get("usage_guidance")
        if isinstance(configured_guidance, str) and configured_guidance.strip():
            guidance_sections.append(" ".join(configured_guidance.split()))

        guidance_lines: List[str] = []
        for agent_name, metadata in sorted(self.tools.items()):
            guidance = metadata.usage_guidance
            if isinstance(guidance, str) and guidance.strip():
                guidance_lines.append(f"- {agent_name}: {' '.join(guidance.split())}")
        if guidance_lines:
            guidance_sections.append(
                "Agent-specific guidance:\n" + "\n".join(guidance_lines)
            )
        return "\n\n".join(guidance_sections) or None
