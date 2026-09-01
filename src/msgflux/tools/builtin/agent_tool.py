from __future__ import annotations

from typing import Any, List, Mapping, Optional

from msgflux.models.gateway import ModelGateway
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
    task_resume_params = ("name", "model")
    task_checkpoint_namespace_param = "name"
    name = "agent"
    display_name = "Agent"
    tool_config = {
        "runtime_inputs": ("handle",),
    }
    description = "Available agents:"
    annotations = {"name": str, "message": str, "return": str}

    def __init__(self):
        self._base_description = type(self).description
        self.refresh()

    def refresh(self) -> None:
        self.description = self._build_description()
        self.usage_guidance = self._build_usage_guidance()

    def patch_schema_annotations(
        self,
        annotations: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        annotations = dict(annotations)
        if self._has_gateway_agents():
            annotations["model"] = Optional[str]
        else:
            annotations.pop("model", None)
        return annotations

    def __call__(
        self,
        name: str,
        message: str,
        model: str | None = None,
        *,
        handle: ToolBucketHandle,
        scope: ExecutionScope | None = None,
    ) -> str:
        self._validate_agent(name, handle)
        model_preference = self._resolve_model_preference(name, model)
        namespace = handle.get_execution_namespace(name)
        runtime_arguments = {"scope": scope or self._build_scope(namespace)}
        if model_preference is not None:
            runtime_arguments["model_preference"] = model_preference
        return handle(
            name,
            message=message,
            _runtime_arguments=runtime_arguments,
        )

    async def acall(
        self,
        name: str,
        message: str,
        model: str | None = None,
        *,
        handle: ToolBucketHandle,
        scope: ExecutionScope | None = None,
    ) -> str:
        self._validate_agent(name, handle)
        model_preference = self._resolve_model_preference(name, model)
        namespace = handle.get_execution_namespace(name)
        runtime_arguments = {"scope": scope or self._build_scope(namespace)}
        if model_preference is not None:
            runtime_arguments["model_preference"] = model_preference
        return await handle.acall(
            name,
            message=message,
            _runtime_arguments=runtime_arguments,
        )

    @staticmethod
    def _validate_agent(name: str, handle: ToolBucketHandle) -> None:
        if handle.has_tool(name):
            return
        available = ", ".join(handle.list_captured_tools()) or "none"
        raise ValueError(f"Agent `{name}` not found. Available agents: {available}.")

    def _resolve_model_preference(
        self,
        agent_name: str,
        model: str | None,
    ) -> str | None:
        if model is None:
            return None
        gateway = self._get_agent_gateway(agent_name)
        if gateway is None:
            return None
        gateway.validate_model_name(model)
        return model

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
            gateway = self._get_agent_gateway(agent_name)
            if gateway is not None:
                agent_lines.append("  Models:")
                for model_name in gateway.model_names:
                    model_description = gateway.get_model_description(model_name)
                    if model_description is None:
                        raise ValueError(
                            f"Agent `{agent_name}` uses ModelGateway model "
                            f"`{model_name}` without a description. Add a non-empty "
                            "`description` to every selectable deployment."
                        )
                    agent_lines.append(f"  - {model_name}: {model_description}")
        if not agent_lines:
            agent_lines.append("- none")
        return f"{self._base_description}\n" + "\n".join(agent_lines)

    def _get_agent_gateway(self, agent_name: str) -> ModelGateway | None:
        metadata = self.tools.get(agent_name)
        if metadata is None:
            return None
        model = getattr(metadata.impl, "model", None)
        return model if isinstance(model, ModelGateway) else None

    def _has_gateway_agents(self) -> bool:
        return any(
            self._get_agent_gateway(agent_name) is not None for agent_name in self.tools
        )

    def _build_usage_guidance(self) -> str | None:
        guidance_lines: List[str] = []
        for agent_name, metadata in sorted(self.tools.items()):
            guidance = metadata.usage_guidance
            if isinstance(guidance, str) and guidance.strip():
                guidance_lines.append(f"- {agent_name}: {' '.join(guidance.split())}")
        if guidance_lines:
            return "Agent-specific guidance:\n" + "\n".join(guidance_lines)
        return None

    def compose_usage_guidance(self, declared: str | None) -> str | None:
        sections = []
        if isinstance(declared, str) and declared.strip():
            sections.append(" ".join(declared.split()))
        if isinstance(self.usage_guidance, str) and self.usage_guidance.strip():
            sections.append(self.usage_guidance)
        return "\n\n".join(sections) or None
