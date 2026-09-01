from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    TypeVar,
    get_args,
    get_origin,
)

from msgflux.tools.dataclasses import ToolMetadata
from msgflux.tools.helpers import (
    BACKGROUND_TASK_TOOL_KIND,
    DEFAULT_AGENT_BACKGROUND_CAPABILITIES,
    RESERVED_TOOL_KINDS,
    TOOL_BUCKET_KIND,
    is_agent_tool_impl,
    normalize_background_capabilities,
)

T = TypeVar("T")


class Hidden(Generic[T]):
    """Type marker for parameters hidden from the model-facing tool schema."""


def is_hidden_annotation(annotation: Any) -> bool:
    """Return whether an annotation is a `Hidden[...]` marker."""
    return annotation is Hidden or get_origin(annotation) is Hidden


def unwrap_hidden_annotation(annotation: Any) -> Any | None:
    """Return the wrapped type from `Hidden[T]`, or Any for bare `Hidden`."""
    if not is_hidden_annotation(annotation):
        return None
    if annotation is Hidden:
        return Any
    args = get_args(annotation)
    return args[0] if args else Any


class ToolBucket:
    """Base class for tools that capture tools matching a configuration."""

    tool_kind = TOOL_BUCKET_KIND
    capture: Mapping[str, Any] | None = None
    expose_captured_names = False

    def patch_schema_annotations(
        self,
        annotations: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Patch the normalized public annotations after bucket changes."""
        return annotations

    def compose_usage_guidance(self, declared: str | None) -> str | None:
        """Combine stable declaration guidance with dynamic bucket guidance."""
        return getattr(self, "usage_guidance", None) or declared

    def add(self, tool: ToolMetadata) -> None:
        """Store a captured tool and refresh metadata derived from its contents."""
        self.validate_capture(tool)
        if tool.name in self.tools:
            raise ValueError(f"Duplicate tool name `{tool.name}` in bucket.")
        self.tools[tool.name] = tool
        try:
            self.refresh()
        except Exception:
            self.tools.pop(tool.name, None)
            raise

    def remove(self, tool_name: str) -> ToolMetadata:
        """Release a captured tool and refresh metadata derived from its contents."""
        try:
            tool = self.tools.pop(tool_name)
        except KeyError as exc:
            raise ValueError(
                f"Tool `{tool_name}` is not captured by this bucket."
            ) from exc
        try:
            self.refresh()
        except Exception:
            self.tools[tool_name] = tool
            raise
        return tool

    @property
    def tools(self) -> Dict[str, ToolMetadata]:
        if not hasattr(self, "_tools"):
            self._tools = {}
        return self._tools

    def get_ref(self, tool_name: str) -> Any:
        """Return a captured tool reference without exposing its implementation."""
        try:
            metadata = self.tools[tool_name]
        except KeyError as exc:
            raise ValueError(
                f"Tool `{tool_name}` is not captured by this bucket."
            ) from exc
        if metadata.ref is None:
            raise RuntimeError(f"Tool `{tool_name}` has no runtime reference")
        return metadata.ref

    def refresh(self) -> None:
        """Refresh presentation metadata after the library captures a tool."""

    @property
    def capture_rules(self) -> Mapping[str, Any]:
        """Return the validated configuration predicates for this bucket."""
        capture = getattr(self, "capture", None)
        if not isinstance(capture, Mapping) or not capture:
            raise ValueError("A bucket tool must define a non-empty `capture` mapping.")
        if not all(isinstance(key, str) and key for key in capture):
            raise ValueError("Bucket capture keys must be non-empty strings.")
        for key, value in capture.items():
            self._capture_values(key, value)
        return capture

    @staticmethod
    def _capture_values(key: str, value: Any) -> tuple[Any, ...]:
        """Normalize a capture value for matching and overlap validation."""
        if key != "tool_kind":
            return (value,)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "Bucket `capture['tool_kind']` must be a non-empty string."
            )
        values = tuple(part.strip() for part in value.split("|"))
        if not all(values):
            raise ValueError("Bucket `capture['tool_kind']` values cannot be empty.")
        if len(set(values)) != len(values):
            raise ValueError("Bucket `capture['tool_kind']` values must be unique.")
        return values

    def captures_config(self, tool_config: Mapping[str, Any]) -> bool:
        return all(
            tool_config.get(key) in self._capture_values(key, value)
            for key, value in self.capture_rules.items()
        )

    def captures(self, metadata: ToolMetadata) -> bool:
        definition = self._definition_from_metadata(metadata)
        return self.captures_config(definition.declaration)

    @staticmethod
    def _definition_from_metadata(metadata: ToolMetadata) -> Any:
        definition = metadata.definition
        if definition is None:
            raise RuntimeError(
                f"Tool `{metadata.name}` must be compiled before bucket routing."
            )
        return definition

    def validate_capture(self, metadata: ToolMetadata) -> None:
        definition = self._definition_from_metadata(metadata)
        declaration = definition.declaration
        loop_options = {
            option
            for option in (
                "dispatch",
                "background",
                "allow_background",
                "detached",
                "call_as_response",
                "return_direct",
                "handoff",
            )
            if declaration.get(option, False)
        }
        if loop_options:
            formatted_options = ", ".join(
                f"`{option}=True`" for option in sorted(loop_options)
            )
            raise ValueError(
                "Bucket-captured tools cannot define model-loop behavior "
                f"({formatted_options}). Configure that behavior on the public "
                f"bucket instead. Tool `{metadata.name}` cannot be captured."
            )
        if not self.captures(metadata):
            raise ValueError(
                f"Tool `{metadata.name}` does not match this bucket's capture rule."
            )

    @classmethod
    def find_bucket(
        cls,
        metadata: ToolMetadata,
        tools: Mapping[str, Any],
        get_definition: Callable[[str], Any],
    ) -> str | None:
        definition = cls._definition_from_metadata(metadata)
        if definition.kind == cls.tool_kind:
            return None
        for bucket_name, tool in tools.items():
            bucket_definition = get_definition(bucket_name)
            if bucket_definition.kind != cls.tool_kind:
                continue
            bucket = getattr(tool, "impl", tool)
            if isinstance(bucket, cls) and bucket.captures(metadata):
                return bucket_name
        return None

    @classmethod
    def find_capturing_bucket(
        cls,
        tool_name: str,
        tools: Mapping[str, Any],
        get_definition: Callable[[str], Any],
    ) -> str | None:
        for bucket_name, tool in tools.items():
            if get_definition(bucket_name).kind != cls.tool_kind:
                continue
            bucket = getattr(tool, "impl", tool)
            if isinstance(bucket, cls) and tool_name in bucket.tools:
                return bucket_name
        return None

    @classmethod
    def find_capture_candidates(
        cls,
        bucket: ToolBucket,
        tools: Mapping[str, Any],
        get_definition: Callable[[str], Any],
    ) -> list[tuple[str, Any]]:
        candidates = []
        for tool_name, tool in tools.items():
            definition = get_definition(tool_name)
            if definition.kind == cls.tool_kind:
                continue
            if bucket.captures_config(definition.declaration):
                candidates.append((tool_name, tool))
        return candidates

    @classmethod
    def validate_registration(
        cls,
        metadata: ToolMetadata,
        tools: Mapping[str, Any],
        get_definition: Callable[[str], Any],
    ) -> None:
        bucket = metadata.impl
        if not isinstance(bucket, cls):
            raise ValueError(
                f"The bucket tool `{metadata.name}` must inherit ToolBucket."
            )
        capture = bucket.capture_rules
        for bucket_name, tool in tools.items():
            if get_definition(bucket_name).kind != cls.tool_kind:
                continue
            registered_bucket = getattr(tool, "impl", tool)
            if not isinstance(registered_bucket, cls):
                continue
            if cls._captures_overlap(capture, registered_bucket.capture_rules):
                raise ValueError(
                    f"The bucket capture for `{metadata.name}` overlaps with "
                    f"`{bucket_name}`."
                )

    @classmethod
    def _captures_overlap(
        cls,
        first: Mapping[str, Any],
        second: Mapping[str, Any],
    ) -> bool:
        """Return whether two capture rules can match the same configuration."""
        for key in first.keys() & second.keys():
            first_values = cls._capture_values(key, first[key])
            second_values = cls._capture_values(key, second[key])
            if not any(
                first_value == second_value
                for first_value in first_values
                for second_value in second_values
            ):
                return False
        return True


class ToolLibraryOperator:
    """Base class for tools that operate through ToolLibraryHandle."""

    tool_config = {"runtime_inputs": ("handle",)}

    @classmethod
    def is_operator_tool(cls, tool: Any | None) -> bool:
        if tool is None:
            return False
        impl = getattr(tool, "impl", tool)
        return isinstance(impl, cls)


class ToolBackground(ToolLibraryOperator):
    """Base class for builtin tools that manage background tasks."""

    tool_kind = BACKGROUND_TASK_TOOL_KIND

    @classmethod
    def is_active_task_tool(
        cls,
        *,
        library: Any,
        tool_name: str,
        definition: Any,
        base_tools: Iterable[Callable],
        capability_tools: Mapping[str, Iterable[Callable]],
        metadata_factory: Callable[[Callable], ToolMetadata],
    ) -> bool:
        if not cls.is_reserved_definition(definition):
            return False
        background_tools = list(cls._iter_background_tools(library))
        if not background_tools:
            return False

        capabilities = {
            capability
            for source_definition in background_tools
            for capability in cls.get_background_capabilities(source_definition)
        }
        task_tools = cls._task_tools_for_capabilities(
            base_tools=base_tools,
            capability_tools=capability_tools,
            capabilities=capabilities,
            metadata_factory=metadata_factory,
        )
        return tool_name in {
            metadata_factory(task_tool).name for task_tool in task_tools
        }

    @staticmethod
    def is_agent_source(tool: Any | None) -> bool:
        return is_agent_tool_impl(getattr(tool, "impl", tool))

    @staticmethod
    def is_reserved_definition(definition: Any) -> bool:
        return definition.kind in RESERVED_TOOL_KINDS

    @staticmethod
    def is_background_definition(definition: Any) -> bool:
        return definition.dispatch.name in {"background", "optional_background"}

    @classmethod
    def get_background_capabilities(
        cls,
        definition: Any,
    ) -> tuple[str, ...]:
        if not cls.is_background_definition(definition):
            return ()
        declared_capabilities = definition.dispatch.options.get("capabilities")
        if declared_capabilities is None:
            if cls.is_agent_source(definition.executor):
                return DEFAULT_AGENT_BACKGROUND_CAPABILITIES
            return ()
        capabilities = normalize_background_capabilities(declared_capabilities)
        agent_capabilities = {"message"}
        if agent_capabilities.intersection(capabilities) and not cls.is_agent_source(
            definition.executor
        ):
            raise ValueError(
                "`message` background capability is currently only supported by "
                "agent sources."
            )
        return capabilities

    @classmethod
    def validate_background_capabilities(
        cls,
        definition: Any,
    ) -> None:
        cls.get_background_capabilities(definition)

    @classmethod
    def sync_task_tools(
        cls,
        *,
        library: Any,
        disabled_tool_names: set[str],
        base_tools: Iterable[Callable],
        capability_tools: Mapping[str, Iterable[Callable]],
        metadata_factory: Callable[[Callable], ToolMetadata],
    ) -> None:
        background_tools = list(cls._iter_background_tools(library))
        all_task_tools = cls._all_task_tools(
            base_tools=base_tools,
            capability_tools=capability_tools,
            metadata_factory=metadata_factory,
        )
        if background_tools:
            capabilities = {
                capability
                for definition in background_tools
                for capability in cls.get_background_capabilities(definition)
            }
            required_task_tools = cls._task_tools_for_capabilities(
                base_tools=base_tools,
                capability_tools=capability_tools,
                capabilities=capabilities,
                metadata_factory=metadata_factory,
            )
            cls._ensure_task_tools(
                library=library,
                disabled_tool_names=disabled_tool_names,
                tools=required_task_tools,
                metadata_factory=metadata_factory,
            )
            required_names = {
                metadata_factory(task_tool).name for task_tool in required_task_tools
            }
            cls._remove_task_tools(
                library=library,
                tools=(
                    task_tool
                    for task_tool in all_task_tools
                    if metadata_factory(task_tool).name not in required_names
                ),
                metadata_factory=metadata_factory,
            )
            return

        cls._remove_task_tools(
            library=library,
            tools=all_task_tools,
            metadata_factory=metadata_factory,
        )
        disabled_tool_names.clear()

    @classmethod
    def _iter_background_tools(
        cls,
        library: Any,
    ) -> Iterator[Any]:
        for tool_name in library.library:
            definition = library.get_tool_definition(tool_name)
            if cls.is_reserved_definition(definition):
                continue
            if cls.is_background_definition(definition):
                yield definition

    @classmethod
    def _all_task_tools(
        cls,
        *,
        base_tools: Iterable[Callable],
        capability_tools: Mapping[str, Iterable[Callable]],
        metadata_factory: Callable[[Callable], ToolMetadata],
    ) -> tuple[Callable, ...]:
        return cls._task_tools_for_capabilities(
            base_tools=base_tools,
            capability_tools=capability_tools,
            capabilities=capability_tools.keys(),
            metadata_factory=metadata_factory,
        )

    @classmethod
    def _task_tools_for_capabilities(
        cls,
        *,
        base_tools: Iterable[Callable],
        capability_tools: Mapping[str, Iterable[Callable]],
        capabilities: Iterable[str],
        metadata_factory: Callable[[Callable], ToolMetadata],
    ) -> tuple[Callable, ...]:
        selected_tools = list(base_tools)
        capability_names = set(capabilities)
        for capability, tools in capability_tools.items():
            if capability in capability_names:
                selected_tools.extend(tools)

        unique_tools: Dict[str, Callable] = {}
        for task_tool in selected_tools:
            metadata = metadata_factory(task_tool)
            unique_tools.setdefault(metadata.name, task_tool)
        return tuple(unique_tools.values())

    @classmethod
    def _ensure_task_tools(
        cls,
        *,
        library: Any,
        disabled_tool_names: set[str],
        tools: Iterable[Callable],
        metadata_factory: Callable[[Callable], ToolMetadata],
    ) -> None:
        for tool in tools:
            metadata = metadata_factory(tool)
            tool_name = metadata.name
            if tool_name in disabled_tool_names:
                continue
            if tool_name in library.library:
                existing = library.get_tool_definition(tool_name)
                if not cls.is_reserved_definition(existing):
                    raise ValueError(
                        f"The background task tool `{tool_name}` conflicts with "
                        "an existing tool."
                    )
                continue
            library.add(metadata)

    @classmethod
    def _remove_task_tools(
        cls,
        *,
        library: Any,
        tools: Iterable[Callable],
        metadata_factory: Callable[[Callable], ToolMetadata],
    ) -> None:
        for tool in tools:
            tool_name = metadata_factory(tool).name
            if tool_name in library.library and cls.is_reserved_definition(
                library.get_tool_definition(tool_name)
            ):
                library._remove_registered_tool(tool_name)
