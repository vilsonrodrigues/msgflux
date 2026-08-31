"""Incubating contracts for the extensible ToolLibrary runtime.

This module intentionally runs beside ``tool.py`` until the V2 pipeline reaches
behavioral parity. It is not exported as public API during that transition.
"""

from __future__ import annotations

import asyncio
import weakref
from copy import deepcopy
from typing import (
    Any,
    Awaitable,
    Callable,
    Collection,
    Mapping,
    Protocol,
    runtime_checkable,
)

import msgspec

import msgflux.nn.functional as F
from msgflux.exceptions import AbortRequestedError, TaskInterruptRequestedError
from msgflux.nn.modules.container import ModuleDict
from msgflux.nn.modules.module import Module
from msgflux.runtime.abort import await_with_abort
from msgflux.tools.dataclasses import ToolMetadata
from msgflux.tools.helpers import RUNTIME_BACKGROUND_PARAM
from msgflux.tools.runtime import FeedbackSpec, ToolError, ToolIntent, ToolOutcome


def _require_name(value: Any, subject: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{subject}` must be a non-empty string")
    return value


def _copy_mapping(value: Mapping[str, Any], subject: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"`{subject}` must be a mapping")
    return deepcopy(dict(value))


@runtime_checkable
class ToolExecutor(Protocol):
    """Execution adapter owned by one logical tool definition.

    Local Python and remote MCP tools implement the same boundary. Dispatch
    extensions decide *when* the adapter runs; the adapter decides *how* the
    action reaches its implementation.
    """

    def __call__(self, **arguments: Any) -> Any: ...

    async def acall(self, **arguments: Any) -> Any: ...


class DispatchSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Open dispatch selection compiled from one tool declaration."""

    name: str = "foreground"
    options: Mapping[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        msgspec.structs.force_setattr(
            self, "name", _require_name(self.name, "dispatch.name")
        )
        msgspec.structs.force_setattr(
            self,
            "options",
            _copy_mapping(self.options, "dispatch.options"),
        )

    @classmethod
    def coerce(cls, value: DispatchSpec | str | None) -> DispatchSpec:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(name=value)
        raise TypeError("`dispatch` must be a DispatchSpec, string, or None")


class ContextBinding(msgspec.Struct, frozen=True, kw_only=True):
    """Bind one named runtime source to one hidden tool parameter."""

    source: str
    parameter: str | None = None
    required: bool = True
    options: Mapping[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        source = _require_name(self.source, "context.source")
        parameter = self.parameter if self.parameter is not None else source
        msgspec.structs.force_setattr(self, "source", source)
        msgspec.structs.force_setattr(
            self,
            "parameter",
            _require_name(parameter, "context.parameter"),
        )
        if not isinstance(self.required, bool):
            raise TypeError("`context.required` must be a bool")
        msgspec.structs.force_setattr(
            self,
            "options",
            _copy_mapping(self.options, "context.options"),
        )

    @classmethod
    def coerce(cls, value: ContextBinding | str) -> ContextBinding:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(source=value)
        raise TypeError("Context bindings must be ContextBinding instances or strings")


class ContextSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Explicit runtime values made available to one tool."""

    bindings: tuple[ContextBinding | str, ...] = ()

    def __post_init__(self) -> None:
        bindings = tuple(ContextBinding.coerce(item) for item in self.bindings)
        parameters = [binding.parameter for binding in bindings]
        if len(parameters) != len(set(parameters)):
            raise ValueError("Context bindings must target unique parameters")
        msgspec.structs.force_setattr(self, "bindings", bindings)

    @classmethod
    def coerce(
        cls,
        value: ContextSpec | Collection[ContextBinding | str] | None,
    ) -> ContextSpec:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str) or not isinstance(value, Collection):
            raise TypeError("`context` must be a ContextSpec or collection of bindings")
        return cls(bindings=tuple(value))


class LoadingSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Catalog visibility policy independent from mutable thread state."""

    deferred: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.deferred, bool):
            raise TypeError("`loading.deferred` must be a bool")


class NativeToolBinding(msgspec.Struct, frozen=True, kw_only=True):
    """Provider-native representation supported by a logical tool."""

    provider: str
    api_mode: str
    kind: str
    execution: str
    options: Mapping[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("provider", "api_mode", "kind"):
            msgspec.structs.force_setattr(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        if self.execution not in {"client", "provider"}:
            raise ValueError("`execution` must be `client` or `provider`")
        msgspec.structs.force_setattr(
            self,
            "options",
            _copy_mapping(self.options, "native_binding.options"),
        )


class ToolRef(msgspec.Struct, frozen=True, kw_only=True):
    """Stable reference used by buckets and handles without exposing internals."""

    library_id: str
    tool_id: str

    def __post_init__(self) -> None:
        msgspec.structs.force_setattr(
            self,
            "library_id",
            _require_name(self.library_id, "library_id"),
        )
        msgspec.structs.force_setattr(
            self,
            "tool_id",
            _require_name(self.tool_id, "tool_id"),
        )


class ToolChoice(msgspec.Struct, frozen=True, kw_only=True):
    """Provider-neutral catalog selection policy."""

    mode: str = "auto"
    name: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "none", "required", "tool"}:
            raise ValueError("`choice.mode` must be auto, none, required, or tool")
        if self.mode == "tool":
            msgspec.structs.force_setattr(
                self,
                "name",
                _require_name(self.name, "choice.name"),
            )
        elif self.name is not None:
            raise ValueError("`choice.name` is only valid when mode is `tool`")

    @classmethod
    def coerce(cls, value: ToolChoice | str | None) -> ToolChoice:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or not value.strip():
            raise TypeError("`choice` must be a ToolChoice, string, or None")
        if value in {"auto", "none", "required"}:
            return cls(mode=value)
        return cls(mode="tool", name=value)


class ToolDefinition(msgspec.Struct, frozen=True, kw_only=True):
    """Stable logical declaration compiled before a tool can be executed."""

    name: str
    executor: ToolExecutor
    input_schema: Mapping[str, Any]
    description: str | None = None
    annotations: Mapping[str, Any] = msgspec.field(default_factory=dict)
    dispatch: DispatchSpec | str = msgspec.field(default_factory=DispatchSpec)
    feedback: FeedbackSpec | str = msgspec.field(default_factory=FeedbackSpec)
    context: ContextSpec | tuple[ContextBinding | str, ...] = msgspec.field(
        default_factory=ContextSpec
    )
    loading: LoadingSpec = msgspec.field(default_factory=LoadingSpec)
    retry: Any = None
    native_bindings: tuple[NativeToolBinding, ...] = ()
    kind: str = "tool"
    display_name: str | None = None
    usage_guidance: str | None = None
    metadata: Mapping[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        msgspec.structs.force_setattr(self, "name", _require_name(self.name, "name"))
        if not isinstance(self.executor, ToolExecutor):
            raise TypeError(
                "`executor` must provide synchronous `__call__` and async `acall`"
            )
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("`description` must be a string or None")
        msgspec.structs.force_setattr(self, "kind", _require_name(self.kind, "kind"))
        msgspec.structs.force_setattr(
            self, "dispatch", DispatchSpec.coerce(self.dispatch)
        )
        msgspec.structs.force_setattr(
            self, "feedback", FeedbackSpec.coerce(self.feedback)
        )
        msgspec.structs.force_setattr(self, "context", ContextSpec.coerce(self.context))
        if not isinstance(self.loading, LoadingSpec):
            raise TypeError("`loading` must be a LoadingSpec")
        native_bindings = tuple(self.native_bindings)
        if not all(
            isinstance(binding, NativeToolBinding) for binding in native_bindings
        ):
            raise TypeError("`native_bindings` must contain NativeToolBinding values")
        native_keys = [
            (binding.provider, binding.api_mode, binding.kind)
            for binding in native_bindings
        ]
        if len(native_keys) != len(set(native_keys)):
            raise ValueError(
                "Native tool bindings must be unique per provider/API/kind"
            )
        msgspec.structs.force_setattr(self, "native_bindings", native_bindings)
        msgspec.structs.force_setattr(
            self,
            "input_schema",
            _copy_mapping(self.input_schema, "input_schema"),
        )
        msgspec.structs.force_setattr(
            self,
            "annotations",
            _copy_mapping(self.annotations, "annotations"),
        )
        msgspec.structs.force_setattr(
            self,
            "metadata",
            _copy_mapping(self.metadata, "metadata"),
        )


class ToolDefinitionCompiler:
    """Compile legacy decorator metadata once into the canonical contract."""

    @classmethod
    def compile(
        cls,
        metadata: ToolMetadata,
        *,
        executor: ToolExecutor,
    ) -> ToolDefinition:
        if not isinstance(metadata, ToolMetadata):
            raise TypeError("`metadata` must be ToolMetadata")
        if not isinstance(executor, ToolExecutor):
            raise TypeError("`executor` must implement ToolExecutor")
        config = dict(metadata.tool_config)
        input_schema, schema_metadata = cls._extract_executor_schema(executor)
        runtime_metadata = {
            **schema_metadata,
            "execution_namespace": metadata.execution_namespace,
            "background_capabilities": config.get("background_capabilities"),
            "disable_input": bool(config.get("disable_input", False)),
            "hidden_params": config.get("_hidden_params"),
        }
        runtime_metadata = {
            key: value for key, value in runtime_metadata.items() if value is not None
        }
        native_bindings = tuple(getattr(executor, "native_bindings", ()))
        return ToolDefinition(
            name=metadata.name,
            executor=executor,
            input_schema=input_schema,
            description=metadata.description,
            annotations=metadata.annotations,
            dispatch=cls._compile_dispatch(config),
            feedback=cls._compile_feedback(config),
            context=cls._compile_context(config),
            loading=LoadingSpec(deferred=bool(config.get("defer_loading", False))),
            retry=config.get("retry"),
            native_bindings=native_bindings,
            kind=config.get("tool_kind", "tool"),
            display_name=metadata.display_name,
            usage_guidance=metadata.usage_guidance,
            metadata=runtime_metadata,
        )

    @classmethod
    def _extract_executor_schema(
        cls,
        executor: ToolExecutor,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        mcp_tool_info = getattr(executor, "_mcp_tool_info", None)
        if mcp_tool_info is not None:
            parameters = getattr(mcp_tool_info, "inputSchema", None)
            return (
                deepcopy(dict(parameters)) if isinstance(parameters, Mapping) else {},
                {},
            )
        get_json_schema = getattr(executor, "get_json_schema", None)
        if not callable(get_json_schema):
            raise TypeError("Compiled tool executors must provide `get_json_schema()`")
        return cls._extract_input_schema(get_json_schema())

    @staticmethod
    def _compile_dispatch(config: Mapping[str, Any]) -> DispatchSpec:
        if config.get("background", False):
            return DispatchSpec(
                name="background",
                options={
                    "capabilities": config.get("background_capabilities", ()),
                },
            )
        if config.get("detached", False):
            return DispatchSpec(name="detached")
        if config.get("allow_background", False):
            return DispatchSpec(
                name="optional_background",
                options={
                    "argument": RUNTIME_BACKGROUND_PARAM,
                    "capabilities": config.get("background_capabilities", ()),
                },
            )
        return DispatchSpec(name="foreground")

    @staticmethod
    def _compile_feedback(config: Mapping[str, Any]) -> FeedbackSpec:
        if config.get("feedback") is not None:
            return FeedbackSpec.coerce(config["feedback"])
        if config.get("call_as_response", False):
            return FeedbackSpec(name="call_as_response")
        if config.get("handoff", False):
            return FeedbackSpec(name="handoff")
        if config.get("return_direct", False):
            return FeedbackSpec(name="direct")
        return FeedbackSpec(name="model")

    @staticmethod
    def _compile_context(config: Mapping[str, Any]) -> ContextSpec:
        bindings = []
        for source in ("message", "messages", "handle"):
            if config.get(f"inject_{source}", False):
                options = (
                    {"copy": True}
                    if source == "messages" and config.get("tool_kind") == "agent"
                    else {}
                )
                bindings.append(ContextBinding(source=source, options=options))
        inject_vars = config.get("inject_vars", False)
        if inject_vars is not False:
            is_selection = isinstance(inject_vars, Collection) and not isinstance(
                inject_vars,
                (str, bytes, Mapping),
            )
            if is_selection and config.get("tool_kind") != "agent":
                bindings.extend(
                    ContextBinding(
                        source="vars",
                        parameter=name,
                        options={"key": name},
                    )
                    for name in inject_vars
                )
            else:
                options = {"select": tuple(inject_vars)} if is_selection else {}
                bindings.append(ContextBinding(source="vars", options=options))
        return ContextSpec(bindings=tuple(bindings))

    @staticmethod
    def _extract_input_schema(
        schema: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(schema, Mapping):
            raise TypeError("Tool schemas must be mappings")
        function = schema.get("function")
        if schema.get("type") == "function" and isinstance(function, Mapping):
            parameters = function.get("parameters", {})
            strict = function.get("strict")
        elif schema.get("type") == "function":
            parameters = schema.get("parameters", {})
            strict = schema.get("strict")
        else:
            raise ValueError("Tool schemas must use a function-tool shape")
        if not isinstance(parameters, Mapping):
            raise TypeError("Tool schema parameters must be a mapping")
        metadata = {"strict": strict} if strict is not None else {}
        return deepcopy(dict(parameters)), metadata


class ToolCatalogEntry(msgspec.Struct, frozen=True, kw_only=True):
    """Execution-free projection of a tool in one thread catalog snapshot."""

    ref: ToolRef
    description: str | None
    input_schema: Mapping[str, Any]
    annotations: Mapping[str, Any] = msgspec.field(default_factory=dict)
    native_bindings: tuple[NativeToolBinding, ...] = ()
    kind: str = "tool"
    deferred: bool = False
    loaded: bool = False
    display_name: str | None = None
    usage_guidance: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ToolRef):
            raise TypeError("`ref` must be a ToolRef")
        if not isinstance(self.deferred, bool) or not isinstance(self.loaded, bool):
            raise TypeError("`deferred` and `loaded` must be bool values")
        if self.loaded and not self.deferred:
            raise ValueError("Only deferred tools can be marked as loaded")
        msgspec.structs.force_setattr(self, "kind", _require_name(self.kind, "kind"))
        msgspec.structs.force_setattr(
            self,
            "input_schema",
            _copy_mapping(self.input_schema, "input_schema"),
        )
        msgspec.structs.force_setattr(
            self,
            "annotations",
            _copy_mapping(self.annotations, "annotations"),
        )

    @property
    def name(self) -> str:
        return self.ref.tool_id

    @classmethod
    def from_definition(
        cls,
        definition: ToolDefinition,
        *,
        library_id: str,
        loaded: bool = False,
    ) -> ToolCatalogEntry:
        return cls(
            ref=ToolRef(library_id=library_id, tool_id=definition.name),
            description=definition.description,
            input_schema=definition.input_schema,
            annotations=definition.annotations,
            native_bindings=definition.native_bindings,
            kind=definition.kind,
            deferred=definition.loading.deferred,
            loaded=loaded,
            display_name=definition.display_name,
            usage_guidance=definition.usage_guidance,
        )


class ToolCatalogView(msgspec.Struct, frozen=True, kw_only=True):
    """Immutable tool catalog snapshot scoped to one conversation thread."""

    library_id: str
    thread_id: str
    entries: tuple[ToolCatalogEntry, ...]
    choice: ToolChoice | str = msgspec.field(default_factory=ToolChoice)

    def __post_init__(self) -> None:
        msgspec.structs.force_setattr(
            self,
            "library_id",
            _require_name(self.library_id, "library_id"),
        )
        msgspec.structs.force_setattr(
            self,
            "thread_id",
            _require_name(self.thread_id, "thread_id"),
        )
        entries = tuple(self.entries)
        if not all(isinstance(entry, ToolCatalogEntry) for entry in entries):
            raise TypeError("`entries` must contain ToolCatalogEntry values")
        names = [entry.name for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("Tool catalog entries must have unique names")
        foreign = [
            entry.name for entry in entries if entry.ref.library_id != self.library_id
        ]
        if foreign:
            formatted = ", ".join(f"`{name}`" for name in foreign)
            raise ValueError(
                f"Tool catalog entries belong to another library: {formatted}"
            )
        choice = ToolChoice.coerce(self.choice)
        if choice.mode == "tool" and choice.name not in set(names):
            raise ValueError(f"Selected tool `{choice.name}` is not in the catalog")
        msgspec.structs.force_setattr(self, "entries", entries)
        msgspec.structs.force_setattr(self, "choice", choice)

    @property
    def has_deferred(self) -> bool:
        return any(entry.deferred and not entry.loaded for entry in self.entries)

    def visible_entries(self) -> tuple[ToolCatalogEntry, ...]:
        selected = self.choice.name if self.choice.mode == "tool" else None
        return tuple(
            entry
            for entry in self.entries
            if not entry.deferred or entry.loaded or entry.name == selected
        )


class ToolRegistry(Module):
    """Own stable logical definitions and their executable Modules."""

    def __init__(
        self,
        library_id: str,
        definitions: Collection[ToolDefinition] = (),
    ) -> None:
        super().__init__()
        self.library_id = _require_name(library_id, "library_id")
        self.executors = ModuleDict()
        self._definitions: dict[str, ToolDefinition] = {}
        for definition in definitions:
            self.add(definition)

    def add(self, definition: ToolDefinition) -> ToolRef:
        if not isinstance(definition, ToolDefinition):
            raise TypeError("`definition` must be a ToolDefinition")
        if definition.name in self._definitions:
            raise ValueError(f"Tool `{definition.name}` is already registered")
        if not isinstance(definition.executor, Module):
            raise TypeError("Tool executors must inherit msgflux.nn.Module")
        self.executors[definition.name] = definition.executor
        self._definitions[definition.name] = definition
        return self.ref(definition.name)

    def remove(self, tool: ToolRef | str) -> ToolDefinition:
        name = self._resolve_name(tool)
        try:
            definition = self._definitions.pop(name)
        except KeyError as exc:
            raise ValueError(f"Tool `{name}` is not registered") from exc
        del self.executors[name]
        return definition

    def get(self, tool: ToolRef | str) -> ToolDefinition:
        name = self._resolve_name(tool)
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise ValueError(f"Tool `{name}` is not registered") from exc

    def has(self, tool: ToolRef | str) -> bool:
        try:
            name = self._resolve_name(tool)
        except ValueError:
            return False
        return name in self._definitions

    def ref(self, name: str) -> ToolRef:
        return ToolRef(library_id=self.library_id, tool_id=name)

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._definitions.values())

    def catalog_view(
        self,
        thread_id: str,
        *,
        loaded_tools: Collection[str] = (),
        choice: ToolChoice | str | None = None,
    ) -> ToolCatalogView:
        loaded = set(loaded_tools)
        unknown = loaded - self._definitions.keys()
        if unknown:
            formatted = ", ".join(f"`{name}`" for name in sorted(unknown))
            raise ValueError(f"Loaded tools are not registered: {formatted}")
        non_deferred = {
            name for name in loaded if not self._definitions[name].loading.deferred
        }
        if non_deferred:
            formatted = ", ".join(f"`{name}`" for name in sorted(non_deferred))
            raise ValueError(f"Only deferred tools can be loaded: {formatted}")
        entries = tuple(
            ToolCatalogEntry.from_definition(
                definition,
                library_id=self.library_id,
                loaded=definition.name in loaded,
            )
            for definition in self._definitions.values()
        )
        return ToolCatalogView(
            library_id=self.library_id,
            thread_id=thread_id,
            entries=entries,
            choice=choice if choice is not None else ToolChoice(),
        )

    def _resolve_name(self, tool: ToolRef | str) -> str:
        if isinstance(tool, ToolRef):
            if tool.library_id != self.library_id:
                raise ValueError(
                    f"Tool ref belongs to `{tool.library_id}`, not `{self.library_id}`"
                )
            return tool.tool_id
        return _require_name(tool, "tool")


class ToolExecutionPlan(msgspec.Struct, frozen=True, kw_only=True):
    """Resolved intent plus private runtime arguments and selected policies."""

    intent: ToolIntent
    definition: ToolDefinition
    visible_arguments: Mapping[str, Any] = msgspec.field(default_factory=dict)
    runtime_arguments: Mapping[str, Any] = msgspec.field(default_factory=dict)
    dispatch: DispatchSpec | str | None = None
    feedback: FeedbackSpec | str | None = None

    def __post_init__(self) -> None:
        if self.intent.name != self.definition.name:
            raise ValueError("Tool intent and definition names must match")
        msgspec.structs.force_setattr(
            self,
            "visible_arguments",
            _copy_mapping(self.visible_arguments, "visible_arguments"),
        )
        if not isinstance(self.runtime_arguments, Mapping):
            raise TypeError("`runtime_arguments` must be a mapping")
        msgspec.structs.force_setattr(
            self,
            "runtime_arguments",
            dict(self.runtime_arguments),
        )
        collisions = self.visible_arguments.keys() & self.runtime_arguments.keys()
        if collisions:
            formatted = ", ".join(f"`{name}`" for name in sorted(collisions))
            raise ValueError(
                "Tool arguments cannot be both visible and runtime-provided: "
                f"{formatted}"
            )
        msgspec.structs.force_setattr(
            self,
            "dispatch",
            DispatchSpec.coerce(self.dispatch)
            if self.dispatch is not None
            else self.definition.dispatch,
        )
        msgspec.structs.force_setattr(
            self,
            "feedback",
            FeedbackSpec.coerce(self.feedback)
            if self.feedback is not None
            else self.definition.feedback,
        )

    @property
    def call_arguments(self) -> dict[str, Any]:
        return {**self.visible_arguments, **self.runtime_arguments}


class ToolRuntimeContext(msgspec.Struct, frozen=True, kw_only=True):
    """Execution-local values available to opt-in context bindings."""

    values: Mapping[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.values, Mapping):
            raise TypeError("`values` must be a mapping")
        msgspec.structs.force_setattr(self, "values", dict(self.values))

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def require(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise RuntimeError(
                f"Runtime context value `{name}` is unavailable"
            ) from exc


ExecuteTool = Callable[[ToolExecutionPlan | None], Awaitable[ToolOutcome]]


class DispatchRequest(msgspec.Struct, frozen=True, kw_only=True):
    """Input delivered to a dispatch extension."""

    plan: ToolExecutionPlan
    context: ToolRuntimeContext
    execute: ExecuteTool


class BeforeToolPolicy(msgspec.Struct, frozen=True, kw_only=True):
    """Typed payload evaluated before runtime arguments are resolved."""

    intent: ToolIntent
    definition: ToolDefinition
    context: ToolRuntimeContext


class BeforeDispatchPolicy(msgspec.Struct, frozen=True, kw_only=True):
    """Typed payload evaluated after an execution plan is prepared."""

    plan: ToolExecutionPlan
    context: ToolRuntimeContext


class AfterToolPolicy(msgspec.Struct, frozen=True, kw_only=True):
    """Typed payload evaluated after a dispatcher produces an outcome."""

    plan: ToolExecutionPlan
    outcome: ToolOutcome
    context: ToolRuntimeContext


class ToolExtension(Module):
    """Owned runtime capability registered by the future ToolLibrary."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = _require_name(name, "extension.name")
        self._registry_ref = None

    @property
    def registry(self) -> ToolExtensionRegistry:
        registry = self._registry_ref() if self._registry_ref is not None else None
        if registry is None:
            raise RuntimeError("The extension is not registered")
        return registry

    def _bind_registry(self, registry: ToolExtensionRegistry) -> None:
        self._registry_ref = weakref.ref(registry)

    def _unbind_registry(self) -> None:
        self._registry_ref = None

    def on_register(self, registry: ToolExtensionRegistry) -> None:
        """Run synchronous setup after the extension becomes visible."""

    async def aon_register(self, registry: ToolExtensionRegistry) -> None:
        """Run async setup; override this for network-backed extensions."""
        self.on_register(registry)

    def on_remove(self, registry: ToolExtensionRegistry) -> None:
        """Clean up synchronously owned resources."""

    async def aon_remove(self, registry: ToolExtensionRegistry) -> None:
        self.on_remove(registry)

    def __getstate__(self):
        state = super().__getstate__()
        state["_registry_ref"] = None
        return state


class ToolDispatch(ToolExtension):
    """Extension that owns one open dispatch name."""

    def __init__(self, name: str, *, dispatch_name: str) -> None:
        super().__init__(name)
        self.dispatch_name = _require_name(dispatch_name, "dispatch_name")

    async def dispatch(self, request: DispatchRequest) -> ToolOutcome:
        raise NotImplementedError


class ToolPolicy(ToolExtension):
    """Sequential policy extension for tool planning and outcomes."""

    async def before_tool(
        self,
        payload: BeforeToolPolicy,
    ) -> BeforeToolPolicy | ToolOutcome:
        return payload

    async def before_dispatch(
        self,
        payload: BeforeDispatchPolicy,
    ) -> BeforeDispatchPolicy | ToolOutcome:
        return payload

    async def after_tool(self, payload: AfterToolPolicy) -> AfterToolPolicy:
        return payload


class ForegroundDispatch(ToolDispatch):
    """Execute and await the tool inside the current run."""

    def __init__(self) -> None:
        super().__init__("dispatch_foreground", dispatch_name="foreground")

    async def dispatch(self, request: DispatchRequest) -> ToolOutcome:
        return await request.execute(None)


class DetachedDispatch(ToolDispatch):
    """Start execution without retaining a task result."""

    def __init__(self) -> None:
        super().__init__("dispatch_detached", dispatch_name="detached")

    async def dispatch(self, request: DispatchRequest) -> ToolOutcome:
        await F.aspawn(request.execute, None)
        return ToolOutcome.dispatched(
            request.plan.intent,
            metadata={"dispatch": self.dispatch_name},
        )


class BackgroundDispatch(ToolDispatch):
    """Delegate durable scheduling to a runtime-provided background service."""

    def __init__(self) -> None:
        super().__init__("dispatch_background", dispatch_name="background")

    async def dispatch(self, request: DispatchRequest) -> ToolOutcome:
        scheduler = request.context.require("background_dispatcher")
        if hasattr(scheduler, "adispatch"):
            outcome = await scheduler.adispatch(request)
        elif hasattr(scheduler, "dispatch"):
            outcome = await asyncio.to_thread(scheduler.dispatch, request)
        else:
            raise TypeError(
                "`background_dispatcher` must define `dispatch` or `adispatch`"
            )
        if not isinstance(outcome, ToolOutcome):
            raise TypeError("Background dispatchers must return ToolOutcome")
        return outcome


class OptionalBackgroundDispatch(ToolDispatch):
    """Choose foreground or background from one reserved model argument."""

    def __init__(self) -> None:
        super().__init__(
            "dispatch_optional_background",
            dispatch_name="optional_background",
        )

    async def dispatch(self, request: DispatchRequest) -> ToolOutcome:
        argument = request.plan.dispatch.options.get(
            "argument",
            RUNTIME_BACKGROUND_PARAM,
        )
        if not isinstance(argument, str) or not argument:
            raise ValueError("Optional background dispatch requires an argument name")
        visible_arguments = dict(request.plan.visible_arguments)
        run_in_background = visible_arguments.pop(argument, False)
        if run_in_background not in {True, False, None}:
            raise TypeError(f"`{argument}` must be a bool or None")
        dispatch_name = "background" if run_in_background is True else "foreground"
        selected_plan = msgspec.structs.replace(
            request.plan,
            visible_arguments=visible_arguments,
            dispatch=DispatchSpec(name=dispatch_name),
        )
        dispatcher = self.registry.get_dispatch(dispatch_name)

        async def execute() -> ToolOutcome:
            return await request.execute(selected_plan)

        return await dispatcher.dispatch(
            DispatchRequest(
                plan=selected_plan,
                context=request.context,
                execute=lambda _plan=None: execute(),
            )
        )


class ContextRequest(msgspec.Struct, frozen=True, kw_only=True):
    """Input delivered to a context provider extension."""

    binding: ContextBinding
    definition: ToolDefinition
    intent: ToolIntent
    context: ToolRuntimeContext


class ToolContextProvider(ToolExtension):
    """Extension that resolves one or more named runtime context sources."""

    def __init__(self, name: str, *, sources: Collection[str]) -> None:
        super().__init__(name)
        if isinstance(sources, str) or not isinstance(sources, Collection):
            raise TypeError("`sources` must be a collection of strings")
        normalized = tuple(
            _require_name(source, "context source") for source in sources
        )
        if not normalized:
            raise ValueError("`sources` must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("`sources` values must be unique")
        self.sources = normalized

    async def resolve(self, request: ContextRequest) -> Any:
        raise NotImplementedError


class RuntimeContextProvider(ToolContextProvider):
    """Resolve explicitly bound values from ToolRuntimeContext."""

    DEFAULT_SOURCES = ("handle", "message", "messages", "vars")

    def __init__(self, sources: Collection[str] = DEFAULT_SOURCES) -> None:
        super().__init__("context_runtime", sources=sources)

    @staticmethod
    def _select_key(binding: ContextBinding, value: Any) -> Any:
        selected_key = binding.options.get("key")
        if selected_key is None:
            return value
        if not isinstance(selected_key, str) or not selected_key:
            raise TypeError("Context binding `key` must be a non-empty string")
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Context source `{binding.source}` does not support key lookup"
            )
        if selected_key not in value:
            raise KeyError(
                f"Context source `{binding.source}` is missing `{selected_key}`"
            )
        return value[selected_key]

    @staticmethod
    def _select_keys(binding: ContextBinding, value: Any) -> Any:
        selected = binding.options.get("select")
        if selected is None:
            return value
        if isinstance(selected, str) or not isinstance(selected, Collection):
            raise TypeError("Context binding `select` must be a collection")
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Context source `{binding.source}` does not support selection"
            )
        missing = [key for key in selected if key not in value]
        if missing:
            formatted = ", ".join(f"`{key}`" for key in missing)
            raise KeyError(f"Context source `{binding.source}` is missing {formatted}")
        return {key: value[key] for key in selected}

    async def resolve(self, request: ContextRequest) -> Any:
        binding = request.binding
        if binding.source not in request.context.values:
            if binding.required:
                raise RuntimeError(
                    f"Runtime context value `{binding.source}` is unavailable"
                )
            return _MISSING_CONTEXT

        value = request.context.values[binding.source]
        value = self._select_key(binding, value)
        value = self._select_keys(binding, value)
        if binding.options.get("copy", False):
            value = deepcopy(value)
        return value


class ToolExtensionHandle:
    """Ownership handle for one registered V2 extension."""

    def __init__(self, registry: ToolExtensionRegistry, name: str) -> None:
        self._registry_ref = weakref.ref(registry)
        self.name = name

    @property
    def active(self) -> bool:
        registry = self._registry_ref()
        return registry is not None and registry.has(self.name)

    def remove(self) -> None:
        registry = self._registry_ref()
        if registry is not None:
            registry.remove(self.name)

    async def aremove(self) -> None:
        registry = self._registry_ref()
        if registry is not None:
            await registry.aremove(self.name)


class ToolExtensionRegistry(Module):
    """Transactional ownership and capability indexes for Tool extensions."""

    def __init__(
        self,
        extensions: Collection[ToolExtension] = (),
        *,
        install_defaults: bool = False,
    ) -> None:
        super().__init__()
        self.extensions = ModuleDict()
        self._dispatches: dict[str, str] = {}
        self._context_sources: dict[str, str] = {}
        self._policies: list[str] = []
        initial = list(extensions)
        if install_defaults:
            initial = [
                ForegroundDispatch(),
                BackgroundDispatch(),
                DetachedDispatch(),
                OptionalBackgroundDispatch(),
                RuntimeContextProvider(),
                *initial,
            ]
        for extension in initial:
            self.register(extension)

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        for extension in self.extensions.values():
            if isinstance(extension, ToolExtension):
                extension._bind_registry(self)

    def _validate_registration(
        self,
        extension: ToolExtension,
    ) -> tuple[str | None, tuple[str, ...]]:
        if not isinstance(extension, ToolExtension):
            raise TypeError("`extension` must be a ToolExtension")
        current_registry = (
            extension._registry_ref() if extension._registry_ref is not None else None
        )
        if current_registry is not None:
            raise ValueError(
                f"Extension `{extension.name}` is already registered on a registry"
            )
        if extension.name in self.extensions:
            raise ValueError(f"Extension `{extension.name}` is already registered")

        dispatch_name = (
            extension.dispatch_name if isinstance(extension, ToolDispatch) else None
        )
        if dispatch_name is not None and dispatch_name in self._dispatches:
            raise ValueError(f"Dispatch `{dispatch_name}` is already registered")

        context_sources = (
            extension.sources if isinstance(extension, ToolContextProvider) else ()
        )
        conflicts = [
            source for source in context_sources if source in self._context_sources
        ]
        if conflicts:
            formatted = ", ".join(f"`{source}`" for source in conflicts)
            raise ValueError(f"Context sources already registered: {formatted}")
        return dispatch_name, context_sources

    def _install(
        self,
        extension: ToolExtension,
    ) -> tuple[str | None, tuple[str, ...]]:
        dispatch_name, context_sources = self._validate_registration(extension)
        extension._bind_registry(self)
        self.extensions[extension.name] = extension
        if dispatch_name is not None:
            self._dispatches[dispatch_name] = extension.name
        for source in context_sources:
            self._context_sources[source] = extension.name
        if isinstance(extension, ToolPolicy):
            self._policies.append(extension.name)
        return dispatch_name, context_sources

    def _rollback_registration(
        self,
        extension: ToolExtension,
        dispatch_name: str | None,
        context_sources: tuple[str, ...],
    ) -> None:
        if extension.name in self.extensions:
            del self.extensions[extension.name]
        if dispatch_name is not None:
            self._dispatches.pop(dispatch_name, None)
        for source in context_sources:
            self._context_sources.pop(source, None)
        if extension.name in self._policies:
            self._policies.remove(extension.name)
        extension._unbind_registry()

    def register(self, extension: ToolExtension) -> ToolExtensionHandle:
        dispatch_name, context_sources = self._install(extension)
        try:
            extension.on_register(self)
        except Exception:
            try:
                extension.on_remove(self)
            finally:
                self._rollback_registration(
                    extension,
                    dispatch_name,
                    context_sources,
                )
            raise
        return ToolExtensionHandle(self, extension.name)

    async def aregister(
        self,
        extension: ToolExtension,
    ) -> ToolExtensionHandle:
        dispatch_name, context_sources = self._install(extension)
        try:
            await extension.aon_register(self)
        except Exception:
            try:
                await extension.aon_remove(self)
            finally:
                self._rollback_registration(
                    extension,
                    dispatch_name,
                    context_sources,
                )
            raise
        return ToolExtensionHandle(self, extension.name)

    def has(self, name: str) -> bool:
        return name in self.extensions

    def get_dispatch(self, name: str) -> ToolDispatch:
        extension_name = self._dispatches.get(name)
        if extension_name is None:
            available = ", ".join(sorted(self._dispatches)) or "none"
            raise ValueError(
                f"Dispatch `{name}` is not registered. Available: {available}."
            )
        extension = self.extensions[extension_name]
        if not isinstance(extension, ToolDispatch):
            raise RuntimeError(f"Extension `{extension_name}` is not a ToolDispatch")
        return extension

    def get_context_provider(self, source: str) -> ToolContextProvider:
        extension_name = self._context_sources.get(source)
        if extension_name is None:
            available = ", ".join(sorted(self._context_sources)) or "none"
            raise ValueError(
                f"Context source `{source}` is not registered. Available: {available}."
            )
        extension = self.extensions[extension_name]
        if not isinstance(extension, ToolContextProvider):
            raise RuntimeError(
                f"Extension `{extension_name}` is not a ToolContextProvider"
            )
        return extension

    async def resolve_context(
        self,
        definition: ToolDefinition,
        intent: ToolIntent,
        context: ToolRuntimeContext,
    ) -> dict[str, Any]:
        resolved = {}
        for binding in definition.context.bindings:
            provider = self.get_context_provider(binding.source)
            value = await provider.resolve(
                ContextRequest(
                    binding=binding,
                    definition=definition,
                    intent=intent,
                    context=context,
                )
            )
            if value is not _MISSING_CONTEXT:
                resolved[binding.parameter] = value
        return resolved

    async def dispatch(self, request: DispatchRequest) -> ToolOutcome:
        dispatcher = self.get_dispatch(request.plan.dispatch.name)
        outcome = await dispatcher.dispatch(request)
        if not isinstance(outcome, ToolOutcome):
            raise TypeError("Tool dispatch extensions must return ToolOutcome")
        return outcome

    async def before_tool(
        self,
        payload: BeforeToolPolicy,
    ) -> BeforeToolPolicy | ToolOutcome:
        for policy in self._iter_policies():
            previous = payload
            result = await policy.before_tool(payload)
            if isinstance(result, ToolOutcome):
                self._validate_blocked_outcome(result, payload.intent)
                return result
            if not isinstance(result, BeforeToolPolicy):
                raise TypeError(
                    "before_tool policies must return BeforeToolPolicy or ToolOutcome"
                )
            if (
                result.intent.id != previous.intent.id
                or result.intent.name != previous.intent.name
                or result.definition is not previous.definition
                or result.context is not previous.context
            ):
                raise ValueError(
                    "before_tool policies may only replace intent arguments"
                )
            payload = result
        return payload

    async def before_dispatch(
        self,
        payload: BeforeDispatchPolicy,
    ) -> BeforeDispatchPolicy | ToolOutcome:
        for policy in self._iter_policies():
            previous = payload
            result = await policy.before_dispatch(payload)
            if isinstance(result, ToolOutcome):
                self._validate_blocked_outcome(result, payload.plan.intent)
                return result
            if not isinstance(result, BeforeDispatchPolicy):
                raise TypeError(
                    "before_dispatch policies must return "
                    "BeforeDispatchPolicy or ToolOutcome"
                )
            if (
                result.plan.intent != previous.plan.intent
                or result.plan.definition is not previous.plan.definition
                or result.plan.visible_arguments != previous.plan.visible_arguments
                or result.plan.runtime_arguments != previous.plan.runtime_arguments
                or result.plan.feedback != previous.plan.feedback
                or result.context is not previous.context
            ):
                raise ValueError(
                    "before_dispatch policies may only replace the dispatch spec"
                )
            payload = result
        return payload

    async def after_tool(self, payload: AfterToolPolicy) -> AfterToolPolicy:
        for policy in self._iter_policies():
            previous = payload
            result = await policy.after_tool(payload)
            if not isinstance(result, AfterToolPolicy):
                raise TypeError("after_tool policies must return AfterToolPolicy")
            if (
                result.plan is not previous.plan
                or result.context is not previous.context
                or result.outcome.intent_id != previous.outcome.intent_id
                or result.outcome.tool_name != previous.outcome.tool_name
            ):
                raise ValueError(
                    "after_tool policies may only replace outcome result fields"
                )
            payload = result
        return payload

    def _iter_policies(self) -> tuple[ToolPolicy, ...]:
        policies = tuple(self.extensions[name] for name in self._policies)
        if not all(isinstance(policy, ToolPolicy) for policy in policies):
            raise RuntimeError("The policy index contains a non-ToolPolicy extension")
        return policies

    @staticmethod
    def _validate_blocked_outcome(
        outcome: ToolOutcome,
        intent: ToolIntent,
    ) -> None:
        if outcome.status != "blocked":
            raise ValueError("A blocking policy must return a blocked ToolOutcome")
        if outcome.intent_id != intent.id or outcome.tool_name != intent.name:
            raise ValueError("A policy returned an outcome for another tool intent")

    def remove(self, name: str) -> None:
        if name not in self.extensions:
            return
        extension = self.extensions[name]
        self._remove_indexes(extension)
        try:
            extension.on_remove(self)
        finally:
            del self.extensions[name]
            extension._unbind_registry()

    async def aremove(self, name: str) -> None:
        if name not in self.extensions:
            return
        extension = self.extensions[name]
        self._remove_indexes(extension)
        try:
            await extension.aon_remove(self)
        finally:
            del self.extensions[name]
            extension._unbind_registry()

    def _remove_indexes(self, extension: ToolExtension) -> None:
        if isinstance(extension, ToolDispatch):
            self._dispatches.pop(extension.dispatch_name, None)
        if isinstance(extension, ToolContextProvider):
            for source in extension.sources:
                self._context_sources.pop(source, None)
        if extension.name in self._policies:
            self._policies.remove(extension.name)


class ToolLibraryV2(Module):
    """Incubating execution facade backed by canonical tool contracts."""

    def __init__(
        self,
        definitions: Collection[ToolDefinition] = (),
        *,
        name: str = "tool_library",
        extensions: Collection[ToolExtension] = (),
    ) -> None:
        super().__init__()
        self.set_name(_require_name(name, "name"))
        self.registry = ToolRegistry(name, definitions)
        self.extensions = ToolExtensionRegistry(
            extensions,
            install_defaults=True,
        )

    def add(self, definition: ToolDefinition) -> ToolRef:
        return self.registry.add(definition)

    def remove(self, tool: ToolRef | str) -> ToolDefinition:
        return self.registry.remove(tool)

    def get_catalog_view(
        self,
        thread_id: str,
        *,
        loaded_tools: Collection[str] = (),
        choice: ToolChoice | str | None = None,
    ) -> ToolCatalogView:
        return self.registry.catalog_view(
            thread_id,
            loaded_tools=loaded_tools,
            choice=choice,
        )

    def forward(
        self,
        intent: ToolIntent,
        context: ToolRuntimeContext | None = None,
    ) -> ToolOutcome:
        return F.wait_for(self.aforward, intent, context)

    async def aforward(
        self,
        intent: ToolIntent,
        context: ToolRuntimeContext | None = None,
    ) -> ToolOutcome:
        if not isinstance(intent, ToolIntent):
            raise TypeError("`intent` must be a ToolIntent")
        context = self._coerce_context(context)

        try:
            definition = self.registry.get(intent.name)
        except ValueError as exc:
            return ToolOutcome.failed(
                intent,
                status="not_found",
                code="tool_not_found",
                message=str(exc),
            )

        before_tool = await self._abefore_tool(intent, definition, context)
        if isinstance(before_tool, ToolOutcome):
            return self._with_feedback(before_tool, definition.feedback)
        intent = before_tool.intent

        try:
            plan = await self._aprepare(intent, definition, context)
        except (AbortRequestedError, TaskInterruptRequestedError) as exc:
            return self._interrupted(intent, definition.feedback, exc)
        except Exception as exc:
            return ToolOutcome.failed(
                intent,
                status="execution_failed",
                code="tool_preparation_failed",
                message=str(exc),
                feedback=definition.feedback,
            )

        before_dispatch = await self._abefore_dispatch(plan, context)
        if isinstance(before_dispatch, ToolOutcome):
            return self._with_feedback(before_dispatch, plan.feedback)
        plan = before_dispatch.plan
        return await self._adispatch(plan, context)

    @staticmethod
    def _coerce_context(
        context: ToolRuntimeContext | None,
    ) -> ToolRuntimeContext:
        if context is None:
            return ToolRuntimeContext()
        if not isinstance(context, ToolRuntimeContext):
            raise TypeError("`context` must be a ToolRuntimeContext or None")
        return context

    async def _aprepare(
        self,
        intent: ToolIntent,
        definition: ToolDefinition,
        context: ToolRuntimeContext,
    ) -> ToolExecutionPlan:
        runtime_arguments = await self.extensions.resolve_context(
            definition,
            intent,
            context,
        )
        return ToolExecutionPlan(
            intent=intent,
            definition=definition,
            visible_arguments=intent.arguments,
            runtime_arguments=runtime_arguments,
        )

    async def _abefore_tool(
        self,
        intent: ToolIntent,
        definition: ToolDefinition,
        context: ToolRuntimeContext,
    ) -> BeforeToolPolicy | ToolOutcome:
        try:
            return await self.extensions.before_tool(
                BeforeToolPolicy(
                    intent=intent,
                    definition=definition,
                    context=context,
                )
            )
        except (AbortRequestedError, TaskInterruptRequestedError) as exc:
            return self._interrupted(intent, definition.feedback, exc)
        except Exception as exc:
            return ToolOutcome.failed(
                intent,
                status="blocked",
                code="tool_policy_failed",
                message=f"before_tool policy failed closed: {exc}",
                feedback=definition.feedback,
            )

    async def _abefore_dispatch(
        self,
        plan: ToolExecutionPlan,
        context: ToolRuntimeContext,
    ) -> BeforeDispatchPolicy | ToolOutcome:
        try:
            return await self.extensions.before_dispatch(
                BeforeDispatchPolicy(plan=plan, context=context)
            )
        except (AbortRequestedError, TaskInterruptRequestedError) as exc:
            return self._interrupted(plan.intent, plan.feedback, exc)
        except Exception as exc:
            return ToolOutcome.failed(
                plan.intent,
                status="blocked",
                code="tool_policy_failed",
                message=f"before_dispatch policy failed closed: {exc}",
                feedback=plan.feedback,
            )

    async def _aexecute(
        self,
        plan: ToolExecutionPlan,
        context: ToolRuntimeContext,
    ) -> ToolOutcome:
        try:
            result = await await_with_abort(
                plan.definition.executor.acall(**plan.call_arguments),
                context.get("abort_signal"),
            )
        except (AbortRequestedError, TaskInterruptRequestedError) as exc:
            return self._interrupted(plan.intent, plan.feedback, exc)
        except Exception as exc:
            return ToolOutcome.failed(
                plan.intent,
                status="execution_failed",
                code="tool_execution_failed",
                message=str(exc),
                feedback=plan.feedback,
            )
        return ToolOutcome.completed(
            plan.intent,
            result,
            feedback=plan.feedback,
        )

    async def _adispatch(
        self,
        plan: ToolExecutionPlan,
        context: ToolRuntimeContext,
    ) -> ToolOutcome:
        async def execute(
            selected_plan: ToolExecutionPlan | None = None,
        ) -> ToolOutcome:
            return await self._aexecute(selected_plan or plan, context)

        try:
            outcome = await self.extensions.dispatch(
                DispatchRequest(
                    plan=plan,
                    context=context,
                    execute=execute,
                )
            )
        except (AbortRequestedError, TaskInterruptRequestedError) as exc:
            return self._interrupted(plan.intent, plan.feedback, exc)
        except Exception as exc:
            return ToolOutcome.failed(
                plan.intent,
                status="execution_failed",
                code="tool_dispatch_failed",
                message=str(exc),
                feedback=plan.feedback,
            )

        if outcome.intent_id != plan.intent.id or outcome.tool_name != plan.intent.name:
            raise ValueError("Dispatch returned an outcome for another tool intent")
        outcome = self._with_feedback(outcome, plan.feedback)
        try:
            after_tool = await self.extensions.after_tool(
                AfterToolPolicy(
                    plan=plan,
                    outcome=outcome,
                    context=context,
                )
            )
        except (AbortRequestedError, TaskInterruptRequestedError) as exc:
            return self._interrupted(plan.intent, plan.feedback, exc)
        except Exception:
            return outcome
        return self._with_feedback(after_tool.outcome, plan.feedback)

    @staticmethod
    def _with_feedback(
        outcome: ToolOutcome,
        feedback: FeedbackSpec,
    ) -> ToolOutcome:
        if outcome.feedback == feedback:
            return outcome
        return msgspec.structs.replace(outcome, feedback=feedback)

    @staticmethod
    def _interrupted(
        intent: ToolIntent,
        feedback: FeedbackSpec,
        error: BaseException,
    ) -> ToolOutcome:
        return ToolOutcome.failed(
            intent,
            status="interrupted",
            code="tool_interrupted",
            message=str(error) or type(error).__name__,
            feedback=feedback,
        )


_MISSING_CONTEXT = object()


__all__ = [
    "AfterToolPolicy",
    "BackgroundDispatch",
    "BeforeDispatchPolicy",
    "BeforeToolPolicy",
    "ContextBinding",
    "ContextRequest",
    "ContextSpec",
    "DetachedDispatch",
    "DispatchRequest",
    "DispatchSpec",
    "FeedbackSpec",
    "ForegroundDispatch",
    "LoadingSpec",
    "NativeToolBinding",
    "RuntimeContextProvider",
    "ToolCatalogEntry",
    "ToolCatalogView",
    "ToolChoice",
    "ToolContextProvider",
    "ToolDefinition",
    "ToolDispatch",
    "ToolError",
    "ToolExecutionPlan",
    "ToolExtension",
    "ToolExtensionHandle",
    "ToolExtensionRegistry",
    "ToolIntent",
    "ToolLibraryV2",
    "ToolOutcome",
    "ToolPolicy",
    "ToolRef",
    "ToolRegistry",
    "ToolRuntimeContext",
]
