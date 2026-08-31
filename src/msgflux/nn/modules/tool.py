import asyncio
import inspect
import weakref
from copy import deepcopy
from dataclasses import replace
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    get_type_hints,
)

import msgspec

import msgflux.nn.functional as F
from msgflux.auto import AutoParams
from msgflux.chat_messages import ChatMessages
from msgflux.core.dotdict import dotdict
from msgflux.exceptions import (
    AbortRequestedError,
    TaskError,
    TaskInterruptRequestedError,
)
from msgflux.nn.extensions.tool_library import (
    BackgroundTasksExtension,
    MCPServersExtension,
    ToolLibraryExtension,
    ToolLibraryExtensionHandle,
    ToolSearchExtension,
)
from msgflux.nn.hooks import Hook
from msgflux.nn.hooks.events import AfterTool, BeforeTool, BeforeToolDispatch
from msgflux.nn.modules.container import ModuleDict
from msgflux.nn.modules.module import Module
from msgflux.nn.modules.tool_v2 import (
    AfterToolPolicy,
    BeforeDispatchPolicy,
    BeforeToolPolicy,
    ContextSpec,
    DispatchRequest,
    ToolCatalogView,
    ToolChoice,
    ToolDefinitionCompiler,
    ToolExecutionPlan,
    ToolExtension,
    ToolExtensionHandle,
    ToolExtensionRegistry,
    ToolRef,
    ToolRegistry,
    ToolRuntimeContext,
)
from msgflux.nn.modules.tool_v2 import (
    ToolDefinition as RuntimeToolDefinition,
)
from msgflux.protocols.mcp import (
    convert_mcp_schema_to_tool_schema,
    extract_tool_result_text,
)
from msgflux.runtime.abort import await_with_abort
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    InMemoryAgentInboxStore,
)
from msgflux.runtime.background import BackgroundTaskDispatcher
from msgflux.runtime.context import get_execution_context
from msgflux.runtime.events import EventType, emit_event, event_source
from msgflux.tasks import InMemoryTaskStore
from msgflux.telemetry.span import (
    aset_tool_attributes,
    set_tool_attributes,
)
from msgflux.tools.dataclasses import ToolMetadata
from msgflux.tools.definitions import ToolCatalog
from msgflux.tools.handles import ToolLibraryHandle
from msgflux.tools.helpers import (
    RESERVED_TOOL_KINDS,
    RUNTIME_BACKGROUND_PARAM,
    coerce_tool_params,
    is_background_capable,
    is_reserved_tool_kind,
    normalize_background_capabilities,
)
from msgflux.tools.responses import ToolCall, ToolResponses
from msgflux.tools.runtime import ToolIntent, ToolOutcome
from msgflux.tools.types import (
    ToolBucket,
    ToolLibraryOperator,
    unwrap_hidden_annotation,
)
from msgflux.utils.chat import generate_tool_json_schema
from msgflux.utils.inspect import fn_has_parameters, get_fn_param_defaults
from msgflux.utils.msgspec import restore_transport_value
from msgflux.utils.tenacity import apply_retry, default_tool_retry


class Tool(Module):
    """Tool is Module type that provide a json schema to tools."""

    _event_source_type = "tool"

    def get_json_schema(self):
        return generate_tool_json_schema(self)


class MCPTool(Tool):
    """MCP Tool Proxy - wraps remote MCP tool as a Tool object.

    This allows MCP tools to be treated exactly like local tools,
    enabling polymorphism and unified telemetry.

    Args:
        name: Tool name (without namespace prefix)
        mcp_client: Connected MCP client
        mcp_tool_info: MCP tool metadata
        namespace: MCP server namespace
        config: Optional tool configuration

    Example:
        >>> mcp_tool = MCPTool(
        ...     name="read_file",
        ...     mcp_client=client,
        ...     mcp_tool_info=tool_info,
        ...     namespace="filesystem"
        ... )
        >>> result = mcp_tool(path="/file.txt")
    """

    def __init__(
        self,
        name: str,
        mcp_client: Any,  # MCPClient type
        mcp_tool_info: Any,  # MCPToolInfo type
        namespace: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # Set full tool name with namespace
        full_name = f"{namespace}__{name}"
        self.set_name(full_name)
        self.register_buffer(
            "display_name",
            (config.get("display_name") or full_name) if config else full_name,
        )
        self.register_buffer(
            "usage_guidance",
            config.get("usage_guidance") if config else None,
        )

        # Store MCP-specific data
        self._mcp_client = mcp_client
        self._mcp_tool_info = mcp_tool_info
        self._namespace = namespace
        self._mcp_tool_name = name

        # Set description from MCP tool info
        if hasattr(mcp_tool_info, "description"):
            self.set_description(mcp_tool_info.description)

        # Store config
        tc = {**(config or {})}
        tc.setdefault("tool_kind", "tool")
        self.register_buffer("tool_config", tc)

        # Apply retry
        retry_config = tc.get("retry")
        self.forward = apply_retry(
            self.forward, retry_config, default=default_tool_retry
        )
        self.aforward = apply_retry(
            self.aforward, retry_config, default=default_tool_retry
        )

    def get_json_schema(self) -> Dict[str, Any]:
        """Convert MCP tool schema to standard tool JSON schema."""
        return convert_mcp_schema_to_tool_schema(self._mcp_tool_info, self._namespace)

    @set_tool_attributes(execution_type="remote", protocol="mcp")
    def forward(self, **kwargs) -> Any:
        """Execute MCP tool call."""
        # Call MCP tool (wrap async in sync)
        result = F.wait_for(self._mcp_client.call_tool, self._mcp_tool_name, kwargs)

        # Handle errors
        if result.isError:
            error_text = extract_tool_result_text(result)
            raise RuntimeError(f"MCP tool error: {error_text}")

        # Extract and return result
        return extract_tool_result_text(result)

    @aset_tool_attributes(execution_type="remote", protocol="mcp")
    async def aforward(self, **kwargs) -> Any:
        """Execute MCP tool call asynchronously."""
        # Call MCP tool
        result = await self._mcp_client.call_tool(self._mcp_tool_name, kwargs)

        # Handle errors
        if result.isError:
            error_text = extract_tool_result_text(result)
            raise RuntimeError(f"MCP tool error: {error_text}")

        # Extract and return result
        return extract_tool_result_text(result)


class LocalTool(Tool):
    """Local tool implementation."""

    def __init__(
        self,
        name: str,
        description: str,
        annotations: Dict[str, Any],
        tool_config: Dict[str, Any],
        impl: Callable,
        display_name: Optional[str] = None,
        transport_params: Optional[Dict[str, Any]] = None,
        usage_guidance: Optional[str] = None,
        execution_namespace: Optional[str] = None,
    ):
        super().__init__()
        self.set_name(name)
        self.set_description(description)
        self.register_buffer("display_name", display_name or name)
        self.register_buffer("usage_guidance", usage_guidance)
        self.register_buffer("execution_namespace", execution_namespace)
        self.set_annotations(annotations)
        self.register_buffer("tool_config", tool_config)
        self.register_buffer("transport_params", transport_params or {})
        self.impl = impl  # Not a buffer for now
        self._param_defaults = get_fn_param_defaults(impl)

        # Apply retry
        retry_config = tool_config.get("retry")
        self.forward = apply_retry(
            self.forward, retry_config, default=default_tool_retry
        )
        self.aforward = apply_retry(
            self.aforward, retry_config, default=default_tool_retry
        )

    def _restore_transport_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Restore transport-lowered tool params using the original annotations."""
        annotations = {
            name: hint
            for name, hint in self.get_module_annotations().items()
            if name != "return"
        }
        if not annotations:
            return kwargs
        restored = dict(kwargs)
        for param_name, type_hint in annotations.items():
            if param_name not in restored:
                continue
            restored[param_name] = restore_transport_value(
                restored[param_name],
                type_hint,
                restore_structs=True,
            )
        return restored

    def _strip_none_default_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Treat `null` tool arguments as omission when Python defaults exist.

        This mirrors the tool schema contract used by LocalTool/function tools:
        strict providers require every field in `required`, so optional/defaulted
        params are represented as nullable in the schema and mapped back to
        Python defaults here when the model emits `null`.
        """
        if not self._param_defaults:
            return kwargs
        return {
            key: value
            for key, value in kwargs.items()
            if not (key in self._param_defaults and value is None)
        }

    def _prepare_call_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = self._restore_transport_params(kwargs)
        return self._strip_none_default_kwargs(kwargs)

    @set_tool_attributes(execution_type="local")
    def forward(self, **kwargs):
        kwargs = self._prepare_call_kwargs(kwargs)
        if inspect.iscoroutinefunction(self.impl):
            return F.wait_for(self.impl, **kwargs)
        return self.impl(**kwargs)

    @aset_tool_attributes(execution_type="local")
    async def aforward(self, *args, **kwargs):
        kwargs = self._prepare_call_kwargs(kwargs)
        if hasattr(self.impl, "acall"):
            return await self.impl.acall(*args, **kwargs)
        elif inspect.iscoroutinefunction(self.impl):
            return await self.impl(*args, **kwargs)
        # Fall back to sync call in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.impl(*args, **kwargs))


def _inspect_tool_metadata(impl: Callable) -> ToolMetadata:  # noqa: C901
    """Extract normalized metadata from a callable tool."""
    tool_config = dotdict(deepcopy(getattr(impl, "tool_config", dotdict())))
    if "spawn" in tool_config:
        raise ValueError("The `spawn` tool option was removed; use `detached`.")
    if ToolLibraryOperator.is_operator_tool(impl):
        inherited_config = getattr(type(impl), "tool_config", {})
        if inherited_config.get("runtime_inputs") is not None:
            tool_config["runtime_inputs"] = inherited_config["runtime_inputs"]
        for key in ("inject_handle", "inject_message", "inject_messages"):
            if inherited_config.get(key):
                tool_config[key] = True
    tool_config.setdefault("defer_loading", False)

    name_overridden = tool_config.pop("name_overridden", None)
    configured_description = tool_config.get("description")
    configured_display_name = tool_config.get("display_name")
    configured_usage_guidance = tool_config.get("usage_guidance")

    # Case 1: Uninitialized or initialized class
    if inspect.isclass(impl) or callable(impl):
        if not callable(impl):
            raise NotImplementedError(
                "To transform a class in `nn.Tool`"
                " is necessary implement a `def __call__`"
            )

        doc = (
            configured_description
            or getattr(impl, "description", None)
            or getattr(impl, "__doc__", None)
            or getattr(impl.__call__, "__doc__", None)
        )
        if doc is None:
            raise NotImplementedError(
                "To transform a class into a `nn.Tool` "
                "it is necessary to implement a docstring. "
                "Can be: a cls attr `self.docstring`, or"
                "a docstring in the class or in `def __call__`"
            )

        name = (
            name_overridden
            or getattr(impl, "name", None)
            or getattr(impl, "__name__", None)
        )
        display_name = configured_display_name or getattr(impl, "display_name", None)
        usage_guidance = configured_usage_guidance or getattr(
            impl, "usage_guidance", None
        )

        # Instantiate class first if needed, so we can get instance attributes
        class_annotation_source = impl if inspect.isclass(impl) else None
        if inspect.isclass(impl):
            impl = impl()  # Initialized
            display_name = display_name or getattr(impl, "display_name", None)
            usage_guidance = usage_guidance or getattr(impl, "usage_guidance", None)

        # Now extract annotations (after instantiation for classes)
        annotation_source = None
        has_bucket_annotations = isinstance(impl, ToolBucket) and (
            "annotations" in vars(type(impl)) or "annotations" in vars(impl)
        )
        annotations = getattr(impl, "annotations", None)
        if isinstance(impl, ToolBucket) and not has_bucket_annotations:
            annotations = getattr(impl.__call__, "__annotations__", None)
            annotation_source = impl.__call__ if annotations is not None else None
        if annotations is None:
            annotations = getattr(impl, "__annotations__", None)
            if annotations is not None:
                annotation_source = class_annotation_source or impl
        if annotations is None:
            annotations = getattr(impl.__call__, "__annotations__", None)
            if annotations is not None:
                annotation_source = impl.__call__
        if annotations is None:
            if fn_has_parameters(impl.__call__):
                raise NotImplementedError(
                    "To transform a class in `nn.Tool` is necessary "
                    "to implement annotations of types hint in "
                    "`self.annotations`, `self.__annotations__` or in `def __call__`"
                )
            annotations = {}
        annotations = _resolve_tool_annotations(annotation_source, annotations)

    # Case 2: Function
    elif inspect.isfunction(impl) or inspect.iscoroutinefunction(impl):
        if configured_description or (
            hasattr(impl, "__doc__") and impl.__doc__ is not None
        ):
            doc = configured_description or impl.__doc__
        else:
            raise NotImplementedError(
                "To transform a function into a `nn.Tool` "
                "is necessary to implement a docstring"
            )

        annotations = impl.__annotations__
        annotation_source = impl

        if annotations is None:
            if fn_has_parameters(impl):
                raise NotImplementedError(
                    "To transform a function into a `nn.Tool` "
                    "is necessary to implement parameters "
                    "annotations of types hint "
                )
            annotations = {}
        annotations = _resolve_tool_annotations(annotation_source, annotations)

        name = name_overridden or impl.__name__
        display_name = configured_display_name or getattr(impl, "display_name", None)
        usage_guidance = configured_usage_guidance or getattr(
            impl, "usage_guidance", None
        )

    else:
        raise ValueError(
            "The given object is not a callable function, class, or instance"
        )

    if tool_config.get("handoff", False):
        name = "transfer_to_" + name

    if isinstance(impl, ToolBucket):
        tool_kind = ToolBucket.tool_kind
    else:
        tool_kind = tool_config.get("tool_kind") or getattr(impl, "tool_kind", None)
        tool_kind = tool_kind or "tool"
    if not isinstance(tool_kind, str) or not tool_kind.strip():
        raise ValueError(f"The tool `{name}` must define a non-empty tool_kind.")
    tool_config["tool_kind"] = tool_kind

    declared_capabilities = tool_config.get("background_capabilities")
    if declared_capabilities is not None:
        if not is_background_capable(tool_config):
            raise ValueError(
                "`background_capabilities` requires `background=True` or "
                "`allow_background=True`."
            )
        tool_config["background_capabilities"] = normalize_background_capabilities(
            declared_capabilities
        )

    annotations, hidden_params = _split_hidden_annotations(annotations)
    if hidden_params:
        tool_config["_hidden_params"] = hidden_params

    if tool_config.get("handoff", False) or tool_config.get("disable_input", False):
        annotations = {}  # pass only the model state
    else:
        configured_context = tool_config.get("runtime_inputs")
        if configured_context is not None:
            for binding in ContextSpec.coerce(configured_context).bindings:
                annotations.pop(binding.parameter, None)
        if tool_config.get("inject_message", False):
            annotations.pop("message", None)
        if tool_config.get("inject_messages", False):
            annotations.pop("messages", None)
        if tool_config.get("inject_handle", False):
            annotations.pop("handle", None)
        if tool_config.get("inject_vars", False):
            annotations.pop("vars", None)
        if tool_config.get("allow_background", False) and not tool_config.get(
            "background", False
        ):
            annotations[RUNTIME_BACKGROUND_PARAM] = Optional[bool]

    if tool_config.get("detached"):
        doc = "This tool will not generate a return. \n" + doc
    if tool_config.get("background"):
        doc = "This tool runs in the background and returns a task id. \n" + doc
    elif tool_config.get("allow_background", False):
        doc = (
            "This tool can run in the background when "
            f"`{RUNTIME_BACKGROUND_PARAM}=true`; otherwise it runs normally. \n" + doc
        )

    return ToolMetadata(
        name=name,
        description=doc,
        annotations=annotations,
        tool_config=tool_config,
        impl=impl,
        display_name=display_name or name,
        usage_guidance=usage_guidance,
        execution_namespace=(
            impl.get_module_name()
            if tool_kind == "agent" and hasattr(impl, "get_module_name")
            else None
        ),
    )


def _convert_metadata_to_local_tool(metadata: ToolMetadata) -> LocalTool:
    return LocalTool(
        name=metadata.name,
        description=metadata.description,
        annotations=metadata.annotations,
        tool_config=metadata.tool_config,
        impl=metadata.impl,
        display_name=metadata.display_name,
        usage_guidance=metadata.usage_guidance,
        execution_namespace=metadata.execution_namespace,
    )


def _resolve_tool_annotations(
    annotation_source: Callable | None,
    annotations: Mapping[str, Any],
) -> Dict[str, Any]:
    resolved = dict(annotations)
    if annotation_source is None:
        return resolved
    try:
        type_hints = get_type_hints(annotation_source)
    except Exception:
        return resolved
    if not type_hints:
        return resolved
    resolved.update(type_hints)
    return resolved


def _split_hidden_annotations(
    annotations: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    public_annotations: Dict[str, Any] = {}
    hidden_params: Dict[str, Any] = {}
    for name, annotation in annotations.items():
        hidden_type = unwrap_hidden_annotation(annotation)
        if hidden_type is not None:
            if name == "return":
                raise ValueError("`Hidden[...]` cannot be used as a return type.")
            hidden_params[name] = hidden_type
            continue
        public_annotations[name] = annotation
    return public_annotations, hidden_params


def _convert_module_to_nn_tool(impl: Callable) -> Tool:
    """Convert a callable in nn.Tool."""
    return _convert_metadata_to_local_tool(_inspect_tool_metadata(impl))


def _metadata_from_tool(tool: Tool) -> ToolMetadata:
    return ToolMetadata(
        name=tool.name,
        description=tool.get_module_description() or "",
        annotations=tool.get_module_annotations(),
        tool_config=getattr(tool, "tool_config", {}),
        impl=getattr(tool, "impl", tool),
        display_name=getattr(tool, "display_name", None) or tool.name,
        usage_guidance=getattr(tool, "usage_guidance", None),
        source_tool=tool,
        execution_namespace=getattr(tool, "execution_namespace", None),
    )


class _ToolBackgroundScheduler:
    """Adapt the durable task dispatcher to the canonical dispatch contract."""

    def __init__(self, library: "ToolLibrary") -> None:
        self._library_ref = weakref.ref(library)

    def dispatch(self, request: DispatchRequest) -> ToolOutcome:
        library = self._library_ref()
        if library is None:
            raise RuntimeError("The ToolLibrary is no longer available")
        plan = request.plan
        call = library.get_background_dispatcher().dispatch(
            tool=plan.definition.executor,
            tool_id=plan.intent.id,
            tool_name=plan.intent.name,
            call_params=plan.call_arguments,
            visible_params=plan.visible_arguments,
            config=library._plan_config(plan),
        )
        if call.error is not None:
            return library._failed_intent(
                plan.intent,
                status="execution_failed",
                code="tool_dispatch_failed",
                message=call.error,
                feedback=plan.feedback,
                arguments=plan.visible_arguments,
            )
        return library._dispatched_intent(
            plan.intent,
            call.result,
            feedback=plan.feedback,
            arguments=plan.visible_arguments,
        )


class ToolLibrary(Module, metaclass=AutoParams):
    """ToolLibrary is a Module type that manage tool calls over the tool library."""

    _event_source_type = "tool_library"

    def __init__(
        self,
        name: str,
        tools: List[Callable],
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        task_store: Any | None = None,
        extensions: Optional[List[ToolLibraryExtension | ToolExtension]] = None,
    ):
        """Initialize the ToolLibrary.

        Args:
        name:
            Library name.
        tools:
            A list of callables.
        mcp_servers:
            List of MCP server configurations. Each config should contain:
            - name: Namespace for tools from this server
            - transport: "stdio" or "http"
            - For stdio: command, args, cwd, env
            - For http: base_url, headers
            - Optional: include_tools, exclude_tools, tool_config
        extensions:
            Optional tool-library capabilities that contribute tools, hooks,
            policies, dispatchers, setup, or cleanup under one removable owner.
        """
        super().__init__()
        self.set_name(f"{name}_tool_library")
        self.library = ModuleDict()
        self.registry = ToolRegistry(self.name)
        self.register_buffer("tool_configs", {})
        self.register_buffer("mcp_clients", {})
        self._task_store = task_store
        self._agent_inbox: Optional[AgentInbox] = None
        self._disabled_background_task_tool_names: set[str] = set()
        self._handle: Optional[ToolLibraryHandle] = None
        self._background_dispatcher: Optional[BackgroundTaskDispatcher] = None
        self._lifecycle_owner_ref: Optional[weakref.ReferenceType[Module]] = None
        self.extensions = ModuleDict()
        self.runtime_extensions = ToolExtensionRegistry(install_defaults=True)
        self._extension_hook_handles: dict[str, list[Any]] = {}
        self._extension_tool_names: dict[str, tuple[str, ...]] = {}
        self.register_extension("background_tasks", BackgroundTasksExtension())
        for extension in extensions or ():
            self.register_extension(extension.name, extension)
        for tool in tools:
            self.add(tool)
        if mcp_servers:
            self.register_extension("mcp_servers", MCPServersExtension(mcp_servers))

    def get_handle(self) -> ToolLibraryHandle:
        if self._handle is None:
            self._handle = ToolLibraryHandle(self)
        return self._handle

    def set_lifecycle_owner(self, owner: Module) -> None:
        """Bind the owning Agent lifecycle without transferring hook ownership."""
        self._lifecycle_owner_ref = weakref.ref(owner)

    @staticmethod
    def inspect_tool_metadata(tool: Callable) -> ToolMetadata:
        """Normalize one callable for extension-managed registration."""
        return _inspect_tool_metadata(tool)

    @staticmethod
    def create_mcp_tool(**kwargs: Any) -> MCPTool:
        """Build the library's canonical remote-tool proxy."""
        return MCPTool(**kwargs)

    def register_extension(  # noqa: C901
        self,
        name: str,
        extension: ToolLibraryExtension | ToolExtension,
    ) -> ToolLibraryExtensionHandle | ToolExtensionHandle:
        """Install a named library extension and return its ownership handle."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("`name` must be a non-empty string")
        if isinstance(extension, ToolExtension):
            if name != extension.name:
                raise ValueError(
                    "Runtime extension registration name must match "
                    f"`{extension.name}`."
                )
            if name in self.extensions or self.runtime_extensions.has(name):
                raise ValueError(f"The extension name `{name}` is already registered")
            return self.runtime_extensions.register(extension)
        if not isinstance(extension, ToolLibraryExtension):
            raise TypeError(
                "`extension` must be a ToolLibraryExtension or ToolExtension, "
                f"given `{type(extension)}`"
            )
        if name in self.extensions or self.runtime_extensions.has(name):
            raise ValueError(f"The extension name `{name}` is already registered")

        extension_tools = tuple(extension.tools())
        extension_hooks = tuple(extension.hooks())
        tool_names: list[str] = []
        hook_handles = []
        extension._bind_library(self)
        try:
            for tool in extension_tools:
                tool_names.append(self.add(tool))
            for hook in extension_hooks:
                if not isinstance(hook, Hook):
                    raise TypeError(
                        f"Extension `{name}` returned a non-Hook contribution: "
                        f"`{type(hook)}`"
                    )
                target = getattr(self, hook.target) if hook.target else self
                hook_handles.append(hook.register(target))
            self.extensions[name] = extension
            self._extension_tool_names[name] = tuple(tool_names)
            self._extension_hook_handles[name] = hook_handles
            extension.on_register(self)
        except Exception:
            for handle in reversed(hook_handles):
                handle.remove()
            for tool_name in reversed(tool_names):
                try:
                    self.remove(tool_name)
                except ValueError:
                    pass
            if name in self.extensions:
                del self.extensions[name]
            self._extension_tool_names.pop(name, None)
            self._extension_hook_handles.pop(name, None)
            try:
                extension.on_remove(self)
            finally:
                extension._unbind_library()
            raise
        return ToolLibraryExtensionHandle(self, name)

    def has_extension(self, name: str) -> bool:
        return name in self.extensions or self.runtime_extensions.has(name)

    def remove_extension(self, name: str) -> None:
        if self.runtime_extensions.has(name):
            self.runtime_extensions.remove(name)
            return
        if name not in self.extensions:
            return
        extension = self.extensions[name]
        for handle in reversed(self._extension_hook_handles.get(name, ())):
            handle.remove()
        for tool_name in reversed(self._extension_tool_names.get(name, ())):
            self.remove(tool_name)
        self._extension_hook_handles.pop(name, None)
        self._extension_tool_names.pop(name, None)
        try:
            extension.on_remove(self)
        finally:
            if name in self.extensions:
                del self.extensions[name]
            extension._unbind_library()

    async def aremove_extension(self, name: str) -> None:
        if self.runtime_extensions.has(name):
            await self.runtime_extensions.aremove(name)
            return
        if name not in self.extensions:
            return
        extension = self.extensions[name]
        for handle in reversed(self._extension_hook_handles.get(name, ())):
            handle.remove()
        for tool_name in reversed(self._extension_tool_names.get(name, ())):
            self.remove(tool_name)
        self._extension_hook_handles.pop(name, None)
        self._extension_tool_names.pop(name, None)
        try:
            await extension.aon_remove(self)
        finally:
            if name in self.extensions:
                del self.extensions[name]
            extension._unbind_library()

    def __getstate__(self):
        state = super().__getstate__()
        state["_lifecycle_owner_ref"] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._lifecycle_owner_ref = None
        for extension in self.extensions.values():
            if isinstance(extension, ToolLibraryExtension):
                extension._bind_library(self)

    def _get_lifecycle_owner(self) -> Optional[Module]:
        if self._lifecycle_owner_ref is None:
            return None
        return self._lifecycle_owner_ref()

    def is_bucket(self, tool_name: str) -> bool:
        """Return whether a registered public tool is a bucket."""
        if tool_name not in self.library:
            return False
        return isinstance(getattr(self.library[tool_name], "impl", None), ToolBucket)

    def get_bucket_tool_names(self, bucket_name: str) -> List[str]:
        bucket = getattr(self.library.get(bucket_name), "impl", None)
        if not isinstance(bucket, ToolBucket):
            return []
        return sorted(bucket.tools)

    def bucket_has_tool(self, bucket_name: str, tool_name: str) -> bool:
        return tool_name in self.get_bucket_tool_names(bucket_name)

    def get_bucket_execution_namespace(
        self,
        bucket_name: str,
        tool_name: str,
    ) -> str:
        bucket = getattr(self.library.get(bucket_name), "impl", None)
        if not isinstance(bucket, ToolBucket):
            raise ValueError(f"The tool `{bucket_name}` is not a tool bucket.")
        metadata = bucket.tools.get(tool_name)
        if metadata is None:
            raise ValueError(f"Tool `{tool_name}` is not captured by `{bucket_name}`.")
        return metadata.execution_namespace or metadata.name

    def get_background_dispatcher(self) -> BackgroundTaskDispatcher:
        if self._background_dispatcher is None:
            self._background_dispatcher = BackgroundTaskDispatcher(self.get_handle())
        return self._background_dispatcher

    def _get_default_task_store(self) -> Any:
        if self._task_store is None:
            self._task_store = InMemoryTaskStore()
        return self._task_store

    def get_agent_inbox(self) -> AgentInbox:
        if self._agent_inbox is None:
            self._agent_inbox = AgentInbox(
                owner=self.name,
                store=InMemoryAgentInboxStore(),
            )
        return self._agent_inbox

    def _validate_new_tool_name(self, tool_name: str) -> None:
        if tool_name in self.library:
            raise ValueError(f"The tool name `{tool_name}` is already in tool library")
        if self.registry.has(tool_name):
            raise ValueError(f"Duplicate tool name `{tool_name}`")

    def add(self, tool: Callable) -> str:
        """Add a local tool in library."""
        if isinstance(tool, ToolMetadata):
            metadata = tool
        elif isinstance(tool, Tool):
            metadata = _metadata_from_tool(tool)
        else:
            metadata = _inspect_tool_metadata(tool)

        metadata.tool_config = dotdict(metadata.tool_config)
        metadata.tool_config.setdefault("defer_loading", False)

        self._validate_new_tool_name(metadata.name)
        if is_background_capable(metadata.tool_config):
            background = self.extensions.get("background_tasks")
            if isinstance(background, BackgroundTasksExtension):
                background.validate_source(metadata.impl, metadata.tool_config)

        # Deferred tools are held by the search bucket. Loading is thread-local.
        if (
            metadata.tool_config.get("defer_loading", False)
            and "tool_search" not in self.library
        ):
            if not self.has_extension("tool_search"):
                self.register_extension("tool_search", ToolSearchExtension())
            else:
                search_extension = self.extensions["tool_search"]
                self.add(next(iter(search_extension.tools())))

        # A matching registered bucket owns the tool instead of direct registration.
        bucket_name = ToolBucket.find_bucket(
            metadata,
            self.library,
            self.tool_configs,
        )
        if bucket_name is not None:
            self._add_to_bucket(bucket_name, metadata)
            return metadata.name

        capturing_bucket = ToolBucket.find_capturing_bucket(
            metadata.name,
            self.library,
            self.tool_configs,
        )
        if capturing_bucket is not None:
            raise ValueError(
                f"The tool name `{metadata.name}` is already in tool library"
            )

        # Normal tools become directly callable and visible according to their config.
        self._register_tool(metadata)
        return metadata.name

    def remove(self, tool_name: str):
        if tool_name in self.library.keys():
            bucket = getattr(self.library[tool_name], "impl", None)
            if isinstance(bucket, ToolBucket) and bucket.tools:
                raise ValueError(
                    f"The bucket tool `{tool_name}` still captures tools and cannot "
                    "be removed."
                )
            config = self.tool_configs.get(tool_name, {})
            background = self.extensions.get("background_tasks")
            is_task_tool = isinstance(
                background, BackgroundTasksExtension
            ) and background.is_active_task_tool(
                library=self, tool_name=tool_name, config=config
            )
            was_background = not is_reserved_tool_kind(
                config
            ) and is_background_capable(config)

            self._remove_registered_tool(tool_name)

            if is_task_tool:
                self._disabled_background_task_tool_names.add(tool_name)
                return

            if was_background:
                self._sync_background_task_tools()
            return

        bucket_name = ToolBucket.find_capturing_bucket(
            tool_name,
            self.library,
            self.tool_configs,
        )
        if bucket_name is None:
            raise ValueError(f"The tool name `{tool_name}` is not in tool library")
        self._remove_from_bucket(bucket_name, tool_name)

    def _remove_registered_tool(self, tool_name: str) -> None:
        if tool_name in self.library:
            self.library.pop(tool_name)
        self.tool_configs.pop(tool_name, None)
        if self.registry.has(tool_name):
            self.registry.remove(tool_name)

    def clear(self):
        self.library.clear()
        self.tool_configs.clear()
        self.registry.clear()
        for mcp_data in self.mcp_clients.values():
            F.wait_for(mcp_data["client"].disconnect)
        self.mcp_clients.clear()
        self._disabled_background_task_tool_names.clear()
        if self._background_dispatcher is not None:
            self._background_dispatcher.clear()

    def _register_tool(self, metadata: ToolMetadata) -> Tool:
        # A bucket must be valid before it becomes visible in the library.
        captures = []
        if metadata.tool_config.get("tool_kind") == ToolBucket.tool_kind:
            ToolBucket.validate_registration(
                metadata,
                self.library,
                self.tool_configs,
            )
            captures = ToolBucket.find_capture_candidates(
                metadata.impl,
                self.library,
                self.tool_configs,
            )

            # Check every pending capture before changing the current library state.
            for _, captured_tool in captures:
                metadata.impl.validate_capture(_metadata_from_tool(captured_tool))

        # Convert callable metadata to the local executable representation when needed.
        tool = (
            metadata.source_tool
            if isinstance(metadata.source_tool, Tool)
            else _convert_metadata_to_local_tool(metadata)
        )

        # Register the public tool and its normalized configuration together.
        tool_config = dotdict(metadata.tool_config)
        if isinstance(metadata.source_tool, Tool):
            tool.register_buffer("tool_config", tool_config)
        definition = ToolDefinitionCompiler.compile(metadata, executor=tool)
        metadata.definition = definition
        metadata.ref = self.registry.add(definition)
        self.tool_configs[tool.name] = tool_config
        self.library.update({tool.name: tool})
        if isinstance(metadata.impl, ToolBucket):
            metadata.impl.refresh()
            self._sync_bucket_presentation(tool.name, metadata.impl)

        # An explicit re-add re-enables a builtin task control tool.
        config = tool_config
        if is_reserved_tool_kind(config):
            self._disabled_background_task_tool_names.discard(tool.name)

        # Background-capable sources determine the shared task control surface.
        self._sync_background_task_tools_for_source(config)

        # Move matching local tools into a newly registered bucket.
        for captured_name, captured_tool in captures:
            captured_metadata = _metadata_from_tool(captured_tool)
            self.remove(captured_name)
            self._add_to_bucket(tool.name, captured_metadata)
        return tool

    def _add_to_bucket(self, bucket_name: str, metadata: ToolMetadata) -> None:
        # Resolve the bucket implementation before changing its captured tools.
        bucket_tool = self.library[bucket_name]
        bucket = getattr(bucket_tool, "impl", None)
        if not isinstance(bucket, ToolBucket):
            raise ValueError(f"The bucket tool `{bucket_name}` cannot capture tools.")

        if not isinstance(metadata.source_tool, Tool):
            metadata.source_tool = _convert_metadata_to_local_tool(metadata)
        metadata.definition = ToolDefinitionCompiler.compile(
            metadata,
            executor=metadata.source_tool,
        )
        metadata.ref = self.registry.add(metadata.definition)

        # Let the bucket validate, retain, and refresh its captured state.
        try:
            bucket.add(metadata)
        except Exception:
            self.registry.remove(metadata.ref)
            raise
        self._sync_bucket_presentation(bucket_name, bucket)

    def _remove_from_bucket(self, bucket_name: str, tool_name: str) -> ToolMetadata:
        bucket_tool = self.library[bucket_name]
        bucket = getattr(bucket_tool, "impl", None)
        if not isinstance(bucket, ToolBucket):
            raise ValueError(f"The bucket tool `{bucket_name}` cannot release tools.")
        metadata = bucket.remove(tool_name)
        self.registry.remove(tool_name)
        if bucket.expose_captured_names and not bucket.tools:
            self._remove_registered_tool(bucket_name)
        else:
            self._sync_bucket_presentation(bucket_name, bucket)
        return metadata

    def _sync_bucket_presentation(self, bucket_name: str, bucket: ToolBucket) -> None:
        bucket_tool = self.library[bucket_name]
        if isinstance(getattr(bucket, "description", None), str):
            bucket_tool.set_description(bucket.description)
        annotations = bucket.patch_schema_annotations(
            bucket_tool.get_module_annotations()
        )
        if not isinstance(annotations, Mapping):
            raise TypeError("Bucket schema annotation patches must return a mapping.")
        bucket_tool.set_annotations(dict(annotations))
        if hasattr(bucket, "usage_guidance"):
            bucket_tool.register_buffer(
                "usage_guidance",
                bucket.usage_guidance,
            )
        bucket_metadata = _metadata_from_tool(bucket_tool)
        self.registry.replace(
            ToolDefinitionCompiler.compile(
                bucket_metadata,
                executor=bucket_tool,
            )
        )

    def get_tools(self) -> Iterator[Dict[str, Tool]]:
        return self.library.items()

    def get_tool_names(self) -> List[str]:
        """Get names of all tools."""
        names = list(self.library.keys())
        for tool in self.library.values():
            bucket = getattr(tool, "impl", None)
            if isinstance(bucket, ToolBucket) and bucket.expose_captured_names:
                names.extend(name for name in bucket.tools if name not in names)
        return names

    def get_tool_display_names(self) -> Dict[str, str]:
        """Return human-readable display names keyed by registered tool name."""
        display_names = {}
        for tool_name, tool in self.library.items():
            display_names[tool_name] = getattr(tool, "display_name", None) or tool_name

        return display_names

    def get_tool_usage_guidance(
        self, tool_names: Optional[set[str]] = None
    ) -> List[Dict[str, str]]:
        """Return usage guidance metadata for tools that define it."""
        guidance = []
        display_names = self.get_tool_display_names()

        for tool_name, tool in self.library.items():
            if tool_names is not None and tool_name not in tool_names:
                continue
            usage_guidance = getattr(tool, "usage_guidance", None)
            if usage_guidance:
                guidance.append(
                    {
                        "name": tool_name,
                        "display_name": display_names.get(tool_name, tool_name),
                        "guidance": usage_guidance,
                    }
                )

        return guidance

    def get_mcp_tool_names(self) -> List[str]:
        """Get names of all MCP tools (with namespace)."""
        tool_names = []
        for namespace, mcp_data in self.mcp_clients.items():
            for tool in mcp_data["tools"]:
                tool_names.append(f"{namespace}__{tool.name}")
        return tool_names

    def get_tool_json_schemas(self) -> List[Dict[str, Any]]:
        """Returns a list of JSON schemas from local and MCP tools."""
        return [tool.get_json_schema() for tool in self.library.values()]

    def get_tool_catalog(self, messages: ChatMessages | None = None) -> ToolCatalog:
        """Build the logical tool surface for one conversation thread."""
        return ToolCatalog.from_view(
            self._build_tool_catalog_view(messages, require_thread=False)
        )

    def _build_tool_catalog_view(
        self,
        messages: ChatMessages | None,
        *,
        choice: ToolChoice | str | Mapping[str, Any] | None = None,
        require_thread: bool,
    ) -> ToolCatalogView:
        if require_thread and not isinstance(messages, ChatMessages):
            raise TypeError("`messages` must be ChatMessages")
        thread_id = messages.thread_id if isinstance(messages, ChatMessages) else None
        if require_thread and (not isinstance(thread_id, str) or not thread_id):
            raise ValueError("Tool catalog views require a configured thread id")
        if not isinstance(thread_id, str) or not thread_id:
            thread_id = f"{self.name}:unscoped"
        catalog_names = set(self.get_tool_names())
        loaded = (
            messages.get_loaded_tools(self.name)
            if isinstance(messages, ChatMessages)
            else set()
        )
        return self.registry.catalog_view(
            thread_id,
            loaded_tools=loaded & catalog_names,
            choice=choice,
            include_tools=catalog_names,
        )

    def get_tool_catalog_view(
        self,
        messages: ChatMessages,
        *,
        choice: ToolChoice | str | Mapping[str, Any] | None = None,
    ) -> ToolCatalogView:
        """Return an immutable definition view for one configured thread."""
        return self._build_tool_catalog_view(
            messages,
            choice=choice,
            require_thread=True,
        )

    @staticmethod
    def _public_annotations(tool: Tool) -> Dict[str, Any]:
        return {
            name: hint
            for name, hint in tool.get_module_annotations().items()
            if name != "return"
        }

    @staticmethod
    def _tool_from_metadata(metadata: ToolMetadata) -> Tool:
        if isinstance(metadata.source_tool, Tool):
            return metadata.source_tool
        return _convert_metadata_to_local_tool(metadata)

    def _resolve_tool(self, tool_name: str) -> Tool | None:
        if tool_name in self.library:
            return self.library[tool_name]
        bucket_name = ToolBucket.find_capturing_bucket(
            tool_name,
            self.library,
            self.tool_configs,
        )
        if bucket_name is None:
            return None
        bucket = getattr(self.library[bucket_name], "impl", None)
        if not isinstance(bucket, ToolBucket):
            return None
        metadata = bucket.tools.get(tool_name)
        if metadata is None:
            return None
        return self._tool_from_metadata(metadata)

    def load_tools(
        self,
        messages: ChatMessages,
        tool_names: List[str],
    ) -> List[str]:
        if not isinstance(messages, ChatMessages):
            raise TypeError("Deferred tool loading requires `ChatMessages`.")
        deferred = {
            tool.name
            for tool in self.get_tool_catalog(messages).tools
            if tool.defer_loading
        }
        unknown = set(tool_names) - deferred
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Deferred tools are not available: {names}")
        return messages.load_tools(self.name, tool_names)

    def get_tool_annotations(self) -> Dict[str, Dict[str, Any]]:
        """Return local tool annotations keyed by tool name."""
        annotations = {}
        for tool_name, tool in self.library.items():
            annotations[tool_name] = {
                name: hint
                for name, hint in tool.get_module_annotations().items()
                if name != "return"
            }
        return annotations

    def get_tool_definition(self, tool_name: str) -> RuntimeToolDefinition:
        """Return the canonical definition for a public or bucket-captured tool."""
        try:
            return self.registry.get(tool_name)
        except ValueError as exc:
            raise ValueError(
                f"The tool name `{tool_name}` is not in tool library"
            ) from exc

    def get_tool_ref(self, tool_name: str) -> ToolRef:
        """Return a stable reference for a public or bucket-captured tool."""
        self.get_tool_definition(tool_name)
        return self.registry.ref(tool_name)

    def set_agent_inbox(self, agent_inbox: AgentInbox) -> None:
        self._agent_inbox = agent_inbox

    def set_task_store(self, task_store: Any) -> None:
        if task_store is not None:
            self._task_store = task_store

    def get_task_store(self, task_store: Any = None) -> Any:
        if task_store is not None:
            return task_store
        context_task_store = get_execution_context().get("task_store")
        if context_task_store is not None:
            return context_task_store
        return self._get_default_task_store()

    def _sync_background_task_tools_for_source(
        self,
        config: Mapping[str, Any],
    ) -> None:
        if is_reserved_tool_kind(config) or not is_background_capable(config):
            return
        self._sync_background_task_tools()

    def _sync_background_task_tools(self) -> None:
        background = self.extensions.get("background_tasks")
        if isinstance(background, BackgroundTasksExtension):
            background.sync(self)

    # --- Tool Call Preparation ---

    def _build_tool_argument_sets(
        self,
        *,
        definition: RuntimeToolDefinition,
        intent: ToolIntent,
        tool_params: Any,
        context: ToolRuntimeContext,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if definition.feedback.name == "handoff" or definition.metadata.get(
            "disable_input",
            False,
        ):
            visible_params: Dict[str, Any] = {}
        else:
            visible_params = coerce_tool_params(intent.name, tool_params)

        for param_name in definition.metadata.get("hidden_params") or {}:
            visible_params.pop(param_name, None)

        runtime_params: Dict[str, Any] = dict(runtime_arguments or {})
        if RUNTIME_BACKGROUND_PARAM in visible_params:
            runtime_params[RUNTIME_BACKGROUND_PARAM] = visible_params.pop(
                RUNTIME_BACKGROUND_PARAM
            )

        resolved = F.wait_for(
            self._aresolve_runtime_inputs,
            definition,
            intent,
            context,
        )
        if isinstance(resolved, TaskError):
            raise resolved.exception
        runtime_params.update(resolved)

        return visible_params, runtime_params

    async def _abuild_tool_argument_sets(
        self,
        *,
        definition: RuntimeToolDefinition,
        intent: ToolIntent,
        tool_params: Any,
        context: ToolRuntimeContext,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if definition.feedback.name == "handoff" or definition.metadata.get(
            "disable_input",
            False,
        ):
            visible_params: Dict[str, Any] = {}
        else:
            visible_params = coerce_tool_params(intent.name, tool_params)
        for param_name in definition.metadata.get("hidden_params") or {}:
            visible_params.pop(param_name, None)
        runtime_params: Dict[str, Any] = dict(runtime_arguments or {})
        if RUNTIME_BACKGROUND_PARAM in visible_params:
            runtime_params[RUNTIME_BACKGROUND_PARAM] = visible_params.pop(
                RUNTIME_BACKGROUND_PARAM
            )
        runtime_params.update(
            await self._aresolve_runtime_inputs(definition, intent, context)
        )
        return visible_params, runtime_params

    async def _aresolve_runtime_inputs(
        self,
        definition: RuntimeToolDefinition,
        intent: ToolIntent,
        context: ToolRuntimeContext,
    ) -> Dict[str, Any]:
        try:
            return await await_with_abort(
                self.runtime_extensions.resolve_context(
                    definition,
                    intent,
                    context,
                ),
                context.get("abort_signal"),
            )
        except KeyError as exc:
            key = "unknown"
            for binding in definition.context.bindings:
                value = context.get(binding.source)
                selected_key = binding.options.get("key")
                if selected_key is not None and (
                    not isinstance(value, Mapping) or selected_key not in value
                ):
                    key = selected_key
                    break
                selected = binding.options.get("select") or ()
                missing = [
                    selected_key
                    for selected_key in selected
                    if not isinstance(value, Mapping) or selected_key not in value
                ]
                if missing:
                    key = missing[0]
                    break
            subject = "agent" if definition.kind == "agent" else "tool"
            raise ValueError(
                f"The {subject} `{intent.name}` requires the injected parameter "
                f"`{key}`, but it was not found."
            ) from exc

    def _record_tool_activity(
        self,
        *,
        activity_recorder: Any,
        definition: RuntimeToolDefinition,
        parameters: Mapping[str, Any] | None,
    ) -> None:
        if (
            activity_recorder is None
            or definition.kind in RESERVED_TOOL_KINDS
            or ToolLibraryOperator.is_operator_tool(definition.executor)
        ):
            return
        activity_recorder.tool_call(definition.name, parameters)

    def _prepare_tool_kwargs(
        self,
        *,
        definition: RuntimeToolDefinition,
        intent: ToolIntent,
        tool_params: Any,
        context: ToolRuntimeContext,
        activity_recorder: Any,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        visible_params, runtime_params = self._build_tool_argument_sets(
            definition=definition,
            intent=intent,
            tool_params=tool_params,
            context=context,
            runtime_arguments=runtime_arguments,
        )
        self._record_tool_activity(
            activity_recorder=activity_recorder,
            definition=definition,
            parameters=visible_params,
        )
        return visible_params, runtime_params

    async def _aprepare_tool_kwargs(
        self,
        *,
        definition: RuntimeToolDefinition,
        intent: ToolIntent,
        tool_params: Any,
        context: ToolRuntimeContext,
        activity_recorder: Any,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        visible_params, runtime_params = await self._abuild_tool_argument_sets(
            definition=definition,
            intent=intent,
            tool_params=tool_params,
            context=context,
            runtime_arguments=runtime_arguments,
        )
        self._record_tool_activity(
            activity_recorder=activity_recorder,
            definition=definition,
            parameters=visible_params,
        )
        return visible_params, runtime_params

    def _build_execution_plan(
        self,
        *,
        definition: RuntimeToolDefinition,
        intent: ToolIntent,
        visible_arguments: Mapping[str, Any],
        runtime_arguments: Mapping[str, Any],
    ) -> ToolExecutionPlan:
        selected_dispatch = definition.dispatch.name
        selected_runtime_arguments = dict(runtime_arguments)
        if selected_dispatch == "optional_background":
            selected_dispatch = (
                "background"
                if selected_runtime_arguments.pop(RUNTIME_BACKGROUND_PARAM, False)
                is True
                else "foreground"
            )
        if (
            selected_dispatch == "foreground"
            and definition.feedback.name != "call_as_response"
        ):
            selected_runtime_arguments["tool_call_id"] = intent.id
        return ToolExecutionPlan(
            intent=intent,
            definition=definition,
            visible_arguments=dict(visible_arguments),
            runtime_arguments=selected_runtime_arguments,
            dispatch=selected_dispatch,
        )

    @staticmethod
    def _with_dispatch_mode(
        plan: ToolExecutionPlan,
        dispatch_mode: str,
    ) -> ToolExecutionPlan:
        selected_dispatch = dispatch_mode
        if selected_dispatch == plan.dispatch.name:
            return plan
        runtime_arguments = dict(plan.runtime_arguments)
        runtime_arguments.pop("tool_call_id", None)
        runtime_arguments.pop(RUNTIME_BACKGROUND_PARAM, None)
        if (
            selected_dispatch == "foreground"
            and plan.feedback.name != "call_as_response"
        ):
            runtime_arguments["tool_call_id"] = plan.intent.id
        return ToolExecutionPlan(
            intent=plan.intent,
            definition=plan.definition,
            visible_arguments=plan.visible_arguments,
            runtime_arguments=runtime_arguments,
            dispatch=selected_dispatch,
            feedback=plan.feedback,
        )

    @staticmethod
    def _definition_config(
        definition: RuntimeToolDefinition,
    ) -> Mapping[str, Any]:
        declaration = definition.metadata.get("declaration", {})
        return declaration if isinstance(declaration, Mapping) else {}

    @classmethod
    def _plan_config(cls, plan: ToolExecutionPlan) -> Mapping[str, Any]:
        return cls._definition_config(plan.definition)

    def _tool_runtime_context(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        message: Any,
        messages: Any,
        vars: Mapping[str, Any],
        sync_dispatch: bool,
    ) -> ToolRuntimeContext:
        execution = get_execution_context()
        handle = self.get_handle().for_tool(
            tool_name=tool_name,
            agent_inbox=execution.get("agent_inbox"),
            task_store=execution.get("task_store"),
            message=message,
            messages=messages,
            vars=vars,
            tool_call_id=tool_call_id,
            activity_recorder=execution.get("task_activity_recorder"),
        )
        return ToolRuntimeContext(
            values={
                "message": message,
                "messages": messages,
                "vars": vars,
                "handle": handle,
                "abort_signal": execution.get("abort_signal"),
                "task_store": execution.get("task_store"),
                "agent_inbox": execution.get("agent_inbox"),
                "activity_recorder": execution.get("task_activity_recorder"),
                "background_dispatcher": _ToolBackgroundScheduler(self),
                "sync_dispatch": sync_dispatch,
            }
        )

    @staticmethod
    def _emit_tool_blocked(event: BeforeTool | BeforeToolDispatch) -> None:
        with event_source(event.tool_name, "tool"):
            emit_event(
                EventType.TOOL_BLOCKED,
                {
                    "tool_call_id": event.tool_call_id,
                    "tool_name": event.tool_name,
                    "arguments": dict(event.arguments),
                    "reason": event.block,
                },
            )

    @staticmethod
    def _emit_policy_blocked(
        intent: ToolIntent,
        outcome: ToolOutcome,
        arguments: Mapping[str, Any],
    ) -> None:
        reason = outcome.error.message if outcome.error is not None else "Tool blocked"
        with event_source(intent.name, "tool"):
            emit_event(
                EventType.TOOL_BLOCKED,
                {
                    "tool_call_id": intent.id,
                    "tool_name": intent.name,
                    "arguments": dict(arguments),
                    "reason": reason,
                },
            )

    @classmethod
    def _normalize_policy_outcome(
        cls,
        outcome: ToolOutcome,
        *,
        intent: ToolIntent,
        feedback: Any,
        arguments: Mapping[str, Any],
    ) -> ToolOutcome:
        if outcome.intent_id != intent.id or outcome.tool_name != intent.name:
            raise ValueError("A policy returned an outcome for another tool intent")
        return msgspec.structs.replace(
            outcome,
            feedback=feedback,
            metadata={
                **dict(outcome.metadata),
                **cls._outcome_metadata(arguments),
            },
        )

    async def _abefore_tool_policy(
        self,
        intent: ToolIntent,
        definition: RuntimeToolDefinition,
        context: ToolRuntimeContext,
    ) -> BeforeToolPolicy | ToolOutcome:
        try:
            result = await await_with_abort(
                self.runtime_extensions.before_tool(
                    BeforeToolPolicy(
                        intent=intent,
                        definition=definition,
                        context=context,
                    )
                ),
                context.get("abort_signal"),
            )
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception as exc:
            return self._failed_intent(
                intent,
                status="blocked",
                code="tool_policy_failed",
                message=f"before_tool policy failed closed: {exc}",
                feedback=definition.feedback,
            )
        if isinstance(result, ToolOutcome):
            return self._normalize_policy_outcome(
                result,
                intent=intent,
                feedback=definition.feedback,
                arguments=intent.arguments,
            )
        return result

    async def _abefore_dispatch_policy(
        self,
        plan: ToolExecutionPlan,
        context: ToolRuntimeContext,
    ) -> BeforeDispatchPolicy | ToolOutcome:
        try:
            result = await await_with_abort(
                self.runtime_extensions.before_dispatch(
                    BeforeDispatchPolicy(plan=plan, context=context)
                ),
                context.get("abort_signal"),
            )
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception as exc:
            return self._failed_intent(
                plan.intent,
                status="blocked",
                code="tool_policy_failed",
                message=f"before_dispatch policy failed closed: {exc}",
                feedback=plan.feedback,
                arguments=plan.visible_arguments,
            )
        if isinstance(result, ToolOutcome):
            return self._normalize_policy_outcome(
                result,
                intent=plan.intent,
                feedback=plan.feedback,
                arguments=plan.visible_arguments,
            )
        return result

    def _validate_before_dispatch_event(
        self,
        event: Any,
        initial_event: BeforeToolDispatch,
    ) -> BeforeToolDispatch:
        if not isinstance(event, BeforeToolDispatch):
            raise TypeError(
                "before_dispatch handlers must return BeforeToolDispatch or None"
            )
        protected_fields = ("tool_call_id", "tool_name", "arguments", "config")
        changed_fields = [
            name
            for name in protected_fields
            if getattr(event, name) != getattr(initial_event, name)
        ]
        if changed_fields:
            formatted = ", ".join(f"`{name}`" for name in changed_fields)
            raise ValueError(
                "before_dispatch handlers may only replace `dispatch_mode` or "
                f"`block`; changed protected fields: {formatted}"
            )
        try:
            self.runtime_extensions.get_dispatch(event.dispatch_mode)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported tool dispatch mode: `{event.dispatch_mode}`"
            ) from exc
        if event.dispatch_mode != initial_event.dispatch_mode and not (
            initial_event.dispatch_mode in {"background", "detached"}
            and event.dispatch_mode == "foreground"
        ):
            raise ValueError(
                "before_dispatch may only keep the selected mode or reduce "
                "`background`/`detached` dispatch to `foreground`"
            )
        return event

    @staticmethod
    def _validate_blocking_hook_payload(
        event_name: str,
        payload: Any,
        expected_type: type,
    ) -> Any:
        if not isinstance(payload, expected_type):
            raise TypeError(
                f"{event_name} handlers must return {expected_type.__name__} or None"
            )
        return payload

    def _run_owned_blocking_hooks(
        self,
        event_name: str,
        payload: Any,
        validator: Callable[[Any], Any],
    ) -> Any:
        def stop_when(current: Any) -> bool:
            return getattr(current, "block", None) is not None

        if self.has_lifecycle_hooks(event_name):
            payload = self._run_lifecycle_hooks(
                event_name,
                payload,
                stop_when=stop_when,
            )
        payload = validator(payload)
        owner = self._get_lifecycle_owner()
        if (
            payload.block is None
            and owner is not None
            and owner.has_lifecycle_hooks(event_name)
        ):
            payload = owner._run_lifecycle_hooks(
                event_name,
                payload,
                stop_when=stop_when,
            )
        return validator(payload)

    async def _arun_owned_blocking_hooks(
        self,
        event_name: str,
        payload: Any,
        validator: Callable[[Any], Any],
    ) -> Any:
        def stop_when(current: Any) -> bool:
            return getattr(current, "block", None) is not None

        if self.has_lifecycle_hooks(event_name):
            payload = await self._arun_lifecycle_hooks(
                event_name,
                payload,
                stop_when=stop_when,
            )
        payload = validator(payload)
        owner = self._get_lifecycle_owner()
        if (
            payload.block is None
            and owner is not None
            and owner.has_lifecycle_hooks(event_name)
        ):
            payload = await owner._arun_lifecycle_hooks(
                event_name,
                payload,
                stop_when=stop_when,
            )
        return validator(payload)

    def _run_before_dispatch_hook(
        self,
        plan: ToolExecutionPlan,
    ) -> BeforeToolDispatch:
        event = BeforeToolDispatch(
            tool_call_id=plan.intent.id,
            tool_name=plan.intent.name,
            arguments=plan.visible_arguments,
            config=self._plan_config(plan),
            dispatch_mode=plan.dispatch.name,
        )
        initial_event = event
        try:
            event = self._run_owned_blocking_hooks(
                "before_dispatch",
                event,
                partial(
                    self._validate_before_dispatch_event,
                    initial_event=initial_event,
                ),
            )
            return event
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception as exc:
            return replace(
                initial_event,
                block=f"before_dispatch hook failed closed: {exc}",
            )

    async def _arun_before_dispatch_hook(
        self,
        plan: ToolExecutionPlan,
    ) -> BeforeToolDispatch:
        event = BeforeToolDispatch(
            tool_call_id=plan.intent.id,
            tool_name=plan.intent.name,
            arguments=plan.visible_arguments,
            config=self._plan_config(plan),
            dispatch_mode=plan.dispatch.name,
        )
        initial_event = event
        try:
            event = await self._arun_owned_blocking_hooks(
                "before_dispatch",
                event,
                partial(
                    self._validate_before_dispatch_event,
                    initial_event=initial_event,
                ),
            )
            return event
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception as exc:
            return replace(
                initial_event,
                block=f"before_dispatch hook failed closed: {exc}",
            )

    def _run_before_tool_hook(
        self,
        *,
        tool_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> BeforeTool:
        event = BeforeTool(
            tool_call_id=tool_id,
            tool_name=tool_name,
            arguments=dict(arguments),
        )
        try:
            return self._run_owned_blocking_hooks(
                "before_tool",
                event,
                partial(
                    self._validate_blocking_hook_payload,
                    "before_tool",
                    expected_type=BeforeTool,
                ),
            )
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception as exc:
            return BeforeTool(
                tool_call_id=tool_id,
                tool_name=tool_name,
                arguments=dict(arguments),
                block=f"before_tool hook failed closed: {exc}",
            )

    async def _arun_before_tool_hook(
        self,
        *,
        tool_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> BeforeTool:
        event = BeforeTool(
            tool_call_id=tool_id,
            tool_name=tool_name,
            arguments=dict(arguments),
        )
        try:
            return await self._arun_owned_blocking_hooks(
                "before_tool",
                event,
                partial(
                    self._validate_blocking_hook_payload,
                    "before_tool",
                    expected_type=BeforeTool,
                ),
            )
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception as exc:
            return BeforeTool(
                tool_call_id=tool_id,
                tool_name=tool_name,
                arguments=dict(arguments),
                block=f"before_tool hook failed closed: {exc}",
            )

    def _resolve_captured_tool(
        self,
        bucket_name: str,
        tool_name: str,
    ) -> Tool:
        if bucket_name not in self.library:
            raise ValueError(f"The bucket `{bucket_name}` is no longer available.")
        bucket = getattr(self.library[bucket_name], "impl", None)
        if not isinstance(bucket, ToolBucket):
            raise ValueError(f"The tool `{bucket_name}` is not a tool bucket.")
        metadata = bucket.tools.get(tool_name)
        if metadata is None:
            available = ", ".join(sorted(bucket.tools)) or "none"
            raise ValueError(
                f"Tool `{tool_name}` is not captured by `{bucket_name}`. "
                f"Available tools: {available}."
            )
        return self._tool_from_metadata(metadata)

    def _execute_prepared_tool(
        self,
        tool: Tool,
        call_params: Mapping[str, Any],
        visible_params: Mapping[str, Any],
    ) -> Any:
        event_data = self._tool_event_data(tool, call_params, visible_params)
        with event_source(event_data["tool_name"], "tool"):
            return self._execute_prepared_tool_impl(tool, call_params, event_data)

    def _execute_prepared_tool_impl(
        self,
        tool: Tool,
        call_params: Mapping[str, Any],
        event_data: Mapping[str, Any],
    ) -> Any:
        emit_event(EventType.TOOL_START, event_data)
        try:
            abort_signal = get_execution_context().get("abort_signal")
            if abort_signal is not None:
                abort_signal.raise_if_aborted()
            result = tool(**call_params)
            if abort_signal is not None:
                abort_signal.raise_if_aborted()
        except BaseException as exc:
            outcome = AfterTool(
                tool_call_id=event_data["tool_call_id"],
                tool_name=event_data["tool_name"],
                arguments=event_data["arguments"],
                error=exc,
            )
        else:
            outcome = AfterTool(
                tool_call_id=event_data["tool_call_id"],
                tool_name=event_data["tool_name"],
                arguments=event_data["arguments"],
                result=result,
            )
        outcome = self._run_after_tool_hook(outcome)
        emit_event(
            EventType.TOOL_END,
            {
                **event_data,
                "result": outcome.result,
                "error": str(outcome.error) if outcome.error is not None else None,
            },
        )
        if outcome.error is not None:
            if isinstance(outcome.error, BaseException):
                raise outcome.error
            raise RuntimeError(str(outcome.error))
        return outcome.result

    async def _aexecute_prepared_tool(
        self,
        tool: Tool,
        call_params: Mapping[str, Any],
        visible_params: Mapping[str, Any],
    ) -> Any:
        event_data = self._tool_event_data(tool, call_params, visible_params)
        with event_source(event_data["tool_name"], "tool"):
            return await self._aexecute_prepared_tool_impl(
                tool,
                call_params,
                event_data,
            )

    async def _aexecute_prepared_tool_impl(
        self,
        tool: Tool,
        call_params: Mapping[str, Any],
        event_data: Mapping[str, Any],
    ) -> Any:
        emit_event(EventType.TOOL_START, event_data)
        try:
            result = await await_with_abort(
                tool.acall(**call_params),
                get_execution_context().get("abort_signal"),
            )
        except BaseException as exc:
            outcome = AfterTool(
                tool_call_id=event_data["tool_call_id"],
                tool_name=event_data["tool_name"],
                arguments=event_data["arguments"],
                error=exc,
            )
        else:
            outcome = AfterTool(
                tool_call_id=event_data["tool_call_id"],
                tool_name=event_data["tool_name"],
                arguments=event_data["arguments"],
                result=result,
            )
        outcome = await self._arun_after_tool_hook(outcome)
        emit_event(
            EventType.TOOL_END,
            {
                **event_data,
                "result": outcome.result,
                "error": str(outcome.error) if outcome.error is not None else None,
            },
        )
        if outcome.error is not None:
            if isinstance(outcome.error, BaseException):
                raise outcome.error
            raise RuntimeError(str(outcome.error))
        return outcome.result

    def _run_after_tool_hook(self, outcome: AfterTool) -> AfterTool:
        try:
            if self.has_lifecycle_hooks("after_tool"):
                outcome = self._run_lifecycle_hooks("after_tool", outcome)
            if not isinstance(outcome, AfterTool):
                raise TypeError("after_tool handlers must return AfterTool or None")
            owner = self._get_lifecycle_owner()
            if owner is not None and owner.has_lifecycle_hooks("after_tool"):
                outcome = owner._run_lifecycle_hooks("after_tool", outcome)
            if not isinstance(outcome, AfterTool):
                raise TypeError("after_tool handlers must return AfterTool or None")
            return outcome
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception as exc:
            emit_event(
                EventType.HANDLER_ERROR,
                {"hook": "after_tool", "error": str(exc)},
            )
            return outcome

    async def _arun_after_tool_hook(self, outcome: AfterTool) -> AfterTool:
        try:
            if self.has_lifecycle_hooks("after_tool"):
                outcome = await self._arun_lifecycle_hooks("after_tool", outcome)
            if not isinstance(outcome, AfterTool):
                raise TypeError("after_tool handlers must return AfterTool or None")
            owner = self._get_lifecycle_owner()
            if owner is not None and owner.has_lifecycle_hooks("after_tool"):
                outcome = await owner._arun_lifecycle_hooks("after_tool", outcome)
            if not isinstance(outcome, AfterTool):
                raise TypeError("after_tool handlers must return AfterTool or None")
            return outcome
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception as exc:
            emit_event(
                EventType.HANDLER_ERROR,
                {"hook": "after_tool", "error": str(exc)},
            )
            return outcome

    @staticmethod
    def _tool_event_data(
        tool: Tool,
        call_params: Mapping[str, Any],
        visible_params: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "tool_call_id": call_params.get("tool_call_id"),
            "tool_name": tool.get_module_name(),
            "arguments": dict(visible_params),
        }

    def run(
        self,
        tool_ref: ToolRef | str,
        arguments: Mapping[str, Any],
        *,
        bucket_name: str | None = None,
        runtime_arguments: Mapping[str, Any] | None = None,
        message: Any = None,
        messages: Any = None,
        vars: Mapping[str, Any] | None = None,
        parent_tool_call_id: str | None = None,
        activity_recorder: Any = None,
    ) -> Any:
        """Execute one logical tool reference through the library pipeline."""
        tool_name = self._resolve_tool_ref_name(tool_ref)
        messages = messages if messages is not None else ChatMessages()
        vars = vars if vars is not None else {}
        owner = bucket_name or self.name
        tool_call_id = (
            f"{parent_tool_call_id}:{tool_name}"
            if parent_tool_call_id
            else f"{owner}:{tool_name}"
        )
        intent = ToolIntent(id=tool_call_id, name=tool_name, arguments=arguments)
        context = self._tool_runtime_context(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            message=message,
            messages=messages,
            vars=vars,
            sync_dispatch=True,
        )
        recorder = (
            activity_recorder
            if activity_recorder is not None
            else get_execution_context().get("task_activity_recorder")
        )
        prepared = self._prepare_intent(
            intent=intent,
            context=context,
            messages=messages,
            activity_recorder=recorder,
            bucket_name=bucket_name,
            runtime_arguments=runtime_arguments,
        )
        if isinstance(prepared, ToolOutcome):
            return self._unwrap_handle_outcome(prepared)
        dispatched = self._dispatch_intent_plan(intent, prepared, context)
        outcome = dispatched if isinstance(dispatched, ToolOutcome) else dispatched()
        if isinstance(outcome, TaskError):
            raise outcome.exception
        return self._unwrap_handle_outcome(outcome)

    async def arun(
        self,
        tool_ref: ToolRef | str,
        arguments: Mapping[str, Any],
        *,
        bucket_name: str | None = None,
        runtime_arguments: Mapping[str, Any] | None = None,
        message: Any = None,
        messages: Any = None,
        vars: Mapping[str, Any] | None = None,
        parent_tool_call_id: str | None = None,
        activity_recorder: Any = None,
    ) -> Any:
        """Async counterpart of :meth:`run`."""
        tool_name = self._resolve_tool_ref_name(tool_ref)
        messages = messages if messages is not None else ChatMessages()
        vars = vars if vars is not None else {}
        owner = bucket_name or self.name
        tool_call_id = (
            f"{parent_tool_call_id}:{tool_name}"
            if parent_tool_call_id
            else f"{owner}:{tool_name}"
        )
        intent = ToolIntent(id=tool_call_id, name=tool_name, arguments=arguments)
        context = self._tool_runtime_context(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            message=message,
            messages=messages,
            vars=vars,
            sync_dispatch=False,
        )
        recorder = (
            activity_recorder
            if activity_recorder is not None
            else get_execution_context().get("task_activity_recorder")
        )
        prepared = await self._aprepare_intent(
            intent=intent,
            context=context,
            messages=messages,
            activity_recorder=recorder,
            bucket_name=bucket_name,
            runtime_arguments=runtime_arguments,
        )
        if isinstance(prepared, ToolOutcome):
            return self._unwrap_handle_outcome(prepared)
        dispatched = await self._adispatch_intent_plan(intent, prepared, context)
        outcome = (
            dispatched if isinstance(dispatched, ToolOutcome) else await dispatched()
        )
        return self._unwrap_handle_outcome(outcome)

    @staticmethod
    def _unwrap_handle_outcome(outcome: ToolOutcome) -> Any:
        """Preserve the handle's value-or-exception interface over outcomes."""
        if outcome.ok:
            return outcome.result
        if outcome.error is None:
            raise RuntimeError(
                f"Tool `{outcome.tool_name}` finished with status `{outcome.status}`."
            )
        if outcome.status == "not_found":
            raise ValueError(outcome.error.message)
        raise RuntimeError(outcome.error.message)

    def _resolve_tool_ref_name(self, tool_ref: ToolRef | str) -> str:
        if isinstance(tool_ref, ToolRef):
            if tool_ref.library_id != self.name:
                raise ValueError(
                    f"Tool ref belongs to `{tool_ref.library_id}`, not `{self.name}`"
                )
            return tool_ref.tool_id
        if not isinstance(tool_ref, str) or not tool_ref:
            raise TypeError("`tool_ref` must be a ToolRef or non-empty string")
        return tool_ref

    def _call_captured_tool(
        self,
        bucket_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Compatibility wrapper for the former bucket execution path."""
        return self.run(
            self.get_tool_ref(tool_name),
            arguments,
            bucket_name=bucket_name,
            **kwargs,
        )

    async def _acall_captured_tool(
        self,
        bucket_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Compatibility wrapper for the former async bucket execution path."""
        return await self.arun(
            self.get_tool_ref(tool_name),
            arguments,
            bucket_name=bucket_name,
            **kwargs,
        )

    def execute_intents(
        self,
        intents: List[ToolIntent] | Tuple[ToolIntent, ...],
        *,
        message: Optional[Any] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[ToolOutcome, ...]:
        """Execute canonical intents without lowering them to legacy responses."""
        normalized = self._validate_intents(intents)
        return self._execute_intent_batch(
            normalized,
            message=message,
            messages=messages,
            vars=vars,
        )

    async def aexecute_intents(
        self,
        intents: List[ToolIntent] | Tuple[ToolIntent, ...],
        *,
        message: Optional[Any] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[ToolOutcome, ...]:
        """Async counterpart of execute_intents."""
        normalized = self._validate_intents(intents)
        return await self._aexecute_intent_batch(
            normalized,
            message=message,
            messages=messages,
            vars=vars,
        )

    @staticmethod
    def _validate_intents(
        intents: List[ToolIntent] | Tuple[ToolIntent, ...],
    ) -> Tuple[ToolIntent, ...]:
        normalized = tuple(intents)
        if not all(isinstance(intent, ToolIntent) for intent in normalized):
            raise TypeError("`intents` must contain ToolIntent values")
        return normalized

    @staticmethod
    def _outcome_metadata(arguments: Mapping[str, Any]) -> Dict[str, Any]:
        return {"arguments": dict(arguments)}

    @classmethod
    def _failed_intent(
        cls,
        intent: ToolIntent,
        *,
        status: str,
        code: str,
        message: str,
        feedback: Any = None,
        arguments: Mapping[str, Any] | None = None,
    ) -> ToolOutcome:
        return ToolOutcome.failed(
            intent,
            status=status,
            code=code,
            message=message,
            feedback=feedback,
            metadata=cls._outcome_metadata(
                intent.arguments if arguments is None else arguments
            ),
        )

    @classmethod
    def _completed_intent(
        cls,
        intent: ToolIntent,
        result: Any,
        *,
        feedback: Any,
        arguments: Mapping[str, Any],
    ) -> ToolOutcome:
        return ToolOutcome.completed(
            intent,
            result,
            feedback=feedback,
            metadata=cls._outcome_metadata(arguments),
        )

    @classmethod
    def _dispatched_intent(
        cls,
        intent: ToolIntent,
        result: Any,
        *,
        feedback: Any,
        arguments: Mapping[str, Any],
    ) -> ToolOutcome:
        return ToolOutcome.dispatched(
            intent,
            result,
            feedback=feedback,
            metadata=cls._outcome_metadata(arguments),
        )

    def _prepare_intent(
        self,
        intent: ToolIntent,
        *,
        context: ToolRuntimeContext,
        messages: ChatMessages | List[Dict[str, Any]],
        activity_recorder: Any,
        bucket_name: str | None = None,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> ToolExecutionPlan | ToolOutcome:
        resolved = (
            self._resolve_captured_tool(bucket_name, intent.name)
            if bucket_name is not None
            else self._resolve_tool(intent.name)
        )
        if resolved is None:
            return self._failed_intent(
                intent,
                status="not_found",
                code="tool_not_found",
                message=f"Error: Tool `{intent.name}` not found.",
            )

        definition = self.get_tool_definition(intent.name)
        before_policy = F.wait_for(
            self._abefore_tool_policy,
            intent,
            definition,
            context,
        )
        if isinstance(before_policy, ToolOutcome):
            self._emit_policy_blocked(intent, before_policy, intent.arguments)
            return before_policy
        intent = before_policy.intent
        if definition.loading.deferred and isinstance(messages, ChatMessages):
            messages.load_tools(self.name, [intent.name])
        visible_arguments, runtime_arguments = self._prepare_tool_kwargs(
            definition=definition,
            intent=intent,
            tool_params=intent.arguments,
            context=context,
            activity_recorder=activity_recorder,
            runtime_arguments=runtime_arguments,
        )
        feedback = definition.feedback
        before_tool = self._run_before_tool_hook(
            tool_id=intent.id,
            tool_name=intent.name,
            arguments=visible_arguments,
        )
        if before_tool.block is not None:
            self._emit_tool_blocked(before_tool)
            return self._failed_intent(
                intent,
                status="blocked",
                code="tool_blocked",
                message=before_tool.block,
                feedback=feedback,
                arguments=before_tool.arguments,
            )
        response_arguments = dict(before_tool.arguments)
        plan = self._build_execution_plan(
            definition=definition,
            intent=intent,
            visible_arguments=response_arguments,
            runtime_arguments=runtime_arguments,
        )
        before_dispatch = self._run_before_dispatch_hook(plan)
        if before_dispatch.block is not None:
            self._emit_tool_blocked(before_dispatch)
            return self._failed_intent(
                intent,
                status="blocked",
                code="tool_dispatch_blocked",
                message=before_dispatch.block,
                feedback=feedback,
                arguments=response_arguments,
            )
        plan = self._with_dispatch_mode(plan, before_dispatch.dispatch_mode)
        before_policy_dispatch = F.wait_for(
            self._abefore_dispatch_policy,
            plan,
            context,
        )
        if isinstance(before_policy_dispatch, ToolOutcome):
            self._emit_policy_blocked(
                intent,
                before_policy_dispatch,
                response_arguments,
            )
            return before_policy_dispatch
        return before_policy_dispatch.plan

    async def _aprepare_intent(
        self,
        intent: ToolIntent,
        *,
        context: ToolRuntimeContext,
        messages: ChatMessages | List[Dict[str, Any]],
        activity_recorder: Any,
        bucket_name: str | None = None,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> ToolExecutionPlan | ToolOutcome:
        resolved = (
            self._resolve_captured_tool(bucket_name, intent.name)
            if bucket_name is not None
            else self._resolve_tool(intent.name)
        )
        if resolved is None:
            return self._failed_intent(
                intent,
                status="not_found",
                code="tool_not_found",
                message=f"Error: Tool `{intent.name}` not found.",
            )

        definition = self.get_tool_definition(intent.name)
        before_policy = await self._abefore_tool_policy(
            intent,
            definition,
            context,
        )
        if isinstance(before_policy, ToolOutcome):
            self._emit_policy_blocked(intent, before_policy, intent.arguments)
            return before_policy
        intent = before_policy.intent
        if definition.loading.deferred and isinstance(messages, ChatMessages):
            messages.load_tools(self.name, [intent.name])
        visible_arguments, runtime_arguments = await self._aprepare_tool_kwargs(
            definition=definition,
            intent=intent,
            tool_params=intent.arguments,
            context=context,
            activity_recorder=activity_recorder,
            runtime_arguments=runtime_arguments,
        )
        feedback = definition.feedback
        before_tool = await self._arun_before_tool_hook(
            tool_id=intent.id,
            tool_name=intent.name,
            arguments=visible_arguments,
        )
        if before_tool.block is not None:
            self._emit_tool_blocked(before_tool)
            return self._failed_intent(
                intent,
                status="blocked",
                code="tool_blocked",
                message=before_tool.block,
                feedback=feedback,
                arguments=before_tool.arguments,
            )
        response_arguments = dict(before_tool.arguments)
        plan = self._build_execution_plan(
            definition=definition,
            intent=intent,
            visible_arguments=response_arguments,
            runtime_arguments=runtime_arguments,
        )
        before_dispatch = await self._arun_before_dispatch_hook(plan)
        if before_dispatch.block is not None:
            self._emit_tool_blocked(before_dispatch)
            return self._failed_intent(
                intent,
                status="blocked",
                code="tool_dispatch_blocked",
                message=before_dispatch.block,
                feedback=feedback,
                arguments=response_arguments,
            )
        plan = self._with_dispatch_mode(plan, before_dispatch.dispatch_mode)
        before_policy_dispatch = await self._abefore_dispatch_policy(plan, context)
        if isinstance(before_policy_dispatch, ToolOutcome):
            self._emit_policy_blocked(
                intent,
                before_policy_dispatch,
                response_arguments,
            )
            return before_policy_dispatch
        return before_policy_dispatch.plan

    def _dispatch_intent_plan(
        self,
        intent: ToolIntent,
        plan: ToolExecutionPlan,
        context: ToolRuntimeContext,
    ) -> ToolOutcome | Callable[[], Any]:
        feedback = plan.feedback
        arguments = plan.visible_arguments
        if feedback.name == "call_as_response":
            return self._completed_intent(
                intent,
                None,
                feedback=feedback,
                arguments=arguments,
            )
        return partial(
            F.wait_for,
            self._adispatch_runtime_plan,
            plan,
            context,
        )

    async def _adispatch_intent_plan(
        self,
        intent: ToolIntent,
        plan: ToolExecutionPlan,
        context: ToolRuntimeContext,
    ) -> ToolOutcome | Callable[[], Any]:
        feedback = plan.feedback
        arguments = plan.visible_arguments
        if feedback.name == "call_as_response":
            return self._completed_intent(
                intent,
                None,
                feedback=feedback,
                arguments=arguments,
            )
        return partial(self._adispatch_runtime_plan, plan, context)

    async def _adispatch_runtime_plan(
        self,
        plan: ToolExecutionPlan,
        context: ToolRuntimeContext,
    ) -> ToolOutcome:
        async def execute(
            selected_plan: ToolExecutionPlan | None = None,
        ) -> ToolOutcome:
            current = selected_plan or plan
            result = await self._aexecute_prepared_tool(
                current.definition.executor,
                current.call_arguments,
                current.visible_arguments,
            )
            return self._completed_intent(
                current.intent,
                result,
                feedback=current.feedback,
                arguments=current.visible_arguments,
            )

        outcome = await await_with_abort(
            self.runtime_extensions.dispatch(
                DispatchRequest(plan=plan, context=context, execute=execute)
            ),
            context.get("abort_signal"),
        )
        result = outcome.result
        if plan.dispatch.name == "detached" and result is None:
            result = (
                f"The `{plan.intent.name}` tool was dispatched. "
                "This tool will not generate a return."
            )
        outcome = ToolOutcome(
            intent_id=outcome.intent_id,
            tool_name=outcome.tool_name,
            status=outcome.status,
            result=result,
            error=outcome.error,
            feedback=plan.feedback,
            metadata={
                **dict(outcome.metadata),
                **self._outcome_metadata(plan.visible_arguments),
            },
        )
        if outcome.intent_id != plan.intent.id or outcome.tool_name != plan.intent.name:
            raise ValueError("Dispatch returned an outcome for another tool intent")
        try:
            after_policy = await await_with_abort(
                self.runtime_extensions.after_tool(
                    AfterToolPolicy(
                        plan=plan,
                        outcome=outcome,
                        context=context,
                    )
                ),
                context.get("abort_signal"),
            )
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception:
            return outcome
        return self._normalize_policy_outcome(
            after_policy.outcome,
            intent=plan.intent,
            feedback=plan.feedback,
            arguments=plan.visible_arguments,
        )

    def _execute_intent_batch(
        self,
        intents: Tuple[ToolIntent, ...],
        *,
        message: Any,
        messages: ChatMessages | List[Dict[str, Any]] | None,
        vars: Mapping[str, Any] | None,
    ) -> Tuple[ToolOutcome, ...]:
        messages = messages if messages is not None else ChatMessages()
        vars = vars if vars is not None else {}
        activity_recorder = get_execution_context().get("task_activity_recorder")
        outcomes: List[ToolOutcome | None] = [None] * len(intents)
        prepared_calls = []
        prepared_metadata = []

        for index, intent in enumerate(intents):
            runtime_context = self._tool_runtime_context(
                tool_name=intent.name,
                tool_call_id=intent.id,
                message=message,
                messages=messages,
                vars=vars,
                sync_dispatch=True,
            )
            prepared = self._prepare_intent(
                intent,
                context=runtime_context,
                messages=messages,
                activity_recorder=activity_recorder,
            )
            if isinstance(prepared, ToolOutcome):
                outcomes[index] = prepared
                continue
            dispatched = self._dispatch_intent_plan(
                intent,
                prepared,
                runtime_context,
            )
            if isinstance(dispatched, ToolOutcome):
                outcomes[index] = dispatched
                continue
            prepared_calls.append(dispatched)
            prepared_metadata.append((index, intent, prepared))

        if prepared_calls:
            results = F.scatter_gather(prepared_calls)
            for (index, intent, plan), result in zip(prepared_metadata, results):
                if isinstance(result, ToolOutcome):
                    outcomes[index] = result
                elif isinstance(result, TaskError):
                    if isinstance(
                        result.exception,
                        (AbortRequestedError, TaskInterruptRequestedError),
                    ):
                        raise result.exception
                    outcomes[index] = self._failed_intent(
                        intent,
                        status="execution_failed",
                        code="tool_execution_failed",
                        message=str(result),
                        feedback=self.get_tool_definition(intent.name).feedback,
                        arguments=plan.visible_arguments,
                    )
                else:
                    outcomes[index] = self._completed_intent(
                        intent,
                        result,
                        feedback=self.get_tool_definition(intent.name).feedback,
                        arguments=plan.visible_arguments,
                    )
        return self._finalize_outcomes(outcomes)

    async def _aexecute_intent_batch(
        self,
        intents: Tuple[ToolIntent, ...],
        *,
        message: Any,
        messages: ChatMessages | List[Dict[str, Any]] | None,
        vars: Mapping[str, Any] | None,
    ) -> Tuple[ToolOutcome, ...]:
        messages = messages if messages is not None else ChatMessages()
        vars = vars if vars is not None else {}
        activity_recorder = get_execution_context().get("task_activity_recorder")
        outcomes: List[ToolOutcome | None] = [None] * len(intents)
        prepared_calls = []
        prepared_metadata = []

        for index, intent in enumerate(intents):
            runtime_context = self._tool_runtime_context(
                tool_name=intent.name,
                tool_call_id=intent.id,
                message=message,
                messages=messages,
                vars=vars,
                sync_dispatch=False,
            )
            prepared = await self._aprepare_intent(
                intent,
                context=runtime_context,
                messages=messages,
                activity_recorder=activity_recorder,
            )
            if isinstance(prepared, ToolOutcome):
                outcomes[index] = prepared
                continue
            dispatched = await self._adispatch_intent_plan(
                intent,
                prepared,
                runtime_context,
            )
            if isinstance(dispatched, ToolOutcome):
                outcomes[index] = dispatched
                continue
            prepared_calls.append(dispatched)
            prepared_metadata.append((index, intent, prepared))

        if prepared_calls:
            results = await F.ascatter_gather(prepared_calls)
            for (index, intent, plan), result in zip(prepared_metadata, results):
                if isinstance(result, ToolOutcome):
                    outcomes[index] = result
                elif isinstance(result, TaskError):
                    if isinstance(
                        result.exception,
                        (AbortRequestedError, TaskInterruptRequestedError),
                    ):
                        raise result.exception
                    outcomes[index] = self._failed_intent(
                        intent,
                        status="execution_failed",
                        code="tool_execution_failed",
                        message=str(result),
                        feedback=self.get_tool_definition(intent.name).feedback,
                        arguments=plan.visible_arguments,
                    )
                else:
                    outcomes[index] = self._completed_intent(
                        intent,
                        result,
                        feedback=self.get_tool_definition(intent.name).feedback,
                        arguments=plan.visible_arguments,
                    )
        return self._finalize_outcomes(outcomes)

    @staticmethod
    def _finalize_outcomes(
        outcomes: List[ToolOutcome | None],
    ) -> Tuple[ToolOutcome, ...]:
        missing = [index for index, outcome in enumerate(outcomes) if outcome is None]
        if missing:
            formatted = ", ".join(str(index) for index in missing)
            raise RuntimeError(f"Tool outcomes are missing at indexes: {formatted}")
        return tuple(outcome for outcome in outcomes if outcome is not None)

    @staticmethod
    def _outcomes_to_responses(
        intents: Tuple[ToolIntent, ...],
        outcomes: Tuple[ToolOutcome, ...],
    ) -> ToolResponses:
        if len(intents) != len(outcomes):
            raise ValueError("Each legacy tool call must have exactly one outcome")
        direct_modes = {"direct", "handoff", "call_as_response"}
        return_directly = bool(outcomes) and all(
            outcome.status == "completed" and outcome.feedback.name in direct_modes
            for outcome in outcomes
        )
        tool_calls = []
        for intent, outcome in zip(intents, outcomes):
            if outcome.intent_id != intent.id:
                raise ValueError("Tool outcomes must preserve intent ordering")
            arguments = outcome.metadata.get("arguments", intent.arguments)
            tool_calls.append(
                ToolCall(
                    id=outcome.intent_id,
                    name=outcome.tool_name,
                    parameters=dict(arguments),
                    result=outcome.result,
                    error=(
                        outcome.error.message if outcome.error is not None else None
                    ),
                )
            )
        return ToolResponses(
            return_directly=return_directly,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _legacy_calls_to_intents(
        tool_callings: List[Tuple[str, str, Any]],
    ) -> Tuple[ToolIntent, ...]:
        return tuple(
            ToolIntent(
                id=tool_id,
                name=tool_name,
                arguments=coerce_tool_params(tool_name, tool_params),
            )
            for tool_id, tool_name, tool_params in tool_callings
        )

    def forward(
        self,
        tool_callings: List[Tuple[str, str, Any]],
        message: Optional[Any] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> ToolResponses:
        """Executes tool calls with tool config logic.

        Args:
            tool_callings:
                A list of tuples containing the tool id, name and parameters.
                !!! example
                    [('123121', 'tool_name1', {'parameter1': 'value1'}),
                    ('322', 'tool_name2', {})]
            messages:
                The current messages (chat history) for the `handoff` functionality.
            message:
                The original message/envelope passed to the parent Agent.
            vars:
                Extra kwargs to be used in tools.

        Returns:
            ToolResponses:
                Structured object containing all tool call results.
        """
        intents = self._legacy_calls_to_intents(tool_callings)
        outcomes = self.execute_intents(
            intents,
            message=message,
            messages=messages,
            vars=vars,
        )
        return self._outcomes_to_responses(intents, outcomes)

    async def aforward(
        self,
        tool_callings: List[Tuple[str, str, Any]],
        message: Optional[Any] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> ToolResponses:
        """Async version of forward. Executes tool calls with logic for
        `handoff`, `return_direct`.

        Args:
            tool_callings:
                A list of tuples containing the tool id, name and parameters.
                !!! example
                    [('123121', 'tool_name1', {'parameter1': 'value1'}),
                    ('322', 'tool_name2', {})]
            messages:
                The current messages (chat history) for the `handoff` functionality.
            message:
                The original message/envelope passed to the parent Agent.
            vars:
                Extra kwargs to be used in tools.

        Returns:
            ToolResponses:
                Structured object containing all tool call results.
        """
        intents = self._legacy_calls_to_intents(tool_callings)
        outcomes = await self.aexecute_intents(
            intents,
            message=message,
            messages=messages,
            vars=vars,
        )
        return self._outcomes_to_responses(intents, outcomes)
