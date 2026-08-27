import asyncio
import inspect
import weakref
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    get_type_hints,
)

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
from msgflux.nn.hooks.events import AfterTool, BeforeTool
from msgflux.nn.modules.container import ModuleDict
from msgflux.nn.modules.module import Module
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
from msgflux.tools.definitions import ToolCatalog, ToolSpec
from msgflux.tools.handles import ToolLibraryHandle
from msgflux.tools.helpers import (
    RUNTIME_BACKGROUND_PARAM,
    coerce_tool_params,
    is_agent_tool_impl,
    is_background_capable,
    is_reserved_tool_kind,
    normalize_background_capabilities,
    should_copy_injected_messages,
    should_dispatch_background,
)
from msgflux.tools.responses import ToolCall, ToolResponses
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
    if ToolLibraryOperator.is_operator_tool(impl):
        inherited_config = getattr(type(impl), "tool_config", {})
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
        annotations = getattr(impl, "annotations", None)
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

    if tool_config.get("spawn"):
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


@dataclass(frozen=True)
class ToolExecutionPlan:
    """Validated dispatch decision for one logical tool call."""

    tool_id: str
    tool_name: str
    tool: Tool
    config: Mapping[str, Any]
    visible_arguments: Mapping[str, Any]
    call_arguments: Mapping[str, Any]
    mode: Literal["foreground", "background", "spawn", "call_as_response"]
    return_direct: bool


class ToolLibrary(Module, metaclass=AutoParams):
    """ToolLibrary is a Module type that manage tool calls over the tool library."""

    _event_source_type = "tool_library"

    def __init__(
        self,
        name: str,
        tools: List[Callable],
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        task_store: Any | None = None,
        extensions: Optional[List[ToolLibraryExtension]] = None,
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
            setup, or cleanup under one removable owner.
        """
        super().__init__()
        self.set_name(f"{name}_tool_library")
        self.library = ModuleDict()
        self.register_buffer("tool_configs", {})
        self.register_buffer("mcp_clients", {})
        self._task_store = task_store
        self._agent_inbox: Optional[AgentInbox] = None
        self._disabled_background_task_tool_names: set[str] = set()
        self._handle: Optional[ToolLibraryHandle] = None
        self._background_dispatcher: Optional[BackgroundTaskDispatcher] = None
        self._lifecycle_owner_ref: Optional[weakref.ReferenceType[Module]] = None
        self.extensions = ModuleDict()
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
        extension: ToolLibraryExtension,
    ) -> ToolLibraryExtensionHandle:
        """Install a named library extension and return its ownership handle."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("`name` must be a non-empty string")
        if not isinstance(extension, ToolLibraryExtension):
            raise TypeError(
                f"`extension` must be a ToolLibraryExtension, given `{type(extension)}`"
            )
        if name in self.extensions:
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
        return name in self.extensions

    def remove_extension(self, name: str) -> None:
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

        if metadata.name in self.library.keys():
            raise ValueError(
                f"The tool name `{metadata.name}` is already in tool library"
            )
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

    def clear(self):
        self.library.clear()
        self.tool_configs.clear()
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

        # Let the bucket validate, retain, and refresh its captured state.
        bucket.add(metadata)
        self._sync_bucket_presentation(bucket_name, bucket)

    def _remove_from_bucket(self, bucket_name: str, tool_name: str) -> ToolMetadata:
        bucket_tool = self.library[bucket_name]
        bucket = getattr(bucket_tool, "impl", None)
        if not isinstance(bucket, ToolBucket):
            raise ValueError(f"The bucket tool `{bucket_name}` cannot release tools.")
        metadata = bucket.remove(tool_name)
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
        loaded = (
            messages.get_loaded_tools(self.name)
            if isinstance(messages, ChatMessages)
            else set()
        )
        tools: list[ToolSpec] = []
        search_tool: ToolSpec | None = None

        for tool_name, tool in self.library.items():
            if tool_name == "tool_search":
                search_tool = ToolSpec.from_function_schema(
                    tool.get_json_schema(),
                    annotations=self._public_annotations(tool),
                )
                bucket = getattr(tool, "impl", None)
                if isinstance(bucket, ToolBucket):
                    for metadata in bucket.tools.values():
                        deferred_tool = self._tool_from_metadata(metadata)
                        tools.append(
                            ToolSpec.from_function_schema(
                                deferred_tool.get_json_schema(),
                                annotations=self._public_annotations(deferred_tool),
                                defer_loading=True,
                                loaded=metadata.name in loaded,
                            )
                        )
                continue
            tools.append(
                ToolSpec.from_function_schema(
                    tool.get_json_schema(),
                    annotations=self._public_annotations(tool),
                )
            )

        return ToolCatalog(
            tools=tools,
            catalog_id=self.name,
            search_tool=search_tool,
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

    def _resolve_tool(self, tool_name: str) -> tuple[Tool, Mapping[str, Any]] | None:
        if tool_name in self.library:
            return self.library[tool_name], self.tool_configs.get(tool_name, {})
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
        return self._tool_from_metadata(metadata), metadata.tool_config

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

    def _build_tool_argument_sets(  # noqa: C901
        self,
        *,
        tool: Tool,
        tool_name: str,
        tool_params: Any,
        config: Mapping[str, Any],
        message: Optional[Any],
        messages: List[Dict[str, Any]],
        vars: Mapping[str, Any],
        tool_call_id: str | None = None,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if config.get("handoff", False) or config.get("disable_input", False):
            visible_params: Dict[str, Any] = {}
        else:
            visible_params = coerce_tool_params(tool_name, tool_params)

        for param_name in config.get("_hidden_params") or {}:
            visible_params.pop(param_name, None)

        runtime_params: Dict[str, Any] = dict(runtime_arguments or {})
        if RUNTIME_BACKGROUND_PARAM in visible_params:
            runtime_params[RUNTIME_BACKGROUND_PARAM] = visible_params.pop(
                RUNTIME_BACKGROUND_PARAM
            )

        inject_vars = config.get("inject_vars", False)
        if inject_vars:
            if isinstance(inject_vars, list):
                selected_vars = {}
                for key in inject_vars:
                    if key not in vars:
                        subject = (
                            "agent"
                            if config.get("tool_kind") == "agent"
                            or is_agent_tool_impl(tool.impl)
                            else "tool"
                        )
                        raise ValueError(
                            f"The {subject} `{tool_name}` requires the injected "
                            f"parameter `{key}`, but it was not found."
                        )
                    selected_vars[key] = vars[key]
                if config.get("tool_kind") == "agent" or is_agent_tool_impl(tool.impl):
                    runtime_params["vars"] = selected_vars
                else:
                    runtime_params.update(selected_vars)
            elif inject_vars is True:
                runtime_params["vars"] = vars

        if config.get("inject_messages", False):
            if should_copy_injected_messages(tool, config):
                runtime_params["messages"] = deepcopy(messages)
            else:
                runtime_params["messages"] = messages

        if config.get("inject_message", False):
            runtime_params["message"] = message

        if config.get("inject_handle", False):
            context = get_execution_context()
            runtime_params["handle"] = self.get_handle().for_tool(
                tool_name=tool_name,
                agent_inbox=context.get("agent_inbox"),
                task_store=context.get("task_store"),
                message=message,
                messages=messages,
                vars=vars,
                tool_call_id=tool_call_id,
                activity_recorder=context.get("task_activity_recorder"),
            )

        return visible_params, runtime_params

    def _record_tool_activity(
        self,
        *,
        activity_recorder: Any,
        tool_name: str,
        tool: Tool,
        config: Mapping[str, Any],
        parameters: Mapping[str, Any] | None,
    ) -> None:
        if (
            activity_recorder is None
            or is_reserved_tool_kind(config)
            or ToolLibraryOperator.is_operator_tool(tool)
        ):
            return
        activity_recorder.tool_call(tool_name, parameters)

    def _prepare_tool_kwargs(
        self,
        *,
        tool: Tool,
        tool_name: str,
        tool_params: Any,
        config: Mapping[str, Any],
        message: Optional[Any],
        messages: List[Dict[str, Any]],
        vars: Mapping[str, Any],
        activity_recorder: Any,
        tool_call_id: str | None = None,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        visible_params, runtime_params = self._build_tool_argument_sets(
            tool=tool,
            tool_name=tool_name,
            tool_params=tool_params,
            config=config,
            message=message,
            messages=messages,
            vars=vars,
            tool_call_id=tool_call_id,
            runtime_arguments=runtime_arguments,
        )
        self._record_tool_activity(
            activity_recorder=activity_recorder,
            tool_name=tool_name,
            tool=tool,
            config=config,
            parameters=visible_params,
        )
        return visible_params, runtime_params

    def _build_execution_plan(
        self,
        *,
        tool_id: str,
        tool_name: str,
        tool: Tool,
        config: Mapping[str, Any],
        visible_arguments: Mapping[str, Any],
        runtime_arguments: Mapping[str, Any],
    ) -> ToolExecutionPlan:
        call_arguments = {**visible_arguments, **runtime_arguments}
        if config.get("spawn", False):
            mode = "spawn"
        elif should_dispatch_background(config=config, call_params=call_arguments):
            mode = "background"
        elif config.get("call_as_response", False):
            mode = "call_as_response"
        else:
            mode = "foreground"
            call_arguments["tool_call_id"] = tool_id
        return ToolExecutionPlan(
            tool_id=tool_id,
            tool_name=tool_name,
            tool=tool,
            config=config,
            visible_arguments=dict(visible_arguments),
            call_arguments=call_arguments,
            mode=mode,
            return_direct=bool(config.get("return_direct", False)),
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
            if self.has_lifecycle_hooks("before_tool"):
                event = self._run_lifecycle_hooks("before_tool", event)
            if not isinstance(event, BeforeTool):
                raise TypeError("before_tool handlers must return BeforeTool or None")
            owner = self._get_lifecycle_owner()
            if owner is not None and owner.has_lifecycle_hooks("before_tool"):
                event = owner._run_lifecycle_hooks("before_tool", event)
            if not isinstance(event, BeforeTool):
                raise TypeError("before_tool handlers must return BeforeTool or None")
            return event
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
            if self.has_lifecycle_hooks("before_tool"):
                event = await self._arun_lifecycle_hooks("before_tool", event)
            if not isinstance(event, BeforeTool):
                raise TypeError("before_tool handlers must return BeforeTool or None")
            owner = self._get_lifecycle_owner()
            if owner is not None and owner.has_lifecycle_hooks("before_tool"):
                event = await owner._arun_lifecycle_hooks("before_tool", event)
            if not isinstance(event, BeforeTool):
                raise TypeError("before_tool handlers must return BeforeTool or None")
            return event
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
    ) -> tuple[Tool, Mapping[str, Any]]:
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
        return self._tool_from_metadata(metadata), metadata.tool_config

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

    def _call_captured_tool(
        self,
        bucket_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        runtime_arguments: Mapping[str, Any] | None = None,
        message: Any = None,
        messages: Any = None,
        vars: Mapping[str, Any] | None = None,
        parent_tool_call_id: str | None = None,
        activity_recorder: Any = None,
    ) -> Any:
        """Execute a captured tool through its normalized library wrapper."""
        tool, config = self._resolve_captured_tool(bucket_name, tool_name)
        visible_params, runtime_params = self._prepare_tool_kwargs(
            tool=tool,
            tool_name=tool_name,
            tool_params=arguments,
            config=config,
            message=message,
            messages=messages if messages is not None else ChatMessages(),
            vars=vars if vars is not None else {},
            activity_recorder=(
                activity_recorder
                if activity_recorder is not None
                else get_execution_context().get("task_activity_recorder")
            ),
            tool_call_id=parent_tool_call_id,
            runtime_arguments=runtime_arguments,
        )
        call_params = {**visible_params, **runtime_params}
        call_params["tool_call_id"] = (
            f"{parent_tool_call_id}:{tool_name}"
            if parent_tool_call_id
            else f"{bucket_name}:{tool_name}"
        )
        return self._execute_prepared_tool(tool, call_params, visible_params)

    async def _acall_captured_tool(
        self,
        bucket_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        runtime_arguments: Mapping[str, Any] | None = None,
        message: Any = None,
        messages: Any = None,
        vars: Mapping[str, Any] | None = None,
        parent_tool_call_id: str | None = None,
        activity_recorder: Any = None,
    ) -> Any:
        """Async counterpart of :meth:`_call_captured_tool`."""
        tool, config = self._resolve_captured_tool(bucket_name, tool_name)
        visible_params, runtime_params = self._prepare_tool_kwargs(
            tool=tool,
            tool_name=tool_name,
            tool_params=arguments,
            config=config,
            message=message,
            messages=messages if messages is not None else ChatMessages(),
            vars=vars if vars is not None else {},
            activity_recorder=(
                activity_recorder
                if activity_recorder is not None
                else get_execution_context().get("task_activity_recorder")
            ),
            tool_call_id=parent_tool_call_id,
            runtime_arguments=runtime_arguments,
        )
        call_params = {**visible_params, **runtime_params}
        call_params["tool_call_id"] = (
            f"{parent_tool_call_id}:{tool_name}"
            if parent_tool_call_id
            else f"{bucket_name}:{tool_name}"
        )
        return await self._aexecute_prepared_tool(tool, call_params, visible_params)

    def forward(  # noqa: C901
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
                    ('322', 'tool_name2', '')]
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
        if messages is None:
            messages = ChatMessages()

        if vars is None:
            vars = {}

        activity_recorder = get_execution_context().get("task_activity_recorder")
        prepared_calls = []
        call_metadata = []
        tool_calls: List[ToolCall] = []
        return_directly = True if tool_callings else False

        for tool_id, tool_name, tool_params in tool_callings:
            resolved = self._resolve_tool(tool_name)
            if resolved is None:
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=tool_params,
                        error=f"Error: Tool `{tool_name}` not found.",
                    )
                )
                return_directly = False
                continue

            tool, config = resolved
            if config.get("defer_loading", False) and isinstance(
                messages, ChatMessages
            ):
                messages.load_tools(self.name, [tool_name])
            visible_arguments, runtime_arguments = self._prepare_tool_kwargs(
                tool=tool,
                tool_name=tool_name,
                tool_params=tool_params,
                config=config,
                message=message,
                messages=messages,
                vars=vars,
                activity_recorder=activity_recorder,
                tool_call_id=tool_id,
            )
            before_tool = self._run_before_tool_hook(
                tool_id=tool_id,
                tool_name=tool_name,
                arguments=visible_arguments,
            )
            if before_tool.block is not None:
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=dict(before_tool.arguments),
                        error=before_tool.block,
                    )
                )
                return_directly = False
                continue
            response_params = dict(before_tool.arguments)
            plan = self._build_execution_plan(
                tool_id=tool_id,
                tool_name=tool_name,
                tool=tool,
                config=config,
                visible_arguments=response_params,
                runtime_arguments=runtime_arguments,
            )

            if plan.mode == "spawn":
                return_directly = False
                F.spawn(plan.tool, **plan.call_arguments)
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=response_params,
                        result=f"The `{tool_name}` tool was dispatched. "
                        "This tool will not generate a return.",
                    )
                )
                continue

            if plan.mode == "background":
                return_directly = False
                tool_calls.append(
                    self.get_background_dispatcher().dispatch(
                        tool=plan.tool,
                        tool_id=tool_id,
                        tool_name=tool_name,
                        call_params=plan.call_arguments,
                        visible_params=response_params,
                        config=config,
                    )
                )
                continue

            if plan.mode == "call_as_response":
                tool_calls.append(
                    ToolCall(id=tool_id, name=tool_name, parameters=response_params)
                )
                return_directly = True
                continue

            if not plan.return_direct:
                return_directly = False

            prepared_calls.append(
                partial(
                    self._execute_prepared_tool,
                    plan.tool,
                    plan.call_arguments,
                    response_params,
                )
            )

            call_metadata.append(
                dotdict(
                    id=tool_id,
                    name=tool_name,
                    config=config,
                    params=response_params,
                )
            )

        if prepared_calls:
            results = F.scatter_gather(prepared_calls)
            for meta, result in zip(call_metadata, results):
                if isinstance(result, TaskError) and isinstance(
                    result.exception,
                    (AbortRequestedError, TaskInterruptRequestedError),
                ):
                    raise result.exception
                tool_calls.append(
                    ToolCall(
                        id=meta.id,
                        name=meta.name,
                        parameters=dict(meta.params),
                        result=None if isinstance(result, TaskError) else result,
                        error=str(result) if isinstance(result, TaskError) else None,
                    )
                )

        return ToolResponses(return_directly=return_directly, tool_calls=tool_calls)

    async def aforward(  # noqa: C901
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
                    ('322', 'tool_name2', '')]
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
        if messages is None:
            messages = ChatMessages()

        if vars is None:
            vars = {}

        activity_recorder = get_execution_context().get("task_activity_recorder")
        prepared_calls = []
        call_metadata = []
        tool_calls: List[ToolCall] = []
        return_directly = True if tool_callings else False

        for tool_id, tool_name, tool_params in tool_callings:
            resolved = self._resolve_tool(tool_name)
            if resolved is None:
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=tool_params,
                        error=f"Error: Tool `{tool_name}` not found.",
                    )
                )
                return_directly = False
                continue

            tool, config = resolved
            if config.get("defer_loading", False) and isinstance(
                messages, ChatMessages
            ):
                messages.load_tools(self.name, [tool_name])
            visible_arguments, runtime_arguments = self._prepare_tool_kwargs(
                tool=tool,
                tool_name=tool_name,
                tool_params=tool_params,
                config=config,
                message=message,
                messages=messages,
                vars=vars,
                activity_recorder=activity_recorder,
                tool_call_id=tool_id,
            )
            before_tool = await self._arun_before_tool_hook(
                tool_id=tool_id,
                tool_name=tool_name,
                arguments=visible_arguments,
            )
            if before_tool.block is not None:
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=dict(before_tool.arguments),
                        error=before_tool.block,
                    )
                )
                return_directly = False
                continue
            response_params = dict(before_tool.arguments)
            plan = self._build_execution_plan(
                tool_id=tool_id,
                tool_name=tool_name,
                tool=tool,
                config=config,
                visible_arguments=response_params,
                runtime_arguments=runtime_arguments,
            )

            if plan.mode == "spawn":
                return_directly = False
                await F.aspawn(plan.tool, **plan.call_arguments)
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=response_params,
                        result=f"The `{tool_name}` tool was dispatched. "
                        "This tool will not generate a return.",
                    )
                )
                continue

            if plan.mode == "background":
                return_directly = False
                tool_calls.append(
                    self.get_background_dispatcher().dispatch(
                        tool=plan.tool,
                        tool_id=tool_id,
                        tool_name=tool_name,
                        call_params=plan.call_arguments,
                        visible_params=response_params,
                        config=config,
                    )
                )
                continue

            if plan.mode == "call_as_response":
                tool_calls.append(
                    ToolCall(id=tool_id, name=tool_name, parameters=response_params)
                )
                return_directly = True
                continue

            if not plan.return_direct:
                return_directly = False

            prepared_calls.append(
                partial(
                    self._aexecute_prepared_tool,
                    plan.tool,
                    plan.call_arguments,
                    response_params,
                )
            )

            call_metadata.append(
                dotdict(
                    id=tool_id,
                    name=tool_name,
                    config=config,
                    params=response_params,
                )
            )

        if prepared_calls:
            results = await F.ascatter_gather(prepared_calls)
            for meta, result in zip(call_metadata, results):
                if isinstance(result, TaskError) and isinstance(
                    result.exception,
                    (AbortRequestedError, TaskInterruptRequestedError),
                ):
                    raise result.exception
                tool_calls.append(
                    ToolCall(
                        id=meta.id,
                        name=meta.name,
                        parameters=dict(meta.params),
                        result=None if isinstance(result, TaskError) else result,
                        error=str(result) if isinstance(result, TaskError) else None,
                    )
                )
        return ToolResponses(return_directly=return_directly, tool_calls=tool_calls)
