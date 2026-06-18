import asyncio
import inspect
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

import msgflux.nn.functional as F
from msgflux.auto import AutoParams
from msgflux.core.dotdict import dotdict
from msgflux.exceptions import TaskError
from msgflux.logger import logger
from msgflux.nn.modules.container import ModuleDict
from msgflux.nn.modules.module import Module
from msgflux.protocols.mcp import (
    MCPClient,
    convert_mcp_schema_to_tool_schema,
    extract_tool_result_text,
    filter_tools,
)
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    ToolNotificationHandle,
)
from msgflux.runtime.background import BackgroundTaskDispatcher
from msgflux.runtime.context import get_execution_context
from msgflux.runtime.task_tools import TaskRuntimeTools
from msgflux.runtime.tool_search import ToolSearchRuntime
from msgflux.tasks import InMemoryTaskStore, TaskHandle
from msgflux.telemetry.span import (
    aset_tool_attributes,
    set_tool_attributes,
)
from msgflux.tools.handles import ToolLibraryHandle
from msgflux.tools.helpers import (
    RUNTIME_BACKGROUND_PARAM as _RUNTIME_BACKGROUND_PARAM,
)
from msgflux.tools.helpers import (
    is_agent_tool_impl as _is_agent_tool_impl,
)
from msgflux.tools.helpers import (
    should_copy_injected_messages as _should_copy_injected_messages,
)
from msgflux.tools.helpers import (
    uses_library_injection as _uses_library_injection,
)
from msgflux.tools.responses import ToolCall, ToolResponses
from msgflux.utils.chat import generate_tool_json_schema
from msgflux.utils.inspect import fn_has_parameters, get_fn_param_defaults
from msgflux.utils.msgspec import restore_transport_value
from msgflux.utils.tenacity import apply_retry, default_tool_retry


class Tool(Module):
    """Tool is Module type that provide a json schema to tools."""

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
        tc = config or {}
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
    ):
        super().__init__()
        self.set_name(name)
        self.set_description(description)
        self.register_buffer("display_name", display_name or name)
        self.register_buffer("usage_guidance", usage_guidance)
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

    @set_tool_attributes(execution_type="local")
    def forward(self, **kwargs):
        kwargs = self._restore_transport_params(kwargs)
        kwargs = self._strip_none_default_kwargs(kwargs)
        if inspect.iscoroutinefunction(self.impl):
            return F.wait_for(self.impl, **kwargs)
        return self.impl(**kwargs)

    @aset_tool_attributes(execution_type="local")
    async def aforward(self, *args, **kwargs):
        kwargs = self._restore_transport_params(kwargs)
        kwargs = self._strip_none_default_kwargs(kwargs)
        if hasattr(self.impl, "acall"):
            return await self.impl.acall(*args, **kwargs)
        elif inspect.iscoroutinefunction(self.impl):
            return await self.impl(*args, **kwargs)
        # Fall back to sync call in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.impl(*args, **kwargs))


def _convert_module_to_nn_tool(impl: Callable) -> Tool:  # noqa: C901
    """Convert a callable in nn.Tool."""
    tool_config = dotdict(deepcopy(getattr(impl, "tool_config", dotdict())))

    name_overridden = tool_config.pop("name_overridden", None)
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
            getattr(impl, "description", None)
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
        if inspect.isclass(impl):
            impl = impl()  # Initialized
            display_name = display_name or getattr(impl, "display_name", None)
            usage_guidance = usage_guidance or getattr(impl, "usage_guidance", None)

        # Now extract annotations (after instantiation for classes)
        annotations = (
            getattr(impl, "annotations", None)
            or getattr(impl, "__annotations__", None)
            or getattr(impl.__call__, "__annotations__", None)
        )
        if annotations is None:
            if fn_has_parameters(impl.__call__):
                raise NotImplementedError(
                    "To transform a class in `nn.Tool` is necessary "
                    "to implement annotations of types hint in "
                    "`self.annotations`, `self.__annotations__` or in `def __call__`"
                )
            annotations = {}

    # Case 2: Function
    elif inspect.isfunction(impl) or inspect.iscoroutinefunction(impl):
        if hasattr(impl, "__doc__") and impl.__doc__ is not None:
            doc = impl.__doc__
        else:
            raise NotImplementedError(
                "To transform a function into a `nn.Tool` "
                "is necessary to implement a docstring"
            )

        annotations = impl.__annotations__

        if annotations is None:
            if fn_has_parameters(impl):
                raise NotImplementedError(
                    "To transform a function into a `nn.Tool` "
                    "is necessary to implement parameters "
                    "annotations of types hint "
                )
            annotations = {}

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

    tool_config["tool_kind"] = "agent" if _is_agent_tool_impl(impl) else "tool"

    annotations = dict(annotations)

    if tool_config.get("handoff", False) or tool_config.get("disable_input", False):
        annotations = {}  # pass only the model state
    else:
        if tool_config.get("inject_message", False):
            annotations.pop("message", None)
        if tool_config.get("inject_messages", False):
            annotations.pop("messages", None)
        if tool_config.get("inject_vars", False):
            annotations.pop("vars", None)
        if tool_config.get("inject_task", False):
            annotations.pop("task", None)
        if tool_config.get("inject_notification", False):
            annotations.pop("notification", None)
        if _uses_library_injection(tool_config):
            annotations.pop("tool_library", None)
        if (
            tool_config.get("allow_background", False)
            and not tool_config.get("background", False)
        ):
            annotations[_RUNTIME_BACKGROUND_PARAM] = Optional[bool]

    if tool_config.get("spawn"):
        doc = "This tool will not generate a return. \n" + doc
    if tool_config.get("background"):
        doc = "This tool runs in the background and returns a task id. \n" + doc
    elif tool_config.get("allow_background", False):
        doc = (
            "This tool can run in the background when "
            f"`{_RUNTIME_BACKGROUND_PARAM}=true`; otherwise it runs normally. \n"
            + doc
        )

    return LocalTool(
        name=name,
        description=doc,
        annotations=annotations,
        tool_config=tool_config,
        impl=impl,
        display_name=display_name or name,
        usage_guidance=usage_guidance,
    )


class ToolLibrary(Module, metaclass=AutoParams):
    """ToolLibrary is a Module type that manage tool calls over the tool library."""

    def __init__(
        self,
        name: str,
        tools: List[Callable],
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
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
        """
        super().__init__()
        self.set_name(f"{name}_tool_library")
        self.library = ModuleDict()
        self.register_buffer("tool_configs", {})
        self.register_buffer("mcp_clients", {})
        self.task_store = InMemoryTaskStore()
        self.agent_inbox = AgentInbox(owner=f"{name}_tool_library")
        self._runtime_tool_names: set[str] = set()
        self._task_runtime_enabled = False
        self._agent_task_runtime_enabled = False
        self.background_dispatcher = BackgroundTaskDispatcher(
            self,
            tool_call_factory=ToolCall,
        )
        self.task_runtime_tools = TaskRuntimeTools(
            self,
            register_tool=self._register_runtime_tool,
            runtime_tool_names=self._runtime_tool_names,
        )
        self.tool_search_runtime = ToolSearchRuntime(
            self,
            register_tool=self._register_runtime_tool,
            runtime_tool_names=self._runtime_tool_names,
        )
        for tool in tools:
            self.add(tool)
        if mcp_servers:
            self._initialize_mcp_clients(mcp_servers)

    def add(self, tool: Callable):
        """Add a local tool in library."""
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        if name in self.library.keys():
            raise ValueError(f"The tool name `{name}` is already in tool library")
        if not isinstance(tool, Tool):
            tool = _convert_module_to_nn_tool(tool)

        # Store tool config (may be empty dict for local tools)
        self.tool_configs[tool.name] = getattr(tool, "tool_config", {})
        self.library.update({tool.name: tool})
        self._apply_tool_registration_effects(tool.name)

    def remove(self, tool_name: str):
        if tool_name in self.library.keys():
            self.library.pop(tool_name)
            self.tool_configs.pop(tool_name, None)
            self.tool_search_runtime.discard_loaded_tool(tool_name)
            self._sync_on_demand_runtime_tools()
        else:
            raise ValueError(f"The tool name `{tool_name}` is not in tool library")

    def clear(self):
        self.library.clear()
        self.tool_configs.clear()
        for mcp_data in self.mcp_clients.values():
            F.wait_for(mcp_data["client"].disconnect)
        self.mcp_clients.clear()
        self._task_runtime_enabled = False
        self._agent_task_runtime_enabled = False
        self._runtime_tool_names.clear()
        self.background_dispatcher.clear()
        self.task_runtime_tools.reset()
        self.tool_search_runtime.reset()

    def _initialize_mcp_clients(self, mcp_servers: List[Dict[str, Any]]):
        """Initialize MCP clients from server configurations."""
        for server_config in mcp_servers:
            namespace = server_config.get("name")
            if not namespace:
                raise ValueError("MCP server config must include 'name' field")

            transport_type = server_config.get("transport", "stdio")

            # Create client based on transport type
            if transport_type == "stdio":
                command = server_config.get("command")
                if not command:
                    raise ValueError(
                        f"MCP server '{namespace}' stdio transport requires 'command'"
                    )
                client = MCPClient.from_stdio(
                    command=command,
                    args=server_config.get("args"),
                    cwd=server_config.get("cwd"),
                    env=server_config.get("env"),
                    timeout=server_config.get("timeout", 30.0),
                )
            elif transport_type == "http":
                base_url = server_config.get("base_url")
                if not base_url:
                    raise ValueError(
                        f"MCP server '{namespace}' http transport requires 'base_url'"
                    )
                client = MCPClient.from_http(
                    base_url=base_url,
                    timeout=server_config.get("timeout", 30.0),
                    headers=server_config.get("headers"),
                    auth=server_config.get("auth"),
                )
            else:
                raise ValueError(
                    f"Unknown transport type: {transport_type}. "
                    "Supported types: 'stdio', 'http'"
                )

            # Connect and list tools with error handling
            try:
                F.wait_for(client.connect)
                all_tools = F.wait_for(client.list_tools, use_cache=False)

                # Apply filters
                include_tools = server_config.get("include_tools")
                exclude_tools = server_config.get("exclude_tools")
                filtered_tools = filter_tools(all_tools, include_tools, exclude_tools)

                # Create MCPTool for each remote tool
                tool_configs = server_config.get("tool_config", {})
                for mcp_tool_info in filtered_tools:
                    tool_config = tool_configs.get(mcp_tool_info.name, {})

                    # Create MCPTool instance
                    mcp_tool = MCPTool(
                        name=mcp_tool_info.name,
                        mcp_client=client,
                        mcp_tool_info=mcp_tool_info,
                        namespace=namespace,
                        config=tool_config,
                    )

                    # Add to library (will have name like "namespace__tool_name")
                    self.library.update({mcp_tool.name: mcp_tool})
                    self.tool_configs[mcp_tool.name] = mcp_tool.tool_config
                    self._apply_tool_registration_effects(mcp_tool.name)

                self.mcp_clients[namespace] = {
                    "client": client,
                    "tools": filtered_tools,
                    "tool_config": tool_configs,
                }

                logger.debug(
                    f"Successfully connected to MCP server `{namespace}` "
                    f"with {len(filtered_tools)} tools"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize MCP server '{namespace}': {e!s}",
                    exc_info=True,
                )
                # Continue with other servers instead of failing completely

    def get_tools(self) -> Iterator[Dict[str, Tool]]:
        return self.library.items()

    def get_tool_names(self) -> List[str]:
        """Get names of all tools."""
        return list(self.library.keys())

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
        schemas = []
        for tool_name in self.library:
            if not self._is_tool_exposed(tool_name):
                continue
            schemas.append(self.library[tool_name].get_json_schema())

        return schemas

    def get_tool_annotations(self) -> Dict[str, Dict[str, Any]]:
        """Return local tool annotations keyed by tool name."""
        annotations = {}
        for tool_name, tool in self.library.items():
            if not self._is_tool_exposed(tool_name):
                continue
            annotations[tool_name] = {
                name: hint
                for name, hint in tool.get_module_annotations().items()
                if name != "return"
            }
        return annotations

    def set_agent_inbox(self, agent_inbox: AgentInbox) -> None:
        self.agent_inbox = agent_inbox

    # --- Tool Visibility Helpers ---

    def _apply_tool_registration_effects(self, tool_name: str) -> None:
        config = self.tool_configs.get(tool_name, {})
        can_run_background = config.get("background", False) or config.get(
            "allow_background", False
        )
        if can_run_background:
            self.task_runtime_tools.ensure_base_tools()
            self._task_runtime_enabled = True
        if can_run_background and config.get("tool_kind") == "agent":
            self.task_runtime_tools.ensure_agent_tools()
            self._agent_task_runtime_enabled = True
        if config.get("on_demand", False):
            self.tool_search_runtime.ensure_tool()

    def _get_on_demand_tool_names(self) -> List[str]:
        return self.tool_search_runtime.get_on_demand_tool_names()

    def _is_tool_exposed(self, tool_name: str) -> bool:
        if tool_name in self._runtime_tool_names:
            return True
        return self.tool_search_runtime.is_tool_exposed(tool_name)

    def _load_on_demand_tools(self, tool_names: List[str]) -> List[str]:
        return self.tool_search_runtime.load_tools(tool_names)

    def _sync_on_demand_runtime_tools(self) -> None:
        self.tool_search_runtime.sync()

    # --- Task Runtime Registration ---

    def _ensure_task_runtime_tools(self) -> None:
        self.task_runtime_tools.ensure_base_tools()
        self._task_runtime_enabled = True

    def _ensure_agent_task_runtime_tools(self) -> None:
        self.task_runtime_tools.ensure_agent_tools()
        self._agent_task_runtime_enabled = True

    def _ensure_on_demand_runtime_tools(self) -> None:
        self.tool_search_runtime.ensure_tool()

    def _register_runtime_tool(
        self,
        *,
        name: str,
        description: str,
        annotations: Dict[str, Any],
        impl: Callable,
    ) -> None:
        tool = LocalTool(
            name=name,
            description=description,
            annotations=annotations,
            tool_config={},
            impl=impl,
        )
        if name in self.library and self.tool_configs.get(name):
            raise ValueError(
                f"The runtime tool `{name}` conflicts with an existing tool."
            )
        if name in self.library and name not in self._runtime_tool_names:
            raise ValueError(
                f"The runtime tool `{name}` conflicts with an existing tool."
            )
        self.library.update({name: tool})
        self.tool_configs[name] = {}

    # --- Task Runtime Tools ---

    def _task_status(self, task_id: str) -> Dict[str, Any]:
        return self.task_runtime_tools.task_status(task_id)

    def _task_list(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.task_runtime_tools.task_list(status)

    def _task_output(self, task_id: str) -> Any:
        return self.task_runtime_tools.task_output(task_id)

    def _tool_search(
        self,
        query: str,
        max_results: Optional[int] = 5,
    ) -> Dict[str, Any]:
        return self.tool_search_runtime.tool_search(query, max_results=max_results)

    def _task_activity(
        self,
        task_id: str,
        limit: Optional[int] = 10,
    ) -> Any:
        return self.task_runtime_tools.task_activity(task_id, limit=limit)

    def _task_wait(self, task_id: str, timeout: Optional[float] = None) -> Any:
        return self.task_runtime_tools.task_wait(task_id, timeout=timeout)

    def _task_stop(self, task_id: str) -> Dict[str, Any]:
        return self.task_runtime_tools.task_stop(task_id)

    def _task_message(self, task_id: str, message: str) -> Dict[str, Any]:
        return self.task_runtime_tools.task_message(task_id, message)

    # --- Task Runtime Helpers ---

    def _build_task_result(self, *, task_id: str, task: Optional[Any]) -> Any:
        return self.task_runtime_tools.build_task_result(task_id=task_id, task=task)

    def _build_task_timeout_result(
        self, *, task_id: str, task: Optional[Any]
    ) -> Dict[str, Any]:
        return self.task_runtime_tools.build_task_timeout_result(
            task_id=task_id,
            task=task,
        )

    def _build_task_timing_fields(self, task: Any) -> Dict[str, Any]:
        return self.task_runtime_tools.build_task_timing_fields(task)

    def _format_task_activity_entry(self, activity: Any) -> str:
        return self.task_runtime_tools.format_task_activity_entry(activity)

    def _select_on_demand_tools(self, requested: List[str]) -> List[str]:
        return self.tool_search_runtime.select_tools(requested)

    def _search_on_demand_tools(self, *, query: str, max_results: int) -> List[str]:
        return self.tool_search_runtime.search_tools(
            query=query,
            max_results=max_results,
        )

    def _build_background_dispatch_result(
        self,
        *,
        task_id: str,
        tool_name: str,
        task_kind: str,
    ) -> str:
        return self.task_runtime_tools.build_background_dispatch_result(
            task_id=task_id,
            tool_name=tool_name,
            task_kind=task_kind,
        )

    @staticmethod
    def _parse_utc_timestamp(value: Optional[str]) -> Optional[float]:
        return TaskRuntimeTools.parse_utc_timestamp(value)

    @staticmethod
    def _truncate_activity_text(value: str, *, limit: int = 140) -> str:
        return TaskRuntimeTools.truncate_activity_text(value, limit=limit)

    def _register_task_future(self, task_id: str, future: Any) -> None:
        self.background_dispatcher.register_task_future(task_id, future)

    def _get_task_future(self, task_id: str) -> Optional[Any]:
        return self.background_dispatcher.get_task_future(task_id)

    def _cleanup_task_future(self, task_id: str, future: Any) -> None:
        self.background_dispatcher.cleanup_task_future(task_id, future)

    def _register_task_inbox(self, task_id: str, inbox: AgentInbox) -> None:
        self.background_dispatcher.register_task_inbox(task_id, inbox)

    def _get_task_inbox(self, task_id: str) -> Optional[AgentInbox]:
        return self.background_dispatcher.get_task_inbox(task_id)

    # --- Tool Call Preparation ---

    def _build_call_params(  # noqa: C901
        self,
        *,
        tool: Tool,
        tool_name: str,
        tool_params: Any,
        config: Mapping[str, Any],
        message: Optional[Any],
        messages: List[Dict[str, Any]],
        vars: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if config.get("handoff", False) or config.get("disable_input", False):
            call_params: Dict[str, Any] = {}
        else:
            call_params = self._coerce_tool_params(tool_name, tool_params)

        inject_vars = config.get("inject_vars", False)
        if inject_vars:
            if isinstance(inject_vars, list):
                for key in inject_vars:
                    if key not in vars:
                        raise ValueError(
                            f"The tool `{tool_name}` requires the injected "
                            f"parameter `{key}`, but it was not found."
                        )
                    call_params[key] = vars[key]
            elif inject_vars is True:
                call_params["vars"] = vars

        if config.get("inject_messages", False):
            if _should_copy_injected_messages(tool, config):
                call_params["messages"] = deepcopy(messages)
            else:
                call_params["messages"] = messages

        if config.get("inject_message", False):
            call_params["message"] = message

        if (
            config.get("inject_notification", False)
            and config.get("tool_kind") != "agent"
        ):
            call_params["notification"] = self._build_notification_handle(
                tool_name=tool_name
            )

        if _uses_library_injection(config):
            call_params["tool_library"] = ToolLibraryHandle(self)

        return call_params

    def _should_dispatch_background(
        self,
        *,
        config: Mapping[str, Any],
        call_params: Dict[str, Any],
    ) -> bool:
        if config.get("background", False):
            call_params.pop(_RUNTIME_BACKGROUND_PARAM, None)
            return True
        if not config.get("allow_background", False):
            return False
        return call_params.pop(_RUNTIME_BACKGROUND_PARAM, False) is True

    @staticmethod
    def _coerce_tool_params(tool_name: str, tool_params: Any) -> Dict[str, Any]:
        if tool_params is None:
            return {}
        if isinstance(tool_params, Mapping):
            return dict(tool_params)
        raise TypeError(
            f"Tool `{tool_name}` parameters must be a mapping or None, "
            f"given `{type(tool_params)}`."
        )

    def _build_call_parameters_for_response(
        self, params: Optional[Mapping[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if params is None:
            return None
        if hasattr(params, "to_dict"):
            parameters = params.to_dict()
        else:
            parameters = dict(params)
        for key in (
            "vars",
            "messages",
            "message",
            "task",
            "notification",
            "scope",
            "tool_library",
            "tool_call_id",
            _RUNTIME_BACKGROUND_PARAM,
        ):
            parameters.pop(key, None)
        return parameters

    def _build_notification_handle(
        self,
        *,
        tool_name: str,
        ref: Optional[str] = None,
        agent_inbox: Optional[AgentInbox] = None,
    ) -> ToolNotificationHandle:
        execution_context = get_execution_context()
        inbox = agent_inbox or execution_context.get("agent_inbox") or self.agent_inbox
        metadata = {"tool": tool_name}
        return ToolNotificationHandle(
            inbox,
            ref=ref,
            metadata=metadata,
        )

    # --- Background Task Execution ---

    def _run_background_tool(
        self,
        *,
        tool: Tool,
        task_handle: TaskHandle,
        tool_name: str,
        call_params: Dict[str, Any],
        execution_scope: Optional[Dict[str, Any]] = None,
        agent_inbox: Optional[AgentInbox] = None,
    ) -> Any:
        return self.background_dispatcher.run_tool(
            tool=tool,
            task_handle=task_handle,
            tool_name=tool_name,
            call_params=call_params,
            execution_scope=execution_scope,
            agent_inbox=agent_inbox,
        )

    def _resume_background_agent_task(self, *, task: Any, message: str) -> str:
        return self.background_dispatcher.resume_agent_task(task=task, message=message)

    def _log_background_task_failure(self, future: Any) -> None:
        self.background_dispatcher.log_task_failure(future)

    # --- Background Task Dispatch ---

    def _dispatch_background_tool(
        self,
        *,
        tool: Tool,
        tool_id: str,
        tool_name: str,
        call_params: Dict[str, Any],
        config: Mapping[str, Any],
    ) -> ToolCall:
        return self.background_dispatcher.dispatch(
            tool=tool,
            tool_id=tool_id,
            tool_name=tool_name,
            call_params=call_params,
            config=config,
        )

    # --- Background Task Notifications ---

    def _publish_task_notification(
        self,
        *,
        task_id: str,
        tool_name: str,
        status: str,
        hint: str,
        agent_inbox: Optional[AgentInbox] = None,
    ) -> Any:
        return self.background_dispatcher.publish_task_notification(
            task_id=task_id,
            tool_name=tool_name,
            status=status,
            hint=hint,
            agent_inbox=agent_inbox,
        )

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
            messages = []

        if vars is None:
            vars = {}

        activity_recorder = get_execution_context().get("task_activity_recorder")
        prepared_calls = []
        call_metadata = []
        tool_calls: List[ToolCall] = []
        return_directly = True if tool_callings else False

        for tool_id, tool_name, tool_params in tool_callings:
            if tool_name not in self.library:
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

            # Get tool
            tool = self.library[tool_name]
            config = self.tool_configs.get(tool_name, {})
            if (
                activity_recorder is not None
                and tool_name not in self._runtime_tool_names
            ):
                activity_recorder.tool_call(tool_name, tool_params)
            call_params = self._build_call_params(
                tool=tool,
                tool_name=tool_name,
                tool_params=tool_params,
                config=config,
                message=message,
                messages=messages,
                vars=vars,
            )

            if config.get("spawn", False):
                return_directly = False
                F.spawn(tool, **call_params)
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=tool_params,
                        result=f"The `{tool_name}` tool was dispatched. "
                        "This tool will not generate a return.",
                    )
                )
                continue

            if self._should_dispatch_background(
                config=config,
                call_params=call_params,
            ):
                return_directly = False
                tool_calls.append(
                    self._dispatch_background_tool(
                        tool=tool,
                        tool_id=tool_id,
                        tool_name=tool_name,
                        call_params=call_params,
                        config=config,
                    )
                )
                continue

            if config.get(
                "call_as_response", False
            ):  # return function call as response
                tool_calls.append(
                    ToolCall(id=tool_id, name=tool_name, parameters=tool_params)
                )
                return_directly = True
                continue

            if not config.get("return_direct", False):
                return_directly = False

            # Add tool_call_id for telemetry
            call_params["tool_call_id"] = tool_id
            prepared_calls.append(partial(tool, **call_params))

            call_metadata.append(
                dotdict(
                    id=tool_id,
                    name=tool_name,
                    config=config,
                    params=call_params,
                )
            )

        if prepared_calls:
            results = F.scatter_gather(prepared_calls)
            for meta, result in zip(call_metadata, results):
                parameters = self._build_call_parameters_for_response(meta.params)
                tool_calls.append(
                    ToolCall(
                        id=meta.id,
                        name=meta.name,
                        parameters=parameters,
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
            messages = []

        if vars is None:
            vars = {}

        activity_recorder = get_execution_context().get("task_activity_recorder")
        prepared_calls = []
        call_metadata = []
        tool_calls: List[ToolCall] = []
        return_directly = True if tool_callings else False

        for tool_id, tool_name, tool_params in tool_callings:
            if tool_name not in self.library:
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

            # Get tool
            tool = self.library[tool_name]
            config = self.tool_configs.get(tool_name, {})
            if (
                activity_recorder is not None
                and tool_name not in self._runtime_tool_names
            ):
                activity_recorder.tool_call(tool_name, tool_params)
            call_params = self._build_call_params(
                tool=tool,
                tool_name=tool_name,
                tool_params=tool_params,
                config=config,
                message=message,
                messages=messages,
                vars=vars,
            )

            if config.get("spawn", False):
                return_directly = False
                await F.aspawn(tool, **call_params)
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=tool_params,
                        result=f"The `{tool_name}` tool was dispatched. "
                        "This tool will not generate a return.",
                    )
                )
                continue

            if self._should_dispatch_background(
                config=config,
                call_params=call_params,
            ):
                return_directly = False
                tool_calls.append(
                    self._dispatch_background_tool(
                        tool=tool,
                        tool_id=tool_id,
                        tool_name=tool_name,
                        call_params=call_params,
                        config=config,
                    )
                )
                continue

            if config.get(
                "call_as_response", False
            ):  # return function call as response
                tool_calls.append(
                    ToolCall(id=tool_id, name=tool_name, parameters=tool_params)
                )
                return_directly = True
                continue

            if not config.get("return_direct", False):
                return_directly = False

            # Add tool_call_id for telemetry
            call_params["tool_call_id"] = tool_id
            prepared_calls.append(partial(tool.acall, **call_params))

            call_metadata.append(
                dotdict(
                    id=tool_id,
                    name=tool_name,
                    config=config,
                    params=call_params,
                )
            )

        if prepared_calls:
            results = await F.ascatter_gather(prepared_calls)
            for meta, result in zip(call_metadata, results):
                parameters = self._build_call_parameters_for_response(meta.params)
                tool_calls.append(
                    ToolCall(
                        id=meta.id,
                        name=meta.name,
                        parameters=parameters,
                        result=None if isinstance(result, TaskError) else result,
                        error=str(result) if isinstance(result, TaskError) else None,
                    )
                )
        return ToolResponses(return_directly=return_directly, tool_calls=tool_calls)
