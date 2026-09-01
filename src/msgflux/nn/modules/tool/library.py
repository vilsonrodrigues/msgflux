import weakref
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
)

import msgflux.nn.functional as F
from msgflux.auto import AutoParams
from msgflux.chat_messages import ChatMessages
from msgflux.core.dotdict import dotdict
from msgflux.nn.extensions.tool_library import (
    BackgroundTasksExtension,
    MCPServersExtension,
    ToolLibraryExtension,
    ToolLibraryExtensionHandle,
    ToolSearchExtension,
)
from msgflux.nn.hooks import Hook
from msgflux.nn.modules.container import ModuleDict
from msgflux.nn.modules.module import Module
from msgflux.nn.modules.tool.execution_runtime import ToolLibraryExecutionMixin
from msgflux.nn.modules.tool.implementations import (
    MCPTool,
    Tool,
    _convert_metadata_to_local_tool,
    _inspect_tool_metadata,
    _metadata_from_tool,
)
from msgflux.nn.modules.tool.runtime import (
    ToolCatalogView,
    ToolChoice,
    ToolDefinitionCompiler,
    ToolExtension,
    ToolExtensionHandle,
    ToolExtensionRegistry,
    ToolRef,
    ToolRegistry,
)
from msgflux.nn.modules.tool.runtime import (
    ToolDefinition as RuntimeToolDefinition,
)
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    InMemoryAgentInboxStore,
)
from msgflux.runtime.background import BackgroundTaskDispatcher
from msgflux.runtime.context import get_execution_context
from msgflux.tasks import InMemoryTaskStore
from msgflux.tools.dataclasses import ToolMetadata
from msgflux.tools.definitions import ToolCatalog
from msgflux.tools.handles import ToolLibraryHandle
from msgflux.tools.types import ToolBackground, ToolBucket


class ToolLibrary(ToolLibraryExecutionMixin, Module, metaclass=AutoParams):
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
        definition = self.get_tool_definition(tool_name)
        return definition.metadata.get("execution_namespace") or definition.name

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

    @staticmethod
    def _compile_tool_metadata(metadata: ToolMetadata) -> ToolMetadata:
        """Compile one normalized declaration before any registration decision."""
        metadata.tool_config = dotdict(metadata.tool_config)
        metadata.tool_config.setdefault("defer_loading", False)
        tool = (
            metadata.source_tool
            if isinstance(metadata.source_tool, Tool)
            else _convert_metadata_to_local_tool(metadata)
        )
        if isinstance(metadata.source_tool, Tool):
            tool.register_buffer("tool_config", dotdict(metadata.tool_config))
        metadata.source_tool = tool
        metadata.definition = ToolDefinitionCompiler.compile(metadata, executor=tool)
        return metadata

    def add(self, tool: Callable) -> str:
        """Add a local tool in library."""
        if isinstance(tool, ToolMetadata):
            metadata = tool
        elif isinstance(tool, Tool):
            metadata = _metadata_from_tool(tool)
        else:
            metadata = _inspect_tool_metadata(tool)

        self._validate_new_tool_name(metadata.name)
        metadata = self._compile_tool_metadata(metadata)
        definition = metadata.definition
        if ToolBackground.is_background_definition(definition):
            background = self.extensions.get("background_tasks")
            if isinstance(background, BackgroundTasksExtension):
                background.validate_source(definition)

        # Deferred tools are held by the search bucket. Loading is thread-local.
        if definition.loading.deferred and "tool_search" not in self.library:
            if not self.has_extension("tool_search"):
                self.register_extension("tool_search", ToolSearchExtension())
            else:
                search_extension = self.extensions["tool_search"]
                self.add(next(iter(search_extension.tools())))

        # A matching registered bucket owns the tool instead of direct registration.
        bucket_name = ToolBucket.find_bucket(
            metadata,
            self.library,
            self.get_tool_definition,
        )
        if bucket_name is not None:
            self._add_to_bucket(bucket_name, metadata)
            return metadata.name

        capturing_bucket = ToolBucket.find_capturing_bucket(
            metadata.name,
            self.library,
            self.get_tool_definition,
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
            definition = self.get_tool_definition(tool_name)
            background = self.extensions.get("background_tasks")
            is_task_tool = isinstance(
                background, BackgroundTasksExtension
            ) and background.is_active_task_tool(
                library=self, tool_name=tool_name, definition=definition
            )
            was_background = not ToolBackground.is_reserved_definition(
                definition
            ) and ToolBackground.is_background_definition(definition)

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
            self.get_tool_definition,
        )
        if bucket_name is None:
            raise ValueError(f"The tool name `{tool_name}` is not in tool library")
        self._remove_from_bucket(bucket_name, tool_name)

    def _remove_registered_tool(self, tool_name: str) -> None:
        if tool_name in self.library:
            self.library.pop(tool_name)
        if self.registry.has(tool_name):
            self.registry.remove(tool_name)

    def clear(self):
        self.library.clear()
        self.registry.clear()
        for mcp_data in self.mcp_clients.values():
            F.wait_for(mcp_data["client"].disconnect)
        self.mcp_clients.clear()
        self._disabled_background_task_tool_names.clear()
        if self._background_dispatcher is not None:
            self._background_dispatcher.clear()

    def _register_tool(self, metadata: ToolMetadata) -> Tool:
        definition = metadata.definition
        if not isinstance(definition, RuntimeToolDefinition):
            raise RuntimeError(f"Tool `{metadata.name}` has not been compiled")
        tool = metadata.source_tool
        if not isinstance(tool, Tool):
            raise RuntimeError(f"Tool `{metadata.name}` has no compiled executor")

        # A bucket must be valid before it becomes visible in the library.
        captures = []
        if definition.kind == ToolBucket.tool_kind:
            ToolBucket.validate_registration(
                metadata,
                self.library,
                self.get_tool_definition,
            )
            captures = ToolBucket.find_capture_candidates(
                metadata.impl,
                self.library,
                self.get_tool_definition,
            )

            # Check every pending capture before changing the current library state.
            for captured_name, captured_tool in captures:
                captured_metadata = _metadata_from_tool(captured_tool)
                captured_metadata.definition = self.get_tool_definition(captured_name)
                metadata.impl.validate_capture(captured_metadata)

        # Register the public executor and its immutable definition together.
        metadata.ref = self.registry.add(definition)
        self.library.update({tool.name: tool})
        if isinstance(metadata.impl, ToolBucket):
            metadata.impl.refresh()
            self._sync_bucket_presentation(tool.name, metadata.impl)

        # An explicit re-add re-enables a builtin task control tool.
        if ToolBackground.is_reserved_definition(definition):
            self._disabled_background_task_tool_names.discard(tool.name)

        # Background-capable sources determine the shared task control surface.
        self._sync_background_task_tools_for_source(definition)

        # Move matching local tools into a newly registered bucket.
        for captured_name, captured_tool in captures:
            captured_metadata = _metadata_from_tool(captured_tool)
            captured_metadata.definition = self.get_tool_definition(captured_name)
            self.remove(captured_name)
            self._add_to_bucket(tool.name, captured_metadata)
        return tool

    def _add_to_bucket(self, bucket_name: str, metadata: ToolMetadata) -> None:
        # Resolve the bucket implementation before changing its captured tools.
        bucket_tool = self.library[bucket_name]
        bucket = getattr(bucket_tool, "impl", None)
        if not isinstance(bucket, ToolBucket):
            raise ValueError(f"The bucket tool `{bucket_name}` cannot capture tools.")

        if not isinstance(metadata.definition, RuntimeToolDefinition):
            metadata = self._compile_tool_metadata(metadata)
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
        current = self.get_tool_definition(bucket_name)
        if isinstance(getattr(bucket, "description", None), str):
            bucket_tool.set_description(bucket.description)
        annotations = bucket.patch_schema_annotations(
            bucket_tool.get_module_annotations()
        )
        if not isinstance(annotations, Mapping):
            raise TypeError("Bucket schema annotation patches must return a mapping.")
        bucket_tool.set_annotations(dict(annotations))
        usage_guidance = bucket.compose_usage_guidance(
            current.metadata.get("declared_usage_guidance")
        )
        bucket_tool.register_buffer("usage_guidance", usage_guidance)
        bucket_metadata = _metadata_from_tool(bucket_tool)
        self.registry.replace(
            ToolDefinitionCompiler.refresh_presentation(
                current,
                bucket_metadata,
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
        return {
            tool_name: self.get_tool_definition(tool_name).display_name or tool_name
            for tool_name in self.library
        }

    def get_tool_usage_guidance(
        self, tool_names: Optional[set[str]] = None
    ) -> List[Dict[str, str]]:
        """Return usage guidance metadata for tools that define it."""
        guidance = []
        display_names = self.get_tool_display_names()

        for tool_name in self.library:
            if tool_names is not None and tool_name not in tool_names:
                continue
            usage_guidance = self.get_tool_definition(tool_name).usage_guidance
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
        return self._build_tool_catalog_view(
            None,
            require_thread=False,
        ).portable_schemas()

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
        thread_id: str | None = None,
    ) -> ToolCatalogView:
        if require_thread and not isinstance(messages, ChatMessages):
            if not isinstance(thread_id, str) or not thread_id:
                raise TypeError("`messages` must be ChatMessages or set `thread_id`")
        if thread_id is None and isinstance(messages, ChatMessages):
            thread_id = messages.thread_id
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
        messages: ChatMessages | None = None,
        *,
        choice: ToolChoice | str | Mapping[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> ToolCatalogView:
        """Return an immutable definition view for one configured thread."""
        return self._build_tool_catalog_view(
            messages,
            choice=choice,
            require_thread=True,
            thread_id=thread_id,
        )

    @property
    def has_deferred_tools(self) -> bool:
        """Return whether any registered logical tool uses deferred loading."""
        return any(
            definition.loading.deferred for definition in self.registry.definitions()
        )

    def _resolve_tool(self, tool_name: str) -> Tool | None:
        if not self.registry.has(tool_name):
            return None
        executor = self.registry.get(tool_name).executor
        return executor if isinstance(executor, Tool) else None

    def load_tools(
        self,
        messages: ChatMessages,
        tool_names: List[str],
    ) -> List[str]:
        if not isinstance(messages, ChatMessages):
            raise TypeError("Deferred tool loading requires `ChatMessages`.")
        deferred = {
            entry.name
            for entry in self._build_tool_catalog_view(
                messages,
                require_thread=False,
            ).tool_entries()
            if entry.deferred
        }
        unknown = set(tool_names) - deferred
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Deferred tools are not available: {names}")
        return messages.load_tools(self.name, tool_names)

    def get_tool_annotations(self) -> Dict[str, Dict[str, Any]]:
        """Return local tool annotations keyed by tool name."""
        return self._build_tool_catalog_view(
            None,
            require_thread=False,
        ).annotations

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
        definition: RuntimeToolDefinition,
    ) -> None:
        if ToolBackground.is_reserved_definition(
            definition
        ) or not ToolBackground.is_background_definition(definition):
            return
        self._sync_background_task_tools()

    def _sync_background_task_tools(self) -> None:
        background = self.extensions.get("background_tasks")
        if isinstance(background, BackgroundTasksExtension):
            background.sync(self)

    # --- Tool Call Preparation ---
