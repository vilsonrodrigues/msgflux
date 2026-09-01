# ruff: noqa: A002

import asyncio
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
)

from msgflux.chat_messages import ChatMessages
from msgflux.models.response import ModelStreamResponse
from msgflux.nn.extensions.base import (
    AgentExtension,
    AgentExtensionHandle,
    _AgentToolsExtension,
    _extension_snapshot,
    _ExtensionHook,
    _get_extension_snapshot,
)
from msgflux.nn.extensions.skills import SkillsExtension
from msgflux.nn.hooks import Hook
from msgflux.nn.hooks.events import (
    OutputContext,
    RunEndContext,
)
from msgflux.nn.modules.container import ModuleDict
from msgflux.runtime.abort import AbortSignal
from msgflux.runtime.context import (
    ExecutionScope,
    execution_context,
    get_execution_context,
)
from msgflux.runtime.event_hub import ThreadWatcher, get_event_hub

if TYPE_CHECKING:
    pass
from msgflux.nn.modules.agent.context import (
    _CURRENT_AGENT_CONTEXT,
    _agent_context,
    _require_lifecycle_payload,
)


class AgentLifecycleMixin:
    """Agent extension, lifecycle-hook, event-stream, and watch behavior."""

    def _initialize_extensions(self) -> None:
        self.extensions = ModuleDict()
        self._extension_hook_handles: dict[str, list[Any]] = {}
        self._extension_library_handles: dict[str, Any] = {}
        self._extension_refcounts: dict[str, int] = {}
        self._pending_extensions: dict[str, AgentExtension] = {}
        self._extension_async_cleanup_waiters: dict[str, Any] = {}

    def _set_extensions(
        self,
        extensions: Union[List[AgentExtension], Mapping[str, AgentExtension]],
    ) -> None:
        entries = (
            extensions.items()
            if isinstance(extensions, Mapping)
            else ((extension.name, extension) for extension in extensions)
        )
        for name, extension in entries:
            self.register_extension(name, extension)

    def register_extension(
        self,
        name: str,
        extension: AgentExtension,
    ) -> AgentExtensionHandle:
        """Install a named extension and return its ownership handle."""
        self._validate_extension_registration(name, extension)
        extension_hooks = tuple(extension.hooks())
        extension_tools = tuple(extension.tools())
        hook_handles = []
        library_handle = None
        try:
            if extension_tools:
                contribution = _AgentToolsExtension(name, extension_tools)
                library_handle = self.tool_library.register_extension(
                    contribution.name,
                    contribution,
                )
            for hook in extension_hooks:
                hook_handles.append(self._register_extension_hook(name, hook))
            self.extensions[name] = extension
            self._extension_hook_handles[name] = hook_handles
            if library_handle is not None:
                self._extension_library_handles[name] = library_handle
            self._extension_refcounts[name] = 0
            extension._bind_agent(self)
            extension.on_register(self)
        except Exception:
            extension._unbind_agent()
            if name in self.extensions:
                del self.extensions[name]
            for handle in reversed(hook_handles):
                handle.remove()
            if library_handle is not None:
                library_handle.remove()
            raise
        return AgentExtensionHandle(self, name)

    def _validate_extension_registration(
        self,
        name: str,
        extension: AgentExtension,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("`name` must be a non-empty string")
        if not isinstance(extension, AgentExtension):
            raise TypeError(
                f"`extension` must be an AgentExtension, given `{type(extension)}`"
            )
        if name in self.extensions or name in self._pending_extensions:
            raise ValueError(f"The extension name `{name}` is already registered")

    def _register_extension_hook(self, name: str, hook: Hook):
        if not isinstance(hook, Hook):
            raise TypeError(
                f"Extension `{name}` returned a non-Hook contribution: `{type(hook)}`"
            )
        owned_hook = _ExtensionHook(self, name, hook)
        target = (
            getattr(self, owned_hook.target) if owned_hook.target is not None else self
        )
        return owned_hook.register(target)

    def has_extension(self, name: str) -> bool:
        """Return whether an extension is enabled for new runs."""
        return name in self.extensions

    def remove_extension(self, name: str) -> None:
        """Disable an extension for new runs and remove it when no run uses it."""
        if name not in self.extensions:
            return
        extension = self.extensions[name]
        del self.extensions[name]
        self._pending_extensions[name] = extension
        if self._extension_refcounts.get(name, 0) == 0:
            self._cleanup_extension(name)

    async def aremove_extension(self, name: str) -> None:
        """Async removal, using async cleanup when no active run retains it."""
        if name not in self.extensions:
            return
        extension = self.extensions[name]
        del self.extensions[name]
        self._pending_extensions[name] = extension
        ready = asyncio.Event()
        self._extension_async_cleanup_waiters[name] = ready
        if self._extension_refcounts.get(name, 0) == 0:
            ready.set()
        try:
            await ready.wait()
        except asyncio.CancelledError:
            self._extension_async_cleanup_waiters.pop(name, None)
            if self._extension_refcounts.get(name, 0) == 0:
                self._cleanup_extension(name)
            raise
        self._cleanup_extension_contributions(name)
        self._pending_extensions.pop(name, None)
        self._extension_refcounts.pop(name, None)
        try:
            await extension.aon_remove(self)
        finally:
            extension._unbind_agent()
            self._extension_async_cleanup_waiters.pop(name, None)

    def _cleanup_extension_contributions(self, name: str) -> None:
        for handle in self._extension_hook_handles.pop(name, ()):
            handle.remove()
        library_handle = self._extension_library_handles.pop(name, None)
        if library_handle is not None:
            library_handle.remove()

    def _cleanup_extension(self, name: str) -> None:
        extension = self._pending_extensions.pop(name, None)
        if extension is None:
            return
        self._cleanup_extension_contributions(name)
        self._extension_refcounts.pop(name, None)
        try:
            extension.on_remove(self)
        finally:
            extension._unbind_agent()

    def _extension_is_visible(self, name: str) -> bool:
        snapshot = _get_extension_snapshot(self)
        if snapshot is not None:
            return name in snapshot
        return name in self.extensions

    @contextmanager
    def _extension_run_snapshot(self):
        current = _get_extension_snapshot(self)
        if current is not None:
            yield current
            return
        names = frozenset(self.extensions)
        for name in names:
            self._extension_refcounts[name] = self._extension_refcounts.get(name, 0) + 1
        try:
            with _extension_snapshot(self, names):
                yield names
        finally:
            for name in names:
                count = self._extension_refcounts.get(name, 1) - 1
                self._extension_refcounts[name] = count
                if count == 0 and name in self._pending_extensions:
                    waiter = self._extension_async_cleanup_waiters.get(name)
                    if waiter is not None:
                        waiter.set()
                    else:
                        self._cleanup_extension(name)

    def _call_impl_with_hooks(self, *args, **kwargs):
        with self._extension_run_snapshot():
            with (
                execution_context(scope=kwargs.get("scope")),
                _agent_context(
                    self,
                    scope=kwargs.get("scope") or get_execution_context()["scope"],
                    vars=kwargs.get("vars") or {},
                ),
            ):
                return super()._call_impl_with_hooks(*args, **kwargs)

    async def _acall_impl_with_hooks(self, *args, **kwargs):
        with self._extension_run_snapshot():
            with (
                execution_context(scope=kwargs.get("scope")),
                _agent_context(
                    self,
                    scope=kwargs.get("scope") or get_execution_context()["scope"],
                    vars=kwargs.get("vars") or {},
                ),
            ):
                return await super()._acall_impl_with_hooks(*args, **kwargs)

    @contextmanager
    def _event_stream_execution_context(self, kwargs: Dict[str, Any]):
        with (
            self._extension_run_snapshot(),
            _agent_context(
                self,
                scope=kwargs.get("scope") or get_execution_context()["scope"],
                vars=kwargs.get("vars") or {},
            ),
        ):
            yield

    def _update_agent_context(self, inputs: Mapping[str, Any]) -> None:
        state = (_CURRENT_AGENT_CONTEXT.get() or {}).get(id(self))
        if state is not None:
            state["scope"] = inputs.get("scope") or state["scope"]
            state["vars"] = inputs.get("vars") or {}

    def _output_context(self, output: Any) -> OutputContext:
        state = (_CURRENT_AGENT_CONTEXT.get() or {}).get(id(self), {})
        return OutputContext(
            output=output,
            scope=state.get("scope") or get_execution_context()["scope"],
            vars=state.get("vars") or {},
        )

    def _transform_module_output(self, output: Any) -> Any:
        if isinstance(output, ModelStreamResponse):
            return output
        context = self._run_lifecycle_hooks(
            "transform_output",
            self._output_context(output),
        )
        if not isinstance(context, OutputContext):
            raise TypeError("Agent `transform_output` hooks must return OutputContext")
        return context.output

    def _run_end_context(
        self,
        *,
        outcome: Literal["completed", "failed", "interrupted", "paused"],
        messages: Any,
        vars: Mapping[str, Any],
        scope: Optional[ExecutionScope],
        output: Any = None,
        error: BaseException | None = None,
    ) -> RunEndContext:
        return RunEndContext(
            scope=scope or get_execution_context()["scope"],
            vars=vars,
            outcome=outcome,
            messages=messages,
            output=output,
            error=error,
        )

    def _run_run_end_hook(self, event: str, context: RunEndContext) -> RunEndContext:
        transformed = self._run_lifecycle_hooks(event, context)
        return _require_lifecycle_payload(event, transformed, RunEndContext)

    async def _arun_run_end_hook(
        self, event: str, context: RunEndContext
    ) -> RunEndContext:
        transformed = await self._arun_lifecycle_hooks(event, context)
        return _require_lifecycle_payload(event, transformed, RunEndContext)

    def _settle_terminal_run(
        self,
        inputs: Mapping[str, Any],
        outcome: Literal["failed", "interrupted", "paused"],
        error: BaseException,
    ) -> RunEndContext:
        run_end = self._run_run_end_hook(
            "before_run_end",
            self._run_end_context(
                outcome=outcome,
                messages=inputs.get("messages"),
                vars=inputs.get("vars", {}),
                scope=inputs.get("scope"),
                error=error,
            ),
        )
        settled_inputs = {**inputs, "messages": run_end.messages}
        if outcome == "interrupted":
            self._checkpoint_interrupted(settled_inputs, error)
        elif outcome == "paused":
            if (
                isinstance(run_end.messages, ChatMessages)
                and run_end.messages.get_active_turn() is not None
            ):
                run_end.messages.end_turn(event="pause")
            self._checkpoint_save(
                run_end.messages,
                inputs.get("vars", {}),
                status="paused",
            )
        else:
            self._checkpoint_save_on_error(settled_inputs)
        return self._run_run_end_hook("after_run_end", run_end)

    async def _asettle_terminal_run(
        self,
        inputs: Mapping[str, Any],
        outcome: Literal["failed", "interrupted", "paused"],
        error: BaseException,
    ) -> RunEndContext:
        run_end = await self._arun_run_end_hook(
            "before_run_end",
            self._run_end_context(
                outcome=outcome,
                messages=inputs.get("messages"),
                vars=inputs.get("vars", {}),
                scope=inputs.get("scope"),
                error=error,
            ),
        )
        settled_inputs = {**inputs, "messages": run_end.messages}
        if outcome == "interrupted":
            await self._acheckpoint_interrupted(settled_inputs, error)
        elif outcome == "paused":
            if (
                isinstance(run_end.messages, ChatMessages)
                and run_end.messages.get_active_turn() is not None
            ):
                run_end.messages.end_turn(event="pause")
            await self._acheckpoint_save(
                run_end.messages,
                inputs.get("vars", {}),
                status="paused",
            )
        else:
            await self._acheckpoint_save_on_error(settled_inputs)
        return await self._arun_run_end_hook("after_run_end", run_end)

    async def _atransform_module_output(self, output: Any) -> Any:
        if isinstance(output, ModelStreamResponse):
            return output
        context = await self._arun_lifecycle_hooks(
            "transform_output",
            self._output_context(output),
        )
        if not isinstance(context, OutputContext):
            raise TypeError("Agent `transform_output` hooks must return OutputContext")
        return context.output

    @property
    def agent_skill_manager(self):
        """Compatibility access to the manager owned by SkillsExtension."""
        extension = self.extensions["skills"]
        if not isinstance(extension, SkillsExtension):
            raise AttributeError("The Agent has no SkillsExtension")
        return extension.manager

    def _get_requested_scope(
        self, kwargs: Mapping[str, Any]
    ) -> Optional[ExecutionScope]:
        scope = kwargs.get("scope")
        if scope is None:
            return None
        if not isinstance(scope, ExecutionScope):
            raise TypeError(
                f"`scope` must be an ExecutionScope or None, given `{type(scope)}`"
            )
        return scope

    def watch(self, thread_id: str) -> ThreadWatcher:
        """Observe a thread snapshot and its future process-local events."""

        def load_messages() -> ChatMessages | None:
            checkpoint_store = self._get_effective_checkpoint_store()
            if checkpoint_store is None:
                return None
            state = checkpoint_store.load_latest_run(
                self.get_module_name(),
                thread_id,
            )
            if not isinstance(state, Mapping):
                return None
            messages_state = state.get("messages")
            if not isinstance(messages_state, Mapping):
                return None
            messages = ChatMessages()
            messages._hydrate_state(messages_state)
            return messages

        return get_event_hub().watch(
            thread_id,
            namespace=self.get_module_name(),
            load_messages=load_messages,
        )

    def _prepare_event_stream_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Give an event stream a cancellable, fully identified Agent scope."""
        prepared = dict(kwargs)
        scope = prepared.get("scope") or get_execution_context()["scope"]
        abort_signal = scope.abort_signal or AbortSignal()
        messages = prepared.get("messages")
        thread_id = self._resolve_thread_id(
            messages=messages,
            thread_id=scope.thread_id,
        )
        run_id = self._resolve_run_id(
            messages=messages,
            run_id=scope.run_id,
        )
        prepared["scope"] = scope.with_overrides(
            thread_id=thread_id,
            namespace=self.get_module_name(),
            run_id=run_id,
            abort_signal=abort_signal,
        )
        return prepared

    # --- Model Execution ---
