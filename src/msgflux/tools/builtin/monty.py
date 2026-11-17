"""Stateful sandboxed Python with an authorized ToolLibrary facade."""

from __future__ import annotations

import hashlib
import keyword
import math
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import msgspec

from msgflux.exceptions import (
    AbortRequestedError,
    TaskError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
from msgflux.nn import functional as F
from msgflux.runtime.context import get_execution_scope
from msgflux.runtime.permissions import require_permissions
from msgflux.tools.builtin._monty_state import SessionFiles, SessionRecord
from msgflux.tools.handles import ToolBucketHandle, ToolLibraryHandle
from msgflux.tools.types import Hidden, ToolBucket, ToolLibraryOperator

_CONTROL_ERRORS = (
    AbortRequestedError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
_DISCOVERY_NAMES = frozenset({"list", "describe", "call"})


def _monty():
    try:
        import pydantic_monty  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("Monty tools require `uv add 'msgflux[monty]'`.") from exc
    return pydantic_monty


def _default_state_directory() -> Path:
    return Path.home() / ".msgflux" / "monty"


# Capture builtins before model code can shadow names like `locals` or `str`.
# Export each global separately so one function/module does not discard data.
# Comparing the JSON round trip also rejects tuples and non-string dictionary
# keys instead of silently changing their types in the next execution.
_EXPORT_SETUP = """
import tools
import json as __msgflux_json
__msgflux_locals = locals

def __msgflux_export(__values, __dumps=__msgflux_json.dumps,
                     __loads=__msgflux_json.loads,
                     __errors=(TypeError, ValueError, OverflowError)):
    __saved = {}
    __omitted = []
    for __name, __value in __values.items():
        if __name == 'tools' or __name.startswith('__msgflux_'):
            continue
        try:
            __encoded = __dumps(__value, allow_nan=False)
            if __loads(__encoded) == __value:
                __saved[__name] = __encoded
            else:
                __omitted.append(__name)
        except __errors:
            __omitted.append(__name)
    return __saved, __omitted
"""
_EXPORT_CALL = "__msgflux_export(__msgflux_locals())"


class _ToolSurface:
    """Host-only projection: no raw handle or executor crosses into Monty."""

    def __init__(self, handle: ToolLibraryHandle):
        self.handle = handle
        self.external_started = False

    def names(self) -> list[str]:
        names = (
            self.handle.list_captured_tools()
            if isinstance(self.handle, ToolBucketHandle)
            else self.handle.list_tools()
        )
        return sorted(name for name in names if self._eligible(name))

    def _eligible(self, name: str) -> bool:
        definition = self.handle.get_tool_definition(name)
        implementation = getattr(definition.executor, "impl", None)
        if isinstance(implementation, MontyTool):
            return False
        try:
            require_permissions(
                definition.required_permissions, definition.required_resources
            )
        except PermissionError:
            return False
        return True

    def require(self, name: str) -> None:
        if not isinstance(name, str) or name not in self.names():
            raise ValueError(f"Tool `{name}` is not available in this Monty session.")

    def describe(self, name: str) -> dict[str, Any]:
        self.require(name)
        definition = self.handle.get_tool_definition(name)
        return {
            "name": name,
            "description": definition.description,
            "input_schema": deepcopy(dict(definition.input_schema)),
            "usage_guidance": definition.usage_guidance,
        }

    def module(self, monty):
        # Placeholders declare host functions; the explicit suspension driver
        # below dispatches them, keeping runtime control exceptions out of Python.
        attrs = dict.fromkeys(_DISCOVERY_NAMES, _external_placeholder)
        attrs.update(
            (name, _external_placeholder)
            for name in self.names()
            if name.isidentifier()
            and not keyword.iskeyword(name)
            and name not in _DISCOVERY_NAMES
            and not name.startswith("_")
        )
        return monty.ExternalModule(attrs)

    def resolve(self, function: str, args: tuple, kwargs: dict):
        name = function.removeprefix("tools.")
        if name == "list":
            if args or kwargs:
                raise TypeError("tools.list() takes no arguments")
            return False, self.names(), {}
        if name == "describe":
            return False, self.describe(*args, **kwargs), {}
        if name == "call":
            if not args or len(args) != 1:
                raise TypeError("Use tools.call(name, **arguments)")
            name = args[0]
        elif args:
            raise TypeError("Pass tool arguments by keyword; see tools.describe(name)")
        self.require(name)
        if "_runtime_arguments" in kwargs:
            raise ValueError("Runtime inputs cannot be supplied by sandbox code")
        return True, name, kwargs


def _external_placeholder(**_kwargs):
    raise RuntimeError("Monty host calls must use the controlled suspension driver")


class MontyTool(ToolLibraryOperator):
    """Execute Python snippets with persistent globals and access to available tools.

    Args:
        code: Python snippet. Import tools; use tools.list(), tools.describe(name),
            and tools.NAME(**arguments). The last expression is returned.
        reset: Discard this session before executing the snippet.
    """

    name = "monty"
    display_name = "Monty Python"
    description = """Execute Python with saved variables and access to available tools.

    Use print(...) to read tool results and inspect variables. Assigning a tool
    result to a variable does not display it. Print selected fields or concise
    summaries instead of entire large results. The tools module is preloaded;
    tools.list() lists names and tools.describe(name) provides the input schema
    and usage guidance. The last expression is also returned.

    Args:
        code: Python snippet to execute. Use print(...) to inspect results.
        reset: Discard all saved variables before executing this snippet.
    """
    annotations = {"code": str, "reset": bool, "return": dict}
    tool_config = {"runtime_inputs": ("handle",)}
    usage_guidance = (
        "Use Python to combine tools and process results without printing raw data. "
        "The tools module is preloaded. Discover names with `tools.list()`; consult "
        "`tools.describe(name)` for description, input_schema and usage_guidance. "
        "Call `tools.NAME(**arguments)` or `tools.call(name, **arguments)` with "
        "keyword arguments. Calls return values directly; do not await them. "
        "Only JSON-compatible variables persist; redefine functions/classes per call. "
        "Return a small final expression or "
        "use print(...) to read results and inspect variables. Print only relevant "
        "fields or concise summaries. No host filesystem or network is exposed "
        "directly."
    )

    def __init__(
        self,
        *,
        max_memory: int = 64 * 1024 * 1024,
        max_duration: float = 10.0,
        max_output_bytes: int = 64 * 1024,
        max_state_bytes: int = 8 * 1024 * 1024,
        state_directory: str | Path | None = None,
    ):
        _monty()
        for name, value in {
            "max_memory": max_memory,
            "max_output_bytes": max_output_bytes,
            "max_state_bytes": max_state_bytes,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer")
        if (
            isinstance(max_duration, bool)
            or not isinstance(max_duration, (int, float))
            or not math.isfinite(max_duration)
            or max_duration <= 0
        ):
            raise ValueError("`max_duration` must be positive")
        self.max_memory = max_memory
        self.max_duration = max_duration
        self.max_output_bytes = max_output_bytes
        self.max_state_bytes = max_state_bytes
        self.state_directory = (
            Path(state_directory)
            if state_directory is not None
            else _default_state_directory()
        )
        self._anonymous_id = uuid4().hex
        self._files = SessionFiles(
            self.state_directory, 2 * max_state_bytes + 2 * max_output_bytes + 65536
        )

    def _key(self, handle):
        scope = get_execution_scope()
        identity = (
            handle.get_ref(self.name).library_id,
            scope.thread_id or self._anonymous_id,
            scope.namespace,
            scope.principal,
            scope.account_name,
        )
        return hashlib.sha256(msgspec.msgpack.encode(identity)).hexdigest()

    def _decode(self, data: bytes) -> SessionRecord:
        state = msgspec.json.decode(data, type=SessionRecord)
        if state.version != 1:
            raise ValueError("Unsupported Monty session dump")
        if len(msgspec.json.encode(state.variables)) > self.max_state_bytes:
            raise ValueError("Monty variables exceed max_state_bytes")
        return state

    @contextmanager
    def _locked_state(self, handle, *, save: bool = False):
        with self._files.locked(self._key(handle)) as path:
            payload = self._files.read(path)
            state = self._decode(payload) if payload is not None else SessionRecord()
            try:
                yield state, path
            except BaseException as error:
                if save:
                    try:
                        self._files.write(path, msgspec.json.encode(state))
                    except Exception as save_error:
                        error.add_note(f"Could not save Monty state: {save_error}")
                raise
            else:
                if save:
                    self._files.write(path, msgspec.json.encode(state))

    def release_session(self, handle: ToolLibraryHandle) -> None:
        """Delete the current scoped session's retained state file."""
        with self._files.locked(self._key(handle)) as path:
            path.unlink(missing_ok=True)

    def dump_session(self, handle: ToolLibraryHandle) -> bytes:
        """Export the latest session for host-managed backup or checkpoint storage."""
        with self._locked_state(handle) as (state, _path):
            return msgspec.json.encode(state)

    def load_session(self, data: bytes, handle: ToolLibraryHandle) -> None:
        """Replace the current state file using a trusted, compatible dump."""
        if not isinstance(data, bytes) or len(data) > self._files.max_bytes:
            raise ValueError("Invalid or oversized Monty session dump")
        self._decode(data)
        with self._files.locked(self._key(handle)) as path:
            self._files.write(path, data)

    def __call__(self, code: str, *, reset: bool = False, handle: Hidden) -> dict:
        result = F.wait_for(self.acall, code, reset=reset, handle=handle)
        if isinstance(result, TaskError):
            raise result.exception
        return result

    async def acall(self, code: str, *, reset: bool = False, handle: Hidden) -> dict:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("`code` must be non-empty Python source")
        if not isinstance(reset, bool):
            raise TypeError("`reset` must be bool")
        with self._locked_state(handle, save=True) as (state, path):
            if reset:
                state.variables, state.invalid, state.completed_result = {}, False, None
            if state.invalid:
                raise RuntimeError(
                    "Monty execution did not finish safely; use reset=True "
                    "after reconciling any external effects"
                )
            call_id = handle.get_tool_call_id() or uuid4().hex
            code_hash = hashlib.sha256(code.encode()).hexdigest()
            if (
                state.completed_result is not None
                and state.call_id == call_id
                and state.run_id == get_execution_scope().run_id
                and state.code_hash == code_hash
            ):
                return msgspec.json.decode(state.completed_result)
            # Persist uncertainty before executing. This prevents crash/restart
            # from silently retrying a snippet whose external effects are unknown.
            state.invalid = True
            self._files.write(path, msgspec.json.encode(state))
            return await self._execute(code, handle, state, call_id)

    async def _execute(self, code, handle, state, call_id):
        monty = _monty()
        surface = _ToolSurface(handle)
        output = monty.CollectString(max_bytes=self.max_output_bytes)
        limits = {
            "max_memory": self.max_memory,
            "max_feed_duration_secs": self.max_duration,
        }
        try:
            async with monty.AsyncMonty(min_processes=1, max_processes=1) as pool:
                async with pool.checkout(
                    limits=limits, external_modules={"tools": surface.module(monty)}
                ) as session:
                    await session.feed_run(_EXPORT_SETUP, print_callback=output)
                    snapshot = await session.feed_start(
                        code, inputs=dict(state.variables), print_callback=output
                    )
                    snapshot = await self._drive(
                        snapshot, monty, surface, handle, call_id
                    )
                    result = {"output": snapshot.output, "stdout": output.output}
                    values, omitted = await session.feed_run(
                        _EXPORT_CALL, print_callback=output
                    )
                    variables = {
                        name: msgspec.json.decode(value)
                        for name, value in values.items()
                    }
                    if len(msgspec.json.encode(variables)) > self.max_state_bytes:
                        raise ValueError("Monty variables exceed max_state_bytes")
                    if omitted:
                        result["not_persisted"] = omitted
                    encoded_result = msgspec.json.encode(result)
                    if len(encoded_result) > self.max_output_bytes:
                        raise ValueError(
                            "Monty result exceeds max_output_bytes; "
                            "return a smaller value"
                        )
                    state.variables, state.completed_result = variables, encoded_result
                    state.call_id = call_id
                    state.code_hash = hashlib.sha256(code.encode()).hexdigest()
                    state.run_id = get_execution_scope().run_id
                    state.invalid = False
                    return result
        except Exception:
            if not surface.external_started:
                # No host tool was admitted. Discard this interpreter and keep
                # the previous variables, allowing a corrected snippet to run.
                state.invalid = False
            raise

    async def _drive(self, snapshot, monty, surface, handle, call_id):
        ordinal = 0
        while not isinstance(snapshot, monty.MontyComplete):
            scope = get_execution_scope()
            if scope.abort_signal is not None:
                scope.abort_signal.raise_if_aborted()
            if (
                isinstance(snapshot, monty.AsyncFunctionSnapshot)
                and not snapshot.is_os_function
                and snapshot.function_name.startswith("tools.")
            ):
                answer = await self._answer_call(
                    snapshot, surface, handle, call_id, ordinal
                )
                ordinal += 1
                snapshot = await snapshot.resume(answer)
            else:
                snapshot = await snapshot.resume_auto()
        return snapshot

    @staticmethod
    async def _answer_call(snapshot, surface, handle, call_id, ordinal):
        try:
            execute, value, arguments = surface.resolve(
                snapshot.function_name, snapshot.args, snapshot.kwargs
            )
            if execute:
                nested = handle.with_call_id(f"{call_id}:{ordinal}")
                surface.external_started = True
                value = await nested.acall(value, **arguments)
            return {"return_value": msgspec.to_builtins(value, builtin_types=(bytes,))}
        except _CONTROL_ERRORS:
            raise
        except Exception as exc:
            return {"exc_type": "RuntimeError", "message": str(exc)}


class MontyToolBucket(MontyTool, ToolBucket):
    """Expose Monty as the public interface to captured non-bucket tools."""

    tool_kind = "bucket"
    capture = {"defer_loading": False}

    def __init__(self, *, capture: Mapping[str, Any] | None = None, **kwargs):
        if capture is not None:
            self.capture = dict(capture)
        super().__init__(**kwargs)

    def validate_capture(self, definition: Any) -> None:
        # Dispatch stays on the underlying tool, including optional background
        # execution. Model-loop feedback cannot be represented inside Python.
        if not self.captures(definition):
            raise ValueError(f"Tool `{definition.name}` does not match this bucket")
        if definition.feedback.name != "model":
            raise ValueError(
                f"Tool `{definition.name}` has model-loop feedback "
                "and cannot be captured by Monty"
            )
