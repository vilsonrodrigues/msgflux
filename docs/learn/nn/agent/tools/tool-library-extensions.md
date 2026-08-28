# ToolLibrary Extensions

`ToolLibraryExtension` packages tools, lifecycle hooks, setup, and cleanup under
one removable owner. Use it for a capability that belongs to tool registration
or execution rather than to the surrounding Agent.

## Add Tools And Hooks

This extension contributes one tool and adjusts its validated arguments before
execution:

```python
from dataclasses import replace

from msgflux.nn import ToolLibrary, ToolLibraryExtension
from msgflux.nn.hooks import Hook


class MathTools(ToolLibraryExtension):
    def __init__(self):
        super().__init__("math")

    def tools(self):
        def double(value: int) -> int:
            """Double a value."""
            return value * 2

        return (double,)

    def hooks(self):
        def clamp(event):
            value = max(0, event.arguments["value"])
            return replace(event, arguments={"value": value})

        return (Hook(event="before_tool", handler=clamp),)


library = ToolLibrary("calculator", [], extensions=[MathTools()])
response = library([("call_1", "double", {"value": -3})])

assert response.tool_calls[0].result == 0
```

The contributed tool still passes through normal schema generation, runtime
injection, abort handling, telemetry, and `ToolExecutionPlan` dispatch. An
extension does not bypass the library core.

Library hooks run before hooks inherited from an owning Agent. `before_tool`
sees only model-visible arguments; injected handles, messages, and vars remain
protected runtime arguments. `after_tool` receives the result or error before
it becomes a `ToolCall` result.

Multiple extensions may contribute hooks for the same event. They run
sequentially in registration order. A `block` from `before_tool` or
`before_dispatch` stops the remaining handlers only for that call. The first
reason is returned to the model and emitted as `tool.blocked`; another tool call
in the same response is processed normally.

Use `before_dispatch` when a policy depends on the normalized dispatch mode:

```python
class ForegroundOnly(ToolLibraryExtension):
    def __init__(self):
        super().__init__("foreground_only")

    def hooks(self):
        def keep_attached(event):
            if event.dispatch_mode in {"background", "spawn"}:
                return replace(event, dispatch_mode="foreground")
            return event

        return (Hook(event="before_dispatch", handler=keep_attached),)
```

This boundary exposes public arguments and normalized config, not injected
runtime values. It may reduce background/spawn execution to foreground or
block the call; it cannot promote a foreground call into detached execution.

## Register And Remove

Registration returns an ownership handle:

```python
extension = MathTools()
handle = library.register_extension(extension.name, extension)

handle.remove()
assert "double" not in library.get_tool_names()
```

Use `await handle.aremove()` when cleanup performs async I/O. Library extension
removal is immediate: tools and hooks can disappear from an execution already
in progress. Configure extensions before serving concurrent requests and treat
runtime removal as an explicit administrative operation.

## Built-In Extensions

`ToolLibrary` uses the same mechanism for optional capabilities:

- `ToolSearchExtension` installs the deferred-tool search bucket when the first
  `defer_loading=True` tool is registered.
- `BackgroundTasksExtension` validates background capabilities and reconciles
  the task-control tool surface.
- `MCPServersExtension` owns MCP connection, discovery, remote tools, and
  connection cleanup. The existing `mcp_servers=` argument constructs this
  extension for compatibility and convenience.

Tool buckets, argument restoration, runtime injection, abort checks, and
execution telemetry remain part of the library core because every extension
must preserve those invariants.
