# ToolLibrary

`ToolLibrary` is the execution boundary for tools in msgFlux.

It sits between orchestration and implementation:

- `Agent` decides that a tool must be called
- `ToolLibrary` resolves and executes that call
- each `Tool` instance performs the actual local or remote work

This separation is important because schema concerns and execution concerns
meet here.

## What It Owns

`ToolLibrary` owns five responsibilities:

- registering local and remote tools
- routing tools into buckets when a bucket capture matches their configuration
- exposing tool schemas to other modules
- executing prepared tool calls
- collecting results into a uniform `ToolResponses` object

It does not decide when a tool should be called. That remains the job of the
provider response path or of a `ToolFlowControl`.

## Two Phases

`ToolLibrary` participates in two different phases of the runtime.

### 1. Schema-Time

Before the model runs, other modules ask `ToolLibrary` for metadata:

- `get_tool_json_schemas()`
- `get_tool_annotations()`

Those methods are used for different reasons.

`get_tool_json_schemas()` supports:

- native provider tool calling
- prompt rendering for custom tool loops
- dynamic provider schemas such as ReAct action variants

`get_tool_annotations()` supports:

- transport restoration after structured output decoding
- typed reconstruction of lowered values before local tool execution

### 2. Runtime

Once tool calls are produced, `ToolLibrary.forward(...)` or
`ToolLibrary.aforward(...)` executes them and returns a `ToolResponses`
container.

That runtime path applies tool configuration rules such as:

- `return_direct`
- `call_as_response`
- `spawn`
- `inject_vars`
- `inject_message`
- `inject_messages`
- `handoff`
- `disable_input`

This keeps the execution policy centralized instead of spreading it across
`Agent`, provider adapters, and tool implementations.

## The Execution Flow

The synchronous path looks like this:

```text
tool_callings
  -> ToolLibrary.forward(...)
  -> resolve tool by name
  -> apply tool config
  -> prepare call params
  -> run before_tool lifecycle hooks
  -> build ToolExecutionPlan
  -> execute tools with scatter_gather
  -> run after_tool lifecycle hooks
  -> collect ToolCall results
  -> return ToolResponses
```

The async path mirrors the same structure through `aforward(...)` and
`ascatter_gather(...)`. `ToolExecutionPlan` freezes the selected tool, visible
arguments, runtime arguments, dispatch mode, and return policy before either
path dispatches it.

## Extensions And Core Invariants

`ToolLibraryExtension` owns optional packages of tools, hooks, setup, and
cleanup. Library lifecycle hooks run before hooks inherited from an owning
Agent. The extension mechanism powers deferred tool search, background task
controls, and MCP server integration.

The core continues to own deterministic registration and bucket routing,
transport restoration, runtime injection, abort handling, telemetry, and plan
dispatch. Extensions can participate at stable lifecycle boundaries but cannot
bypass these invariants.

## Local And Remote Tools

The library can store both local and MCP-backed tools behind the same
interface.

```text
ToolLibrary
  -> LocalTool
  -> MCPTool
```

That means the caller does not need different orchestration logic for:

- a Python function
- an `nn.Module`-style tool
- a proxied MCP tool

The library normalizes all of them into a single execution surface.

## Deferred Tools

`ToolLibrary` can also keep tools registered without exposing them immediately
to the model.

The contract is intentionally small:

- `@tool_config(defer_loading=True)` keeps the tool out of the initial portable
  callable surface
- deferred tools are captured by the builtin `tool_search` bucket
- if at least one deferred tool exists, `ToolLibrary` registers `tool_search`
- `tool_search` can search both local and MCP-backed deferred tools
- keyword searches return matching tool names, and `description=True` includes
  tool metadata
- `select=["tool_name"]` records matching names in the current
  `ChatMessages.metadata`

This is useful when a session can register a large number of tools but should
keep the active tool context small.

An explicitly injected `ToolLibraryHandle` is the natural companion feature
here: a tool can add a new deferred tool at runtime, and `ToolLibrary` will
expose `tool_search` automatically if needed.

`ToolSearchExtension` installs `tool_search`, which is both a builtin operator
and a `ToolBucket` with
`capture={"defer_loading": True}`. It owns the searchable metadata, while the
thread's `ChatMessages` owns which names have been loaded. The shared library is
never mutated by a selection, so concurrent threads can expose different tool
subsets safely.

Background task control tools follow the same pattern through
`BackgroundTasksExtension`, backed by `ToolBackground`.
The `task_status`, `task_wait`, `task_output`, `task_interrupt`,
`task_activity`, and `task_message` tools are callable objects with
reserved background tool kinds: the common controls use `"background"`,
activity uses `"background_activity"`, and messaging uses
`"background_message"`. `ToolLibrary` asks `ToolBackground` to reconcile the
surface from currently registered tools. `ToolBackground` derives the common
controls from background execution and the optional controls from the union of
declared `background_capabilities`.

## Tool Buckets

Some tools are not independent public tools. They are better represented as
members of another tool.

`ToolBucket` is the base type for that pattern:

```python
class ToolBucket:
    tool_kind = "bucket"
    capture: Mapping[str, Any]

    def add(self, tool: ToolMetadata) -> None:
        ...

    def refresh(self) -> None:
        ...
```

`capture` matches entries in `tool_config`. For example,
`{"tool_kind": "agent", "defer_loading": False}` captures regular agents, while
`{"defer_loading": True}` captures every deferred tool. Every entry must match.
`capture["tool_kind"]` can name one kind or several kinds separated by `|`,
such as `"catalog|orders"`.

Two bucket captures cannot overlap. This makes routing deterministic without a
priority system. A kind bucket that coexists with deferred tools should include
`"defer_loading": False`, leaving `{"defer_loading": True}` to `tool_search`. The
base executable bucket rejects captured tools that configure
`background`, `allow_background`, `spawn`, `call_as_response`,
`return_direct`, or `handoff`; these model-loop policies belong to the
public bucket. `ToolSearchTool` overrides this validation because it catalogs
deferred tools rather than proxy-executing them behind its own call.

`ToolLibrary` routes matching tools to `ToolBucket.add(...)`. The base method
keeps metadata in `bucket.tools`, rejects duplicate names, and calls
`bucket.refresh()`. The base refresh hook does nothing; a subclass can rebuild
derived presentation data such as its description and usage guidance.

Buckets do not call `ToolMetadata.impl` to execute a child. An injected
`ToolBucketHandle` accepts `handle(tool_name, **arguments)` and
`handle.acall(...)`, validates membership, resolves the captured `LocalTool` or
`MCPTool`, and re-enters the library's normal argument preparation and telemetry
path.

The registration rule is:

- if a registered bucket matches the tool configuration, the bucket captures it
- otherwise, the tool is registered normally

When a bucket is registered, it also captures matching tools that are already
registered. Therefore, the order of `tools` in `ToolLibrary(...)` does not
change capture behavior.

For agents, `nn.Agent` is normalized to `tool_config["tool_kind"]="agent"`.
`AgentTool` is a bucket with
`capture={"tool_kind": "agent", "defer_loading": False}`, so adding an agent to
a library that already has `AgentTool` updates the single public
`agent(name, message)` tool instead of exposing the agent as a separate tool.

```python
library = ToolLibrary(
    name="team",
    tools=[
        AgentTool(),
        reviewer_agent,
        planner_agent,
    ],
)
```

The model only sees `agent(...)`. The bucket description and usage guidance are
refreshed on the wrapping `LocalTool`, so provider schemas and prompt guidance
reflect the captured agents.

The public parameters stay as `agent(name, message)`. `AgentTool` receives a
scoped bucket handle carrying the parent call context. When it selects a child,
`ToolLibrary` reapplies that child's configuration: for example,
`inject_messages=True` gives the child agent a copy of the parent's history,
while an agent without that option receives no inherited history. The scoped
handle itself keeps the original context references; copying happens only while
preparing a selected agent call. Agent execution namespaces are normalized from
the agent's canonical module name, so a public tool alias does not change its
checkpoint identity.

Deferred agents are captured by `tool_search` and represented as logical tools
in `ToolCatalog`. Once loaded for a thread, the provider can call the agent by
its own tool name without promoting it into the shared `AgentTool` bucket.

## Typed Restoration

One of the most important newer responsibilities of this layer is restoring
transport-lowered arguments back to the logical tool parameter types.

For local tools, that work happens in `LocalTool._restore_transport_params(...)`
using the original annotations.

That means a model can return a provider-friendly transport shape such as:

```json
{
  "fields": {
    "entries": [
      {"key": "city", "value": "Austin"},
      {"key": "country", "value": "USA"}
    ]
  }
}
```

and the local tool still receives:

```python
{"fields": {"city": "Austin", "country": "USA"}}
```

This is the point where transport compatibility is converted back into runtime
types.

## Agent Relationship

The relationship between `Agent` and `ToolLibrary` is intentionally narrow.

```text
Agent
  -> ToolLibrary.get_tool_json_schemas()
  -> ToolLibrary.get_tool_annotations()
  -> ToolLibrary(...) / ToolLibrary.acall(...)
```

That gives `Agent` just enough surface area to:

- expose tools to providers
- build custom flow-control schemas
- execute tool calls

without turning `Agent` itself into a tool registry or argument decoder.

## ASCII Diagram

This is the role of `ToolLibrary` in the larger flow:

```text
                    +----------------------+
                    |        Agent         |
                    +----------+-----------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
   +------------------------+        +------------------------+
   | get_tool_json_schemas  |        | get_tool_annotations   |
   +-----------+------------+        +------------+-----------+
               |                                  |
               +----------------+-----------------+
                                |
                                v
                     +----------------------+
                     |     ToolLibrary      |
                     | register / resolve   |
                     | config / execute     |
                     +----------+-----------+
                                |
          +---------------------+----------------------+
          |                                            |
          v                                            v
 +----------------------+                    +----------------------+
 |      LocalTool       |                    |       MCPTool        |
 | restore params       |                    | proxy remote tool    |
 | call Python impl     |                    | call MCP client      |
 +----------+-----------+                    +----------+-----------+
            |                                           |
            +-------------------+-----------------------+
                                |
                                v
                     +----------------------+
                     |    ToolResponses     |
                     |  ToolCall entries    |
                     +----------------------+
```

## Why This Shape Matters

If tool execution lived directly in `Agent`, every new transport rule,
injection rule, or remote-tool integration would make the orchestrator more
fragile.

If schema restoration lived only in provider code, local tool execution would
become provider-specific.

`ToolLibrary` keeps that boundary clean:

- `Agent` stays focused on orchestration
- providers stay focused on transport and decoding
- tool implementations stay focused on business logic

## Related Pages

- [Agent](agent.md)
- [ToolFlowControl](tool-flow-control.md)
- [Dict Lowering and Restoration](dict-lowering-and-restoration.md)
