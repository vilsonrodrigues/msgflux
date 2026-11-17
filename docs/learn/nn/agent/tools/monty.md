# Monty Python

Monty executes a subset of Python in a separate interpreter process. Use it to
combine tool calls, filter large results and return a small answer to the model.
Install the optional interpreter with:

```bash
uv add 'msgflux[monty]'
```

Two compositions share the same execution engine:

| Tool | What the model sees | Tools accessible from Python |
|------|---------------------|------------------------------|
| `MontyTool` | Monty and the other public tools | Available tools in its library |
| `MontyToolBucket` | Monty and any tools outside its capture rule | Captured tools only |

Neither composition changes the tools' implementations. Nested calls use the
library's normal execution pipeline, including hooks, permissions, tool filters
and dispatch settings.

## Add Monty Alongside Tools

```python
import msgflux as mf
from msgflux.nn import Agent
from msgflux.tools.builtin import MontyTool


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


agent = Agent(
    name="analyst",
    model=mf.Model.chat_completion("opencode-go/deepseek-v4.1-flash"),
    tools=[MontyTool(), multiply],
)

answer = agent(
    "Use Python to calculate products for (2, 3), (4, 5) and (6, 7), "
    "then report their sum.",
    scope=mf.ExecutionScope(thread_id="products"),
)
```

The model can call `multiply` directly or combine several calls inside `monty`.
This composition is useful when individual tools should remain discoverable in
the model's normal tool list.

## Use Monty As A Bucket

Replace `MontyTool()` with `MontyToolBucket()` to capture the ordinary tools:

```python
from msgflux.nn import ToolLibrary
from msgflux.tools.builtin import MontyToolBucket


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


library = ToolLibrary(
    name="analyst",
    tools=[MontyToolBucket(), multiply],
)
assert library.get_tool_names() == ["monty"]

response = library([
    ("products-1", "monty", {
        "code": "import tools\nsum(tools.multiply(a=a, b=b) "
                "for a, b in [(2, 3), (4, 5), (6, 7)])",
    }),
])
assert response.tool_calls[0].result["output"] == 68
```

The model receives one Python execution schema. The captured `multiply` schema
remains in the canonical library registry and is available through discovery.
Use the same `tools=` composition in an `Agent` to let a model generate the code.

The default capture rule includes all non-deferred ordinary tools, agents and
background task controls, including custom tool kinds. An explicit
`capture={"tool_kind": "math", "defer_loading": False}` selects a custom kind,
using the same rules as other [tool buckets](tool-bucket.md). Other buckets are
not captured. Buckets with overlapping capture rules, such as a default
`AgentTool` alongside the default Monty bucket, cannot be combined. Deferred
tools keep their `ToolSearchTool` interface and are outside the default Monty
bucket. Omit `defer_loading` when Monty should own discovery for all tools. Model-loop feedback such as handoff or
return-direct cannot be captured. Optional background dispatch is preserved on
captured tools, including their `run_in_background` argument and task controls.

## Discover And Call Tools

The Python `tools` module is preloaded in every execution and exposes three
helpers. An explicit `import tools` is also supported:

```python
import tools

names = tools.list()
metadata = tools.describe("multiply")
result = tools.multiply(a=6, b=7)
print(result)
```

`describe` returns `name`, `description`, `input_schema` and `usage_guidance`
from the registered definition. Full schemas are fetched on demand rather than
concatenated into Monty's model-facing description. Print only the metadata
needed for the current task.

Use `tools.call("multiply", a=6, b=7)` for names that are not Python identifiers,
start with an underscore, or collide with `list`, `describe` or `call`. Arguments
must be supplied by keyword. Discovery and execution both check the current
filter and permissions; sandbox code cannot access raw handles or executors.
Other Monty tools are excluded to prevent recursive interpreter calls.

Calls return their values directly inside the interpreter. Do not write
`await tools.multiply(...)`. The tool's `acall` implementation still awaits
asynchronous host tools without blocking on their result in a synchronous
Python call. Calls inside a snippet execute sequentially. Use an existing
tool's optional background argument when independent work should run in a task.

## Keep Variables Across Calls

All saved variables are loaded at the beginning of each execution. Variables
that are not changed remain available; assigning the same name replaces its
previous value:

```python
from msgflux.nn import ToolLibrary
from msgflux.runtime.context import execution_context
from msgflux.tools.builtin import MontyTool

python = MontyTool(state_directory=".msgflux/monty")
library = ToolLibrary("example", [python])

with execution_context(thread_id="research", namespace="analyst"):
    library([("50", "monty", {"code": 'x = "value"'})])
    library([("51", "monty", {"code": "y = [1, 2, 3]"})])
    response = library([("52", "monty", {"code": "print(x)\ny"})])
    assert response.tool_calls[0].result == {
        "output": [1, 2, 3],
        "stdout": "value\n",
    }
```

This example retains both `x` and `y`. Replacing `x` leaves `y` intact. Setting
`x = None` releases its previous value; `reset=True` starts with no saved
variables. The current Monty parser does not implement `del`.

Only JSON-compatible data survives between executions: `None`, booleans,
strings, finite JSON numbers, lists and dictionaries with string keys, including
nested values. Functions, classes, modules, bytes, tuples, cyclic containers and
live tool references must be recreated when needed. Reference identity between
separately saved variables is not preserved. The response includes
`not_persisted` when named values cannot be retained. The reserved `tools`
namespace and names beginning with `__msgflux_` are never persisted.

Use `print(...)` to inspect tool results and saved variables. Assigning a result
does not display it. Print selected fields or short summaries instead of an
entire response.

A successful response contains the trailing expression as `output` and captured
printed text as `stdout`. Return small JSON-compatible values. Variables may
hold larger data without printing it into the model's context.

## Storage And Recovery

The default directory is `~/.msgflux/monty`; `state_directory` changes it.
Relative state directories resolve against the host process working directory.
Each session has one latest JSON record and an advisory lock file. Its
identity includes the library name, thread id, agent namespace, principal and
account. New tool instances using those same values and directory recover the
variables automatically. Use a stable thread id for recovery across processes;
without one, a tool instance gets an anonymous identity.

State is replaced atomically with a private temporary file, flush and rename.
On POSIX the directory is private (`0700`) and state files are private (`0600`).
Concurrent calls for one session fail with a busy error rather than overwrite
one another. Separate sessions have independent locks. Idle sessions retain no
interpreter worker or variable dictionary in the tool instance.

Only the latest variables and small execution metadata are stored. Neither old
variable values nor previous Python snippets accumulate. Native Monty session
dumps are deliberately not used: those also retain interpreter structures and
source from previous feeds.

The host can back up or explicitly replace state:

```python
with execution_context(thread_id="research", namespace="analyst"):
    handle = library.get_handle().for_tool(tool_name="monty")
    backup = python.dump_session(handle)
    python.release_session(handle)       # Delete this session's state file.
    python.load_session(backup, handle)   # Restore a trusted exported record.
```

These are host APIs, not tools exposed to the model. Restore only trusted state
from a compatible application. Session files are separate from Agent
checkpoints and are not atomically committed with conversation history.

### Interrupted Executions

Before running a snippet, the tool records that execution is in progress. A
crash, or an interruption after admitting a host tool, leaves the session marked
uncertain.
Future executions fail until the host reconciles possible external effects and
explicitly resets or replaces that state. Earlier external effects are never
silently replayed by restarting the snippet. The last completed invocation can
return its stored result when the same call id and code are retried.

Errors before any host tool is admitted discard the interpreter and retain the
previous committed variables, so the model can correct a syntax or calculation
error without resetting its saved data.

Only data is persisted, so a suspended Python stack cannot be resumed after a
pause. Control exceptions propagate to the host. Nested tools covered by an
Agent approval policy remain guarded, but this integration does not create
interactive approval requests for calls discovered inside Python: those calls
fail closed if they lack an approved binding. An approval on the outer `monty`
call can review the whole snippet before execution. It does not approve a
separately protected nested tool.

## Limits And Boundaries

```python
python = MontyTool(
    max_memory=64 * 1024 * 1024,
    max_duration=10.0,
    max_output_bytes=64 * 1024,
    max_state_bytes=8 * 1024 * 1024,
)
```

These defaults bound interpreter heap, execution time per feed, returned output
and serialized variables. The time budget excludes time waiting for host tools;
those tools keep their own limits. An uncertain execution requires explicit reset
after reconciling external effects. Interpreter-only failures retain the last
committed variables. Persistence includes filesystem I/O;
measure it for your deployment before using large variable sets.

Monty receives no direct host filesystem mounts or network clients. Filesystem,
network and process access happens through authorized tools. Those tools still
run in their configured execution environment: Monty does not move a Bash tool
or another Python tool into its interpreter sandbox. See the upstream
[Monty documentation](https://pydantic.dev/docs/monty/) for its supported Python
subset and interpreter limits.
