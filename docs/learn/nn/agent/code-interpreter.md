# Code Interpreter

`Agent` can expose a sandboxed code interpreter as a normal tool. The first
implementation is a local Python interpreter:

```python
import msgflux as mf
from msgflux import nn

agent = nn.Agent(
    name="support_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    tools=[lookup_ticket],
    code_interpreter=mf.Sandbox.python("local"),
    config={
        "code_interpreter": {
            "ptc": True,
            "ptc_tools": {"allow": ["lookup_ticket"]},
        }
    },
)
```

Monty is available as an optional dependency for future provider-backed
sandboxes:

```bash
pip install msgflux[monty]
```

Use it by selecting the Monty provider:

```python
agent = nn.Agent(
    name="support_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    tools=[lookup_ticket],
    code_interpreter=mf.Sandbox.python("monty"),
    config={
        "code_interpreter": {
            "ptc": True,
            "artifacts": True,
            "ptc_tools": {"allow": ["lookup_ticket"]},
        }
    },
)
```

Use dict-style namespace access inside every sandbox. This is the portable form
for both the local interpreter and Monty:

```python
ticket = tools["lookup_ticket"](ticket_id="MSGFLUX-42")
```

The same applies to artifacts:

```python
chunk = await artifacts["read"]("commands", offset=0, limit=4000)
```

When `ptc=True`, the sandbox is registered as a tool named
`python_interpreter`. The model can call it with Python code, and that code can
call explicitly allowed msgFlux tools through the `tools` namespace:

```python
ticket = await tools["lookup_ticket"].acall(ticket_id="MSGFLUX-42")
result = f"Ticket context: {ticket}"
```

## Programmatic Tool Calls

Programmatic tool calls are filtered separately from the agent's visible tools.
Use `ptc_tools` to decide which tools can be called from inside the interpreter:

```python
config={
    "code_interpreter": {
        "ptc": True,
        "ptc_tools": {"allow": "*", "block": ["send_user_message"]},
    }
}
```

Supported forms:

| Config | Meaning |
|--------|---------|
| `{"allow": ["search"]}` | Only `search` is available inside `tools` |
| `{"allow": "*", "block": ["write_file"]}` | All visible tools except `write_file` |
| `"*"` | Shorthand for `{"allow": "*"}` |

The agent also applies the runtime `tool_filter` before rendering the
interpreter description. If a tool is hidden from the current model request, it
is also hidden from the interpreter's PTC catalog.

## Runtime Vars

Runtime `vars` can be injected into the interpreter under the `vars` namespace:

```python
response = agent(
    "Inspect this ticket.",
    vars={"ticket_id": "MSGFLUX-42", "risk_threshold": 0.7},
)
```

Inside the interpreter:

```python
ticket_id = vars["ticket_id"]
threshold = vars["risk_threshold"]
```

By default, the agent sends a small `<system_note>` telling the model which vars
are available. The note includes names, Python type names, and size when the
value supports `len(...)`; it does not include raw values.

```xml
<system_note>
<runtime_context>
<code_interpreter name="python_interpreter" vars_namespace="vars">
<vars>
<var name="ticket_id" type="str" size="10" />
<var name="risk_threshold" type="float" />
</vars>
</code_interpreter>
</runtime_context>
</system_note>
```

Useful options:

```python
config={
    "code_interpreter": {
        "inject_vars": True,
        "notify_vars": True,
        "notify_vars_max": 20,
    }
}
```

Set `inject_vars=False` to prevent runtime vars from reaching the interpreter.
Set `notify_vars=False` to inject vars without adding the runtime note.

## Runtime Artifacts

Use `artifacts` when the agent needs access to large local files without putting
the file content directly in the model context or in interpreter `vars`.
Artifacts require the code interpreter to be exposed as a PTC tool and must be
enabled explicitly:

```python
config={
    "code_interpreter": {
        "ptc": True,
        "artifacts": True,
    }
}
```

If `artifacts` is passed to `forward`/`acall` without this config, msgFlux raises
an error before calling the model.

```python
response = await agent.acall(
    "Search the mounted command log and summarize the relevant entries.",
    artifacts={
        "commands": "data/commands.txt",
        "readme": "README.md",
    },
)
```

Artifacts are mounted by logical name. Inside the code interpreter they are
available through the `artifacts` namespace:

```python
print(artifacts["list"]())
print(artifacts["info"]("commands"))

chunk = artifacts["read"]("commands", offset=0, limit=4000)
result = await tools["llm_query"].acall(
    task="Find the deployment command and explain it.",
    task_context=chunk,
)
```

For debug output, prefer `print(...)` instead of retaining large slices in
interpreter globals. When possible, pass the bounded read directly to another
tool:

```python
print(artifacts["read"]("commands", offset=0, limit=300))
answer = await tools["llm_query"].acall(
    task="Find the deployment command and explain it.",
    context=artifacts["read"]("commands", offset=0, limit=4000),
)
result = answer
```

The async methods are also available:

```python
chunk = await artifacts["aread"]("commands", offset=4000, limit=4000)
matches = await artifacts["asearch"]("commands", "deploy", limit=5)
```

Available methods:

| Method | Purpose |
|--------|---------|
| `artifacts["list"]()` | Return mounted artifact names |
| `artifacts["info"](name)` | Return name, filename, byte size and unit |
| `artifacts["read"](name, offset=0, limit=4000)` | Read a byte-bounded text slice |
| `artifacts["aread"](name, offset=0, limit=4000)` | Async version of `read` |
| `artifacts["search"](name, query, limit=10)` | Search text and return offsets/previews |
| `artifacts["asearch"](name, query, limit=10)` | Async version of `search` |
| `artifacts["help"]()` | Return a short usage hint |

`limit` is required for reads and is measured in bytes. This is intentional:
the model must choose bounded slices instead of loading an entire large file into
the REPL memory. The default maximum read size is 12,000 bytes per call.

When artifacts are mounted, the agent sends a compact system note with names,
filenames and byte sizes:

```xml
<system_note>
<artifacts code_interpreter="python_interpreter" namespace="artifacts">
<artifact name="commands" filename="commands.txt" size="13211" unit="bytes"/>
</artifacts>
</system_note>
```

The note is emitted once per session/namespace for the same artifact set. If the
mounted artifacts change, the agent emits a new note.

Limitations in this version:

- Artifact contents are not copied into interpreter memory unless the model
  explicitly assigns a read result to a variable.
- Artifact contents are not persisted in the checkpointer.
- Resume requires passing the same `artifacts` mapping again.
- Artifact writes and URL loading are intentionally not part of v1.

## Capabilities And Limits

`Sandbox.python("local")` is intentionally narrow:

- It persists Python globals between interpreter calls.
- It exposes allowed msgFlux tools through `tools["name"](...)` and
  `await tools["name"].acall(...)`.
- It exposes runtime values through `vars["name"]`.
- It exposes mounted files through `artifacts["read"](...)` when `artifacts` are
  passed to the agent call.
- It does not provide direct network access.
- It does not provide direct host filesystem access.

Use file, shell, or workspace tools for host operations. The code interpreter is
for lightweight computation and controlled programmatic calls, not for direct
workspace mutation.

`Sandbox.python("monty")` uses the same msgFlux tool name
(`python_interpreter`) and keeps REPL state, but runs through Monty's Python
subset. Unsupported Python syntax or stdlib modules may fail. Use dict-style
namespaces (`tools["name"](...)`, `artifacts["read"](...)`) for code that works
across sandbox providers. The local interpreter exposes full `nn.Tool` objects,
so async calls can use `await tools["name"].acall(...)`.

`print(...)` output is captured and returned in the tool result. If the code
also sets a `result` variable, stdout is returned first and `result` is appended
after it.

## Runnable Example

The repository includes an offline example that uses a scripted model:

```bash
uv run python examples/code_interpreter_ptc_demo.py
```

With a different ticket id:

```bash
uv run python examples/code_interpreter_ptc_demo.py --ticket-id MSGFLUX-99
```

The artifacts example mounts a local file into the local Python sandbox through
the agent call:

```bash
uv run python examples/code_interpreter_artifacts_demo.py
```

With a specific file:

```bash
uv run python examples/code_interpreter_artifacts_demo.py --artifact-path README.md
```

A real OpenAI-backed version uses the same local sandbox, artifact mounting, and
an `llm_query` PTC tool that receives a bounded artifact slice:

```bash
uv run python examples/code_interpreter_artifacts_openai.py
```

With a specific file:

```bash
uv run python examples/code_interpreter_artifacts_openai.py --artifact-path README.md
```

A real OpenAI-backed example demonstrates a `llm_query` PTC tool. The tool
receives a task and a context slice from interpreter code, injects that context
into a focused worker agent through `vars`, and returns the worker answer:

```bash
uv run python examples/code_interpreter_llm_query_real.py
```
