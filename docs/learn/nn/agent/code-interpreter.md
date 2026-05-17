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

When `ptc=True`, the sandbox is registered as a tool named
`python_interpreter`. The model can call it with Python code, and that code can
call explicitly allowed msgFlux tools through the `tools` namespace:

```python
ticket = await tools.lookup_ticket(ticket_id="MSGFLUX-42")
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

## Capabilities And Limits

`Sandbox.python("local")` is intentionally narrow:

- It persists Python globals between interpreter calls.
- It exposes allowed msgFlux tools through `tools.<name>(...)`.
- It exposes runtime values through `vars["name"]`.
- It does not provide direct network access.
- It does not provide direct host filesystem access.

Use file, shell, or workspace tools for host operations. The code interpreter is
for lightweight computation and controlled programmatic calls, not for direct
workspace mutation.

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

A real OpenAI-backed example demonstrates a `llm_query` PTC tool. The tool
receives a task and a context slice from interpreter code, injects that context
into a focused worker agent through `vars`, and returns the worker answer:

```bash
uv run python examples/code_interpreter_llm_query_real.py
```
