# Event Streaming

Use `astream_events()` when you need live execution signals for a CLI, web UI,
debugger, or orchestration layer.

Event streaming is separate from response streaming. Response streaming yields
model text chunks. Event streaming yields runtime events such as agent start,
model requests, tool calls, task updates, inbox notifications, and checkpoint
saves.

## Synchronous Use

```python
import msgflux as mf
import msgflux.nn as nn


def lookup_order(order_id: str) -> str:
    """Lookup an order."""
    return "shipped"


agent = nn.Agent(
    name="support_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    tools=[lookup_order],
)

events = agent.stream_events("Check order 123.")

for event in events:
    print(event.name, event.attributes)
```

`stream_events()` is synchronous. It captures events during execution and returns
them after the module finishes. If you pass a callback, the callback is invoked
after execution with the captured events, not as a live stream:

```python
seen = []
result = agent.stream_events(
    "Check order 123.",
    callback=seen.append,
)
```

## Live Async Use

Use `astream_events()` to consume events while execution is still running:

```python
async for event in agent.astream_events("Check order 123."):
    print(event.name, event.attributes)
```

## Common Event Names

Use `mf.EventType` or `nn.EventType` instead of hard-coded strings when possible.

```python
if event.name == mf.EventType.TOOL_STARTED:
    print("tool started", event.attributes["tool_name"])
```

Important event groups:

- Agent: `AGENT_START`, `AGENT_RESUMED`, `AGENT_COMPLETE`, `AGENT_ERROR`.
- Turn: `TURN_START`, `TURN_COMPLETE`, `TURN_ERROR`.
- Model: `MODEL_REQUEST`, `MODEL_RESPONSE`.
- Tool: `TOOL_CALL`, `TOOL_STARTED`, `TOOL_RESULT`, `TOOL_ERROR`,
  `TOOL_UPDATE`.
- Subagent: `SUBAGENT_START`, `SUBAGENT_COMPLETE`, `SUBAGENT_ERROR`.
- Task: `TASK_CREATED`, `TASK_RUNNING`, `TASK_PROGRESS`, `TASK_COMPLETED`,
  `TASK_FAILED`, `TASK_PAUSED`, `TASK_STOPPED`, `TASK_STOP_REQUESTED`.
- Runtime: `INBOX_NOTIFICATION`, `CONTROL_RECEIVED`, `CHECKPOINT_LOADED`,
  `CHECKPOINT_SAVED`, `USER_MESSAGE_RECEIVED`, `USER_MESSAGE_INJECTED`.
- Compaction: `COMPACTION_PRE`, `COMPACTION_POST`.

Subagent events are emitted only when a tool wraps an `Agent`. In that case the
stream includes both the generic tool events and the subagent events:

```text
TOOL_STARTED
SUBAGENT_START
SUBAGENT_COMPLETE
TOOL_RESULT
```

Use `tool.*` to answer "which tool did the model call?" and `subagent.*` to
render or replay nested agent execution.

Compaction events are not emitted by the agent automatically. They are exposed
for hooks or custom runtime code that compact chat history, snapshots, or other
context state:

```python
mf.emit_compaction_pre(
    target="messages",
    strategy="summarize",
    message_count=len(messages),
)

# run compaction

mf.emit_compaction_post(
    target="messages",
    strategy="summarize",
    message_count_before=24,
    message_count_after=6,
)
```

## Tool Metadata

Tool events include the call identity and caller context:

```python
{
    "tool_call_id": "call_abc",
    "tool_name": "lookup_order",
    "caller_name": "support_agent",
    "caller_namespace": "support_agent",
    "caller_session_id": "customer_42",
    "caller_run_id": "ticket_9001",
    "arguments": {"order_id": "123"},
}
```

The metadata is injected into the tool module before the tool implementation
runs, but it is not passed to the Python function itself. A tool can keep a
normal signature:

```python
def lookup_order(order_id: str) -> str:
    """Lookup an order."""
    return "shipped"
```

## Execution Control

Event streaming works with `ExecutionScope`, checkpointers, and `AgentInbox`.

```python
checkpointer = mf.Store.checkpoint("sqlite", path=".msgflux/checkpoints.sqlite3")
inbox_store = mf.Store.agent_inbox("sqlite", path=".msgflux/inbox.sqlite3")

agent = nn.Agent(
    name="support_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    checkpointer=checkpointer,
    agent_inbox=mf.AgentInbox(store=inbox_store),
)

scope = mf.ExecutionScope(
    session_id="customer_42",
    run_id="ticket_9001",
)

async for event in agent.astream_events("Investigate the ticket.", scope=scope):
    if event.name == mf.EventType.CHECKPOINT_SAVED:
        print("checkpoint", event.attributes)
```

While the agent is running, another writer can add an incoming user message to
the same inbox. The agent drains it before the next model call.

```python
external_inbox = mf.AgentInbox(
    store=inbox_store,
    namespace="support_agent",
    session_id="customer_42",
    run_id="ticket_9001",
)

external_inbox.user_message("The customer added the latest invoice number.")
```

The stream emits `INBOX_NOTIFICATION` when the message is published. The model
receives the content as an `<incoming_user_message>` block at the next safe
boundary.

Control messages are also visible:

```python
external_inbox.pause(reason="Need human approval.")
```

When the agent drains the inbox, it emits `CONTROL_RECEIVED` and raises the
corresponding control exception.

## Telemetry

Events are emitted to the active msgtrace/OpenTelemetry span and to the active
event stream. If there is no active stream, event emission still works for the
span and has no queue consumer.
