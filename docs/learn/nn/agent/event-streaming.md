# Execution Event Streaming

Use an execution event stream when an application needs the whole agent
lifecycle rather than only the final answer. Assistant content, model requests,
tool activity, failures, and run boundaries share one ordered stream.

## Synchronous Usage

`stream_events()` runs the Agent and yields events as they occur:

```python
for event in agent.stream_events("Investigate the warehouse alert"):
    if event.type == "message.delta":
        print(event.data["delta"], end="", flush=True)
    elif event.type == "tool.start":
        print(f"\nRunning {event.data['tool_name']}")
```

The Agent response is included in the terminal `run.end` event:

```python
for event in agent.stream_events("Summarize the incident"):
    if event.type == "run.end":
        final_output = event.data["output"]
```

## Asynchronous Usage

`astream_events()` is an async iterator with the same event contract:

```python
async for event in agent.astream_events("Summarize the incident"):
    if event.type == "message.delta":
        await websocket.send_text(event.data["delta"])
```

When the model itself streams, the event iterator owns and consumes that model
stream. Do not separately consume a `ModelStreamResponse`: the content is
already delivered through `message.delta`, reasoning through
`reasoning.delta`, and the complete output through `message.end` and `run.end`.

## Event Shape

Every `ExecutionEvent` includes:

| Field | Description |
| --- | --- |
| `type` | Stable lifecycle event name |
| `sequence` | Zero-based order within this event stream |
| `timestamp` | UTC ISO 8601 emission time |
| `data` | Event-specific payload |
| `thread_id` | Durable conversation identity, when available |
| `namespace` | Agent/checkpoint namespace |
| `run_id` | Current run identity |
| `parent_run_id` | Parent run for nested execution |
| `root_run_id` | Root execution identity |

Use `sequence` for presentation order. Timestamps are useful for diagnostics but
should not be used to reorder concurrent events.

## Core Events

| Event | Meaning |
| --- | --- |
| `run.start` | Execution was accepted |
| `turn.start` | An agent turn started |
| `model.request` | A request is being sent to a model |
| `model.response` | The model returned a response or opened a stream |
| `message.start` | Assistant output presentation started |
| `message.delta` | Assistant content chunk |
| `reasoning.delta` | Reasoning content chunk, when exposed by the provider |
| `reasoning_summary.delta` | Provider reasoning-summary chunk |
| `message.end` | Complete assistant output |
| `tool.start` | A validated tool call is starting |
| `tool.update` | Intermediate tool progress |
| `tool.end` | Tool execution completed or failed |
| `turn.end` | The turn reached a terminal boundary |
| `run.end` | The run completed normally |
| `run.error` | Execution failed; the iterator raises after this event |

Closing an event iterator early requests cancellation through the execution's
abort signal. Async execution is also cancelled locally so an abandoned client
does not retain ownership of the stream.

## Event Isolation

The event sink belongs to one execution. Concurrent Agent calls therefore keep
independent sequence numbers and do not receive each other's events. Nested
executions retain their `parent_run_id` and `root_run_id`, allowing a UI to
group subagent activity without relying on global callbacks.
