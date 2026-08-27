# Execution Event Streaming

Use an execution event stream when an application needs the whole agent
lifecycle rather than only the final answer. Assistant content, model requests,
tool activity, failures, and run boundaries share one ordered stream.

## Usage

`stream_events()` is an async iterator. It runs the Agent and yields events as
they occur:

```python
async for event in agent.stream_events("Investigate the warehouse alert"):
    if event.type == "message.delta":
        await websocket.send_text(event.data["delta"])
    elif event.type == "tool.start":
        print(f"\nRunning {event.data['tool_name']}")
```

The complete Agent response is included in `message.end`. The following
`run.end` event closes the execution without repeating that potentially large
content:

```python
async for event in agent.stream_events("Summarize the incident"):
    if event.type == "message.end":
        final_output = event.data["content"]
    elif event.type == "run.end":
        outcome = event.data["outcome"]
```

When the model itself streams, the event iterator owns and consumes that model
stream. Do not separately consume a `ModelStreamResponse`: the content is
already delivered through `message.delta`, reasoning through
`reasoning.delta`, and the complete output through `message.end`.

## Event Shape

Every `ExecutionEvent` includes:

| Field | Description |
| --- | --- |
| `type` | Stable lifecycle event name |
| `timestamp` | UTC ISO 8601 emission time |
| `data` | Event-specific payload |
| `run_id` | Current run identity |
| `source_path` | Nested `type:name` identities from the observed root to the emitter |

Consume events in iterator order. Live event ordinals are not persisted or
exposed; a future durable event log would own a separate storage sequence.
Timestamps are useful for diagnostics but should not be used to reorder events.

`run.start` carries the less frequently changing execution context in its
`data`: `thread_id`, `namespace`, `root_run_id`, and `parent_run_id` for a
child operation when one exists. Foreground subagents may participate in the
same logical run as their caller; their nested `source_path` distinguishes
them. All following events remain correlatable through `run_id` and
`source_path` without repeating context on every token delta.

## Core Events

| Event | Meaning |
| --- | --- |
| `run.start` | Execution was accepted |
| `turn.start` | An agent turn started |
| `model.request` | A request is being sent to a model |
| `model.response` | The model response settled; includes compact token usage and timing when available |
| `message.start` | Assistant output presentation started |
| `message.delta` | Assistant content chunk |
| `reasoning.delta` | Reasoning content chunk, when exposed by the provider |
| `reasoning_summary.delta` | Provider reasoning-summary chunk |
| `message.end` | Complete assistant output |
| `tool.start` | A validated tool call is starting |
| `tool.update` | Intermediate tool progress |
| `tool.end` | Tool execution completed or failed |
| `turn.end` | The turn reached a terminal boundary; it has no output payload |
| `run.end` | The run completed and reports its outcome |
| `run.error` | Execution failed; the iterator raises after this event |

## Model Metrics

`model.response` is emitted once for every LM request. For a streamed model
response, it arrives after the model's content deltas have been drained, when
terminal usage and timing are available:

```python
async for event in agent.stream_events("Summarize the incident"):
    if event.type == "model.response":
        usage = event.data.get("usage", {})
        timing = event.data.get("timing", {})

        print(usage.get("input_tokens"))
        print(usage.get("output_tokens"))
        print(usage.get("cached_input_tokens"))
        print(timing.get("latency_ms"))
        print(timing.get("ttft_ms"))
```

The payload is intentionally compact. `usage` contains `input_tokens`,
`output_tokens`, and `cached_input_tokens` when reported; provider-native raw
usage, derived totals, and cache percentages are omitted. `timing` contains
`source`, `latency_ms`, and streaming `ttft_ms` when available.

Metrics stay on their individual `model.response` events instead of being
summed into `run.end`. This keeps multi-step tool loops and nested agents
auditable: each model call retains its own `run_id` and `source_path`, and a
consumer can aggregate only the calls relevant to its view.

Closing an event iterator early requests cancellation through the execution's
abort signal. Async execution is also cancelled locally so an abandoned client
does not retain ownership of the stream.

## Complete-Output Transformations

For Agents, a `Hook(event="transform_output", ...)` receives an `OutputContext`
with `output`, runtime `vars`, and the execution `scope`. Replacing `ctx.output`
changes the value presented to the caller without changing the canonical
assistant message stored in history. If the model streams tokens, the event
iterator accumulates assistant content and sets
`message.start.data["buffered"]` to `True`. It then emits the transformed value
once in `message.end`; raw assistant deltas are not exposed.

Reasoning, tool, and progress events are independent and continue to arrive
while assistant content is buffered.

See [Canonical Responses vs. Presented Output](hooks.md#canonical-responses-vs-presented-output)
for a complete example that replaces an artifact reference in the presented
event output while preserving the original reference in `ChatMessages`.

## Event Isolation

The event sink belongs to one execution. Concurrent Agent calls therefore keep
independent queues and do not receive each other's events. Foreground nested
executions publish into their root stream. Use `run_id` and `source_path` to
separate the root Agent, ToolLibrary, called tool, and subagent while retaining
one total delivery order.

A detached background task can outlive the root stream. Its later events are
therefore not delivered through the already-closed root iterator. Observe its
durable task activity and notifications until a task-scoped event stream is
available.

## Live OpenAI Validation

The repository includes a paid integration script for the Responses protocol:

```bash
uv run python scripts/validate_openai_event_streaming.py
```

It loads `OPENAI_API_KEY` from `.env`, uses `store=False`, and validates basic
token streaming, complete-output transformation, a local tool loop, and a
foreground AgentTool delegation. Use `--scenario basic`, `--scenario transform`,
`--scenario tool`, or `--scenario nested` to run one case.
