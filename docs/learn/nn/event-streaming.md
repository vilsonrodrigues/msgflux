# Event Streaming

Event streaming allows you to capture real-time events during module execution. This is useful for building interactive UIs, debugging, monitoring agent behavior, and implementing custom logging.

## Overview

The event system provides **dual emission**:

1. **OpenTelemetry spans** - Events are persisted to OTel backends (Jaeger, Zipkin, etc.)
2. **Streaming queue** - Events are streamed in real-time to consumers

When no streaming consumer is active, there is **zero overhead** - events only go to OTel spans.

## Quick Start

### Async Streaming with `astream()`

The simplest way to consume events is via `Module.astream()`:

```python
import asyncio
import msgflux.nn as nn
from msgflux import Model

model = Model.chat_completion("openai/gpt-4o-mini")
agent = nn.Agent(
    name="assistant",
    model=model,
    instructions="You are a helpful assistant.",
)

async def main():
    async for event in agent.astream("Hello, how are you?"):
        print(f"[{event.name}] {event.attributes}")

asyncio.run(main())
```

### Sync Streaming with `stream()`

For synchronous code, use `Module.stream()`:

```python
# Collect all events
events = agent.stream("Hello!")
for event in events:
    print(f"[{event.name}] {event.attributes}")

# Or use a callback
def on_event(event):
    print(f"Event: {event.name}")

result = agent.stream("Hello!", callback=on_event)
print(f"Result: {result}")
```

## Event Types

Events follow [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). Available event types:

### Agent Events

| Event Type | Description |
|------------|-------------|
| `gen_ai.agent.start` | Agent execution started |
| `gen_ai.agent.step` | Agent completed a step (e.g., tool call iteration) |
| `gen_ai.agent.complete` | Agent execution completed |
| `gen_ai.agent.error` | Agent encountered an error |

### Model Events

| Event Type | Description |
|------------|-------------|
| `gen_ai.model.request` | Model API request sent |
| `gen_ai.model.response` | Model response received |
| `gen_ai.model.response.chunk` | Streaming response chunk |
| `gen_ai.model.reasoning` | Model reasoning/thinking content |

### Tool Events

| Event Type | Description |
|------------|-------------|
| `gen_ai.tool.call` | Tool invocation started |
| `gen_ai.tool.result` | Tool returned a result |
| `gen_ai.tool.error` | Tool raised an error |

### Flow Control Events

| Event Type | Description |
|------------|-------------|
| `gen_ai.flow.step` | Flow control step (ReAct, CoT) |
| `gen_ai.flow.reasoning` | Flow reasoning content |
| `gen_ai.flow.complete` | Flow completed |

### Module Events

| Event Type | Description |
|------------|-------------|
| `gen_ai.module.start` | Any module started |
| `gen_ai.module.complete` | Any module completed |
| `gen_ai.module.error` | Any module errored |

## StreamEvent Structure

Each event is a `StreamEvent` dataclass:

```python
@dataclass(frozen=True)
class StreamEvent:
    name: str              # Event type (e.g., "gen_ai.tool.call")
    attributes: dict       # Event-specific data
    timestamp_ns: int      # Nanoseconds since epoch
    span_name: str         # Parent span name
    span_id: str           # Hex span ID
    trace_id: str          # Hex trace ID
```

## Advanced Usage

### Using EventStream Directly

For more control, use `EventStream` as a context manager:

```python
from msgflux.nn import EventStream
import asyncio

async def main():
    async with EventStream() as stream:
        # Run agent in background task
        async def run_agent():
            result = await agent.acall("What is 2+2?")
            stream.close()  # Signal end of events
            return result

        task = asyncio.create_task(run_agent())

        # Process events as they arrive
        async for event in stream:
            if event.name == "gen_ai.tool.call":
                print(f"Calling tool: {event.attributes['tool_name']}")
            elif event.name == "gen_ai.agent.complete":
                print(f"Agent finished: {event.attributes['response']}")

        result = await task

asyncio.run(main())
```

### Filtering Events

Filter events by type for specific use cases:

```python
from msgflux.nn import EventType

async for event in agent.astream("Calculate 15 + 27"):
    match event.name:
        case EventType.TOOL_CALL:
            print(f"Tool: {event.attributes['tool_name']}")
        case EventType.TOOL_RESULT:
            print(f"Result: {event.attributes['result']}")
        case EventType.AGENT_COMPLETE:
            print(f"Final: {event.attributes['response']}")
```

### Event Callbacks

Register callbacks for specific processing:

```python
async with EventStream() as stream:
    stream.on_event(lambda e: log_event(e))
    stream.on_event(lambda e: update_ui(e))

    task = asyncio.create_task(agent.acall("Hello"))
    async for event in stream:
        pass  # Callbacks are invoked automatically
    await task
```

## Example: Building a Chat UI

```python
import asyncio
from msgflux.nn import EventType

async def chat_with_streaming(agent, message):
    """Display events in real-time for a chat interface."""

    print(f"User: {message}")
    print("Assistant: ", end="", flush=True)

    final_response = None

    async for event in agent.astream(message):
        match event.name:
            case EventType.MODEL_RESPONSE_CHUNK:
                # Stream text chunks as they arrive
                chunk = event.attributes.get("chunk", "")
                print(chunk, end="", flush=True)

            case EventType.TOOL_CALL:
                tool = event.attributes["tool_name"]
                print(f"\n[Calling {tool}...]", end="", flush=True)

            case EventType.TOOL_RESULT:
                result = event.attributes["result"]
                print(f" Done.", flush=True)

            case EventType.AGENT_COMPLETE:
                final_response = event.attributes["response"]

    print()  # New line after streaming
    return final_response
```

## Emitting Custom Events

You can emit custom events from within your tools or modules:

```python
from msgflux.nn.events import add_event

def my_custom_tool(query: str) -> str:
    """A tool that emits custom events."""
    add_event("custom.search.start", {"query": query})

    # Do work...
    results = perform_search(query)

    add_event("custom.search.complete", {
        "query": query,
        "result_count": len(results)
    })

    return results
```

## Integration with OpenTelemetry

Events are automatically recorded in OTel spans. Enable telemetry export:

```bash
export MSGTRACE_TELEMETRY_ENABLED=true
export MSGTRACE_EXPORTER=otlp  # or "console" for debugging
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

View events in Jaeger, Zipkin, or any OTel-compatible backend.
