# Event Streaming

Event streaming allows you to capture real-time events during module execution. This is useful for building interactive UIs, debugging, monitoring agent behavior, and implementing custom logging.

## Overview

The event system provides **dual emission**:

1. **OpenTelemetry spans** - Events are persisted to OTel backends (Jaeger, Zipkin, etc.)
2. **Streaming queue** - Events are streamed in real-time to consumers

When no streaming consumer is active, there is **zero overhead** - events only go to OTel spans.

### Key Features

- **Agent-centric events**: All events include `agent_name` for traceability in multi-agent systems
- **Separate content/reasoning streams**: Streaming responses have dedicated queues for content and reasoning
- **Hierarchical tracking**: Sub-agent events are automatically linked via OTel parent-child spans
- **Zero overhead when disabled**: Events only incur cost when actively consumed

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

## Streaming Model Responses

To receive streaming chunks from the model, enable streaming via the `config` parameter:

```python
agent = nn.Agent(
    name="assistant",
    model=model,
    instructions="You are a helpful assistant.",
    config={"stream": True},  # Enable streaming
)

async for event in agent.astream("Tell me a story"):
    if event.name == EventType.MODEL_RESPONSE_CHUNK:
        print(event.attributes["chunk"], end="", flush=True)
```

### Separate Content and Reasoning Streams

When using models that support reasoning (like OpenAI o1 or Claude with extended thinking), the streaming response separates content from reasoning:

```python
from msgflux.nn import EventType

async for event in agent.astream("Solve this math problem"):
    match event.name:
        case EventType.MODEL_RESPONSE_CHUNK:
            # Main content chunks
            print(event.attributes["chunk"], end="")
        case EventType.MODEL_REASONING_CHUNK:
            # Reasoning/thinking chunks (when available)
            print(f"[Thinking: {event.attributes['chunk']}]")
```

The response object also provides separate async generators:

```python
# Direct access to separate streams (when using ModelStreamResponse)
async for chunk in response.consume_content():
    print(chunk, end="")

async for reasoning in response.consume_reasoning():
    print(f"[Reasoning: {reasoning}]")
```

## Event Types

Events follow [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). Available event types:

### Agent Events

| Event Type | Description | Key Attributes |
|------------|-------------|----------------|
| `gen_ai.agent.start` | Agent execution started | `agent_name` |
| `gen_ai.agent.step` | Agent completed a step (e.g., tool call iteration) | `agent_name`, `step` |
| `gen_ai.agent.complete` | Agent execution completed | `agent_name`, `response` |
| `gen_ai.agent.error` | Agent encountered an error | `agent_name`, `error` |

### Model Events

| Event Type | Description | Key Attributes |
|------------|-------------|----------------|
| `gen_ai.model.request` | Model API request sent | `agent_name` |
| `gen_ai.model.response` | Model response received | `agent_name`, `response_type` |
| `gen_ai.model.response.chunk` | Streaming response chunk | `agent_name`, `chunk`, `index` |
| `gen_ai.model.reasoning` | Model reasoning/thinking content | `agent_name`, `reasoning` |
| `gen_ai.model.reasoning.chunk` | Streaming reasoning chunk | `agent_name`, `chunk`, `index` |

### Tool Events

| Event Type | Description | Key Attributes |
|------------|-------------|----------------|
| `gen_ai.tool.call` | Tool invocation started | `agent_name`, `tool_name`, `tool_id`, `arguments`, `step` |
| `gen_ai.tool.result` | Tool returned a result | `agent_name`, `tool_name`, `tool_id`, `result`, `step` |
| `gen_ai.tool.error` | Tool raised an error | `agent_name`, `tool_name`, `tool_id`, `error`, `step` |

### Flow Control Events

| Event Type | Description | Key Attributes |
|------------|-------------|----------------|
| `gen_ai.flow.step` | Flow control step (ReAct, CoT) | `step` |
| `gen_ai.flow.reasoning` | Flow reasoning content | `reasoning` |
| `gen_ai.flow.complete` | Flow completed | `result` |

### Module Events

| Event Type | Description | Key Attributes |
|------------|-------------|----------------|
| `gen_ai.module.start` | Any module started | `module_name`, `module_type` |
| `gen_ai.module.complete` | Any module completed | `module_name`, `module_type` |
| `gen_ai.module.error` | Any module errored | `module_name`, `module_type`, `error` |

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

## Event Attributes

### Agent Name Tracking

All events include the `agent_name` attribute, making it easy to track which agent emitted each event:

```python
async for event in agent.astream("Hello"):
    agent = event.attributes.get("agent_name")
    print(f"[{agent}] {event.name}")
```

### Tool Call Tracking

Tool events include a unique `tool_id` for correlating calls with results:

```python
pending_calls = {}

async for event in agent.astream("What's the weather?"):
    match event.name:
        case EventType.TOOL_CALL:
            tool_id = event.attributes["tool_id"]
            tool_name = event.attributes["tool_name"]
            pending_calls[tool_id] = tool_name
            print(f"Calling {tool_name} (id={tool_id})")

        case EventType.TOOL_RESULT:
            tool_id = event.attributes["tool_id"]
            tool_name = pending_calls.pop(tool_id)
            print(f"Result from {tool_name}: {event.attributes['result']}")
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

## Multi-Agent Systems

When using sub-agents (agents as tools), events maintain traceability through `agent_name`:

```python
# Create sub-agent
researcher = nn.Agent(
    name="researcher",
    model=model,
    instructions="You research topics.",
)

# Create manager that uses researcher as a tool
manager = nn.Agent(
    name="manager",
    model=model,
    instructions="You coordinate research tasks.",
    tools=[researcher],
)

async for event in manager.astream("Research AI trends"):
    agent = event.attributes.get("agent_name", "unknown")

    # Events from manager show "manager"
    # Events from researcher show "researcher"
    if event.name == EventType.TOOL_CALL:
        print(f"[{agent}] Calling: {event.attributes['tool_name']}")
    elif event.name == EventType.MODEL_RESPONSE_CHUNK:
        print(f"[{agent}] {event.attributes['chunk']}", end="")
```

## Example: Building a Chat UI

```python
import asyncio
from msgflux.nn import EventType

async def chat_with_streaming(agent, message):
    """Display events in real-time for a chat interface."""

    print(f"User: {message}")
    print("Assistant: ", end="", flush=True)

    current_tool = None
    final_response = None

    async for event in agent.astream(message):
        match event.name:
            case EventType.MODEL_RESPONSE_CHUNK:
                # Stream text chunks as they arrive
                chunk = event.attributes.get("chunk", "")
                print(chunk, end="", flush=True)

            case EventType.MODEL_REASONING_CHUNK:
                # Optionally show reasoning (dimmed or collapsed)
                pass

            case EventType.TOOL_CALL:
                tool = event.attributes["tool_name"]
                tool_id = event.attributes["tool_id"]
                current_tool = tool
                print(f"\n  ↳ Calling {tool}...", end="", flush=True)

            case EventType.TOOL_RESULT:
                print(f" ✓", flush=True)
                print("Assistant: ", end="", flush=True)

            case EventType.TOOL_ERROR:
                error = event.attributes["error"]
                print(f" ✗ Error: {error}", flush=True)

            case EventType.AGENT_COMPLETE:
                final_response = event.attributes.get("response")

    print()  # New line after streaming
    return final_response


# Usage
async def main():
    model = Model.chat_completion("openai/gpt-4o-mini")
    agent = nn.Agent(
        name="assistant",
        model=model,
        instructions="You are a helpful assistant.",
        tools=[calculator, web_search],
        config={"stream": True},
    )

    result = await chat_with_streaming(agent, "What is 25 * 17?")

asyncio.run(main())
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

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `EventStream` | Context manager for capturing events |
| `StreamEvent` | Immutable event dataclass |
| `EventType` | Constants for event type names |

### Convenience Functions

| Function | Description |
|----------|-------------|
| `add_event(name, attributes)` | Emit a custom event |
| `add_agent_start_event(...)` | Emit agent start event |
| `add_agent_complete_event(...)` | Emit agent complete event |
| `add_agent_step_event(...)` | Emit agent step event |
| `add_model_request_event(...)` | Emit model request event |
| `add_model_response_event(...)` | Emit model response event |
| `add_model_response_chunk_event(...)` | Emit response chunk event |
| `add_model_reasoning_event(...)` | Emit reasoning event |
| `add_tool_call_event(...)` | Emit tool call event |
| `add_tool_result_event(...)` | Emit tool result event |
| `add_tool_error_event(...)` | Emit tool error event |

### Imports

```python
from msgflux.nn import (
    EventStream,
    EventType,
    StreamEvent,
)

from msgflux.nn.events import (
    add_event,
    add_agent_start_event,
    add_agent_complete_event,
    add_tool_call_event,
    add_tool_result_event,
    # ... etc
)
```
