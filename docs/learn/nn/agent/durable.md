# Durable Execution

Durable execution makes agent runs resilient to crashes and failures. When enabled, the agent automatically checkpoints its state after each tool-call iteration. If the process dies mid-execution, progress is preserved and the run can be resumed from the last checkpoint instead of restarting from scratch.

## Overview

| Concept | Description |
|---------|-------------|
| **checkpointer** | A `CheckpointStore` passed to the agent constructor. `None` (default) disables durability with zero overhead. |
| **Save points** | After each tool-call iteration (`status="running"`) and after the final response (`status="completed"`). |
| **Failure** | On exception, saves `status="failed"` and re-raises the original error. |
| **Resume** | On next call, detects incomplete runs and rehydrates the conversation to continue where it left off. |

## Quick Start

```python
import msgflux as mf
from msgflux import nn

store = mf.SQLiteCheckpointStore(path=".msgflux/checkpoints.sqlite3")


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 22C in {city}"


class WeatherBot(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    tools = [get_weather]


agent = WeatherBot(checkpointer=store)
chat = mf.ChatMessages(session_id="user_42")

response = agent("What's the weather in Paris?", messages=chat)
```

With `checkpointer` set, every tool-call round-trip is persisted. Without it, the agent behaves exactly as before.

## How It Works

The agent inserts checkpoint saves at two points during execution:

```
Input
  |
  v
Model call
  |
  v
Tool call? ──yes──> Execute tools
  |                    |
  no                   v
  |              Save checkpoint (status="running")
  |                    |
  v                    v
  |              Model call (loop)
  |                    |
  v  <─────────────────┘
Final response
  |
  v
Save checkpoint (status="completed")
```

### Checkpoint key

Each checkpoint is stored under a composite key derived automatically from the agent and conversation:

| Key | Source | Example |
|-----|--------|---------|
| `namespace` | Agent name | `"WeatherBot"` |
| `session_id` | `ChatMessages.session_id` (or `"default"`) | `"user_42"` |
| `run_id` | Current turn ID from `ChatMessages` | `"turn_0"` |

### What is saved

Each checkpoint is a full snapshot:

```python
{
    "status": "running",          # or "completed" or "failed"
    "messages": chat._to_state(), # full conversation state
    "vars": {"temperature": 0.7}, # agent vars
}
```

## Resume from Crash

When the agent is called with a `checkpointer`, it checks for incomplete runs before starting a new one. If an incomplete run is found (status `"running"`), it rehydrates the `ChatMessages` and continues execution from where it left off.

```python
import msgflux as mf
from msgflux import nn

store = mf.SQLiteCheckpointStore(path=".msgflux/checkpoints.sqlite3")


class ResearchBot(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    tools = [search, summarize]


agent = ResearchBot(checkpointer=store)
chat = mf.ChatMessages(session_id="user_42")

# First attempt: crashes after 50 tool calls
try:
    response = agent("Research quantum computing", messages=chat)
except Exception:
    print("Process crashed, but 50 tool iterations are saved")

# Second attempt: resumes from iteration 50
response = agent("Research quantum computing", messages=chat)
```

The resume logic:

1. Agent checks `find_incomplete_runs(namespace, session_id)`
2. If an incomplete run exists, loads the state
3. Rehydrates `ChatMessages` via `_hydrate_state()`
4. Continues the tool-call loop from the last saved point

### Manual recovery

You can also inspect and recover incomplete runs directly through the store:

```python
store = mf.SQLiteCheckpointStore(path=".msgflux/checkpoints.sqlite3")

# Find runs that didn't finish
incomplete = store.find_incomplete_runs("ResearchBot", "user_42")

for run in incomplete:
    state = store.load_state("ResearchBot", "user_42", run["run_id"])
    chat = mf.ChatMessages()
    chat._hydrate_state(state["messages"])

    print(f"Run {run['run_id']}: {len(chat)} items saved")
    print(f"Last messages: {chat.to_chatml()[-2:]}")
```

## Error Handling

When an exception occurs during agent execution, the checkpoint is saved with `status="failed"` before the exception is re-raised. This preserves the conversation state up to the point of failure for debugging.

```python
store = mf.InMemoryCheckpointStore()


class Bot(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    tools = [my_tool]


agent = Bot(checkpointer=store)
chat = mf.ChatMessages(session_id="session_1")

try:
    response = agent("Do something", messages=chat)
except Exception:
    # State is saved with status="failed"
    runs = store.list_runs("Bot", "session_1", status="failed")
    if runs:
        state = store.load_state("Bot", "session_1", runs[0]["run_id"])
        print(f"Failed at status: {state['status']}")

        # Inspect conversation up to the failure
        restored = mf.ChatMessages()
        restored._hydrate_state(state["messages"])
        print(restored.to_chatml())
```

The three statuses:

| Status | Meaning |
|--------|---------|
| `running` | Execution in progress (saved after each tool iteration) |
| `completed` | Final response produced successfully |
| `failed` | Exception occurred, state saved before re-raise |

!!! note
    Failed runs are **not** picked up by automatic resume -- they are terminal. To retry after a failure, delete the failed run and call the agent again.

## Configuration

### Enabling durable execution

Pass a `CheckpointStore` to the `checkpointer` parameter:

```python
# In-memory (tests / prototyping)
store = mf.InMemoryCheckpointStore()
agent = nn.Agent("bot", model, checkpointer=store)

# SQLite (persistent across restarts)
store = mf.SQLiteCheckpointStore(path=".msgflux/checkpoints.sqlite3")
agent = nn.Agent("bot", model, checkpointer=store)
```

### Default behavior

When `checkpointer=None` (the default), no checkpoint logic runs. There is zero overhead -- the agent follows the same code path as without the feature.

### Streaming incompatibility

Durable execution is not compatible with `stream=True`. Using both raises a `ValueError` at initialization:

```python
# This raises ValueError
agent = nn.Agent(
    "bot",
    model,
    config={"stream": True},
    checkpointer=store,  # ValueError: `checkpointer` is not `stream=True` compatible
)
```

### Async support

Durable execution works with both `forward()` and `aforward()`. The async path uses `asave_state` and `afind_incomplete_runs` when available on the store, falling back to sync methods otherwise.

```python
import asyncio
import msgflux as mf
from msgflux import nn

store = mf.SQLiteCheckpointStore(path=".msgflux/checkpoints.sqlite3")


class AsyncBot(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    tools = [my_tool]


agent = AsyncBot(checkpointer=store)
chat = mf.ChatMessages(session_id="user_42")

response = asyncio.run(agent.acall("Do something", messages=chat))
```

## Integration with CheckpointStore

The agent works with any `CheckpointStore` implementation. See the [CheckpointStore documentation](../../stores/checkpoint.md) for the full API.

### InMemoryCheckpointStore

Best for tests and local prototyping. State is lost when the process exits.

```python
import msgflux as mf
from msgflux import nn

store = mf.InMemoryCheckpointStore()


class Bot(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    tools = [my_tool]


agent = Bot(checkpointer=store)
chat = mf.ChatMessages(session_id="test_session")

response = agent("Hello", messages=chat)

# Inspect saved state
runs = store.list_runs("Bot", "test_session")
# [{"run_id": "turn_0", "status": "completed", ...}]
```

### SQLiteCheckpointStore

Persistent storage that survives process restarts. Recommended for production.

```python
import msgflux as mf
from msgflux import nn

store = mf.SQLiteCheckpointStore(path=".msgflux/checkpoints.sqlite3")


class Bot(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    tools = [my_tool]


agent = Bot(checkpointer=store)
chat = mf.ChatMessages(session_id="user_42")

response = agent("Do a complex task", messages=chat)

# State persists across restarts
store.close()
```

### Custom implementations

Any backend that implements the `CheckpointStore` interface works with durable agents. See [Custom Implementations](../../stores/checkpoint.md#custom-implementations) for details.

## Failure Recovery Example

A complete example showing crash recovery with a long-running agent:

```python
import msgflux as mf
from msgflux import nn

store = mf.SQLiteCheckpointStore(path=".msgflux/checkpoints.sqlite3")


def fetch_page(url: str) -> str:
    """Fetch and return the content of a web page."""
    # ... HTTP request ...
    return f"Content of {url}"


def save_summary(filename: str, content: str) -> str:
    """Save a summary to a file."""
    # ... write to disk ...
    return f"Saved to {filename}"


class Researcher(nn.Agent):
    """Fetches multiple pages and writes summaries."""

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    tools = [fetch_page, save_summary]


agent = Researcher(checkpointer=store)
chat = mf.ChatMessages(session_id="research_session")

# Run 1: agent fetches 10 pages, then the process crashes
try:
    response = agent(
        "Fetch these 20 URLs and summarize each one: ...",
        messages=chat,
    )
except Exception as e:
    print(f"Crashed: {e}")
    print("Progress saved -- 10 tool iterations checkpointed")

# Run 2: resume from page 11
chat = mf.ChatMessages(session_id="research_session")
response = agent(
    "Fetch these 20 URLs and summarize each one: ...",
    messages=chat,
)
print("Completed all 20 pages")

# Cleanup old runs
store.clear(namespace="Researcher", session_id="research_session")
store.close()
```
