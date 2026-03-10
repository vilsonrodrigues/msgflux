# CheckpointStore

`CheckpointStore` is a unified persistence layer for agent and pipeline checkpoints. It uses a composite key `(namespace, session_id, run_id)` to partition state across components, sessions, and individual runs.

## Overview

| Concept | Description |
|---------|-------------|
| **State** | Complete snapshot of current execution (UPSERT). Each save replaces the entire state. |
| **Events** | Append-only audit trail for debugging and replay. |
| **Composite Key** | `namespace` (component), `session_id` (user/session), `run_id` (execution). |
| **Snapshot model** | Every `save_state` persists a **full snapshot**, not incremental diffs. Recovery loads one row. |

## Quick Start

```python
import msgflux as mf

# In-memory (tests / prototyping)
store = mf.InMemoryCheckpointStore()

# SQLite (persistent)
store = mf.SQLiteCheckpointStore(path=".msgflux/checkpoints.sqlite3")

# Save state
store.save_state("my_agent", "session_1", "run_001", {
    "status": "running",
    "step": 1,
    "data": {"key": "value"},
})

# Load state
state = store.load_state("my_agent", "session_1", "run_001")
```

## Implementations

### InMemoryCheckpointStore

In-memory store for tests and local prototyping. Thread-safe via `RLock`. Does not persist across process restarts.

```python
store = mf.InMemoryCheckpointStore()
```

### SQLiteCheckpointStore

SQLite-backed store with production features:

- **WAL mode** for read/write concurrency
- **Foreign keys with CASCADE** delete (events are removed with their run)
- **UPSERT** semantics on state (no duplicates)
- **JSON as TEXT** for easy inspection via `sqlite3` CLI

```python
store = mf.SQLiteCheckpointStore(path=".msgflux/checkpoints.sqlite3")

# Close when done
store.close()
```

## State Operations

State is a full snapshot (a dict). Each `save_state` replaces the previous snapshot for that key.

```python
# Save (creates or updates)
store.save_state("agent", "session", "run_1", {
    "status": "running",
    "messages": chat._to_state(),
    "vars": {"temperature": 0.7},
})

# Load
state = store.load_state("agent", "session", "run_1")
# None if not found
```

## Event Operations

Events are append-only and ordered chronologically. Use them for audit trails and debugging.

```python
# Append
store.append_event("agent", "session", "run_1", {
    "event_type": "turn_completed",
    "turn": 1,
})

# Load all events for a run
events = store.load_events("agent", "session", "run_1")
# [{"event_type": "turn_completed", "turn": 1}, ...]
```

## Atomic Save

`save_with_event` persists state and event together. The SQLite implementation uses a transaction for atomicity.

```python
store.save_with_event(
    "agent", "session", "run_1",
    state={"status": "completed", "messages": chat._to_state()},
    event={"event_type": "run_completed"},
)
```

## Integration with ChatMessages

`ChatMessages` supports serialization via `_to_state()` / `_hydrate_state()`. This allows full conversation state to be persisted and recovered.

```python
import msgflux as mf

store = mf.InMemoryCheckpointStore()
chat = mf.ChatMessages(session_id="user_42")

# Build conversation
chat.begin_turn(inputs="What is 2+2?", vars={"temperature": 0.7})
chat.add_user("What is 2+2?")
chat.add_assistant("4")
chat.end_turn(assistant_output="4")

# Persist
store.save_state("math_agent", "user_42", "run_1", {
    "status": "completed",
    "messages": chat._to_state(),
})

# Recover
loaded = store.load_state("math_agent", "user_42", "run_1")
restored = mf.ChatMessages()
restored._hydrate_state(loaded["messages"])

restored.to_chatml()
# [{"role": "user", "content": "What is 2+2?"},
#  {"role": "assistant", "content": "4"}]

restored.turns[0]["inputs"]
# "What is 2+2?"
```

### What is serialized

`_to_state()` captures the full ChatMessages state:

| Field | Content |
|-------|---------|
| `items` | All messages (user, assistant, reasoning, tool, turn markers) |
| `turns` | Complete turn records: inputs, context_inputs, vars, assistant_output, response_type, response_metadata, timestamps, status |
| `session_id` | Session identifier |
| `namespace` | Namespace |
| `metadata` | Custom metadata |
| `active_turn_index` | Index of the in-progress turn (if any) |

Each `save_state` is a **full snapshot** - not a diff. Recovery is a single `load_state` + `_hydrate_state`.

## Agent Integration

Pass a `checkpointer` to the Agent constructor to enable automatic checkpointing. The Agent saves state after **each tool-call iteration** and at the end of execution.

```python
import msgflux as mf
import msgflux.nn as nn

store = mf.SQLiteCheckpointStore()

class MyAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    tools = [my_tool]

agent = MyAgent(checkpointer=store)
chat = mf.ChatMessages(session_id="user_42")

# If the agent makes 300 tool calls and fails on 301,
# the first 300 are saved and can be recovered.
response = agent("Do a complex task", messages=chat)
```

### What gets checkpointed

| Moment | Status | Content |
|--------|--------|---------|
| After each tool-call iteration | `running` | Messages so far (including tool results) |
| After final response | `completed` | Full conversation with response |

The checkpoint key is derived automatically:

- **namespace**: agent name (e.g. `"MyAgent"`)
- **session_id**: from `ChatMessages.session_id` (or `"default"`)
- **run_id**: from the current turn ID

### Failure Recovery

If the process crashes mid-execution, the run stays with `status="running"`:

```python
store = mf.SQLiteCheckpointStore()

# Find runs that didn't finish
incomplete = store.find_incomplete_runs("MyAgent", "user_42")

for run in incomplete:
    state = store.load_state("MyAgent", "user_42", run["run_id"])
    chat = mf.ChatMessages()
    chat._hydrate_state(state["messages"])

    # The active turn and all tool-call results are preserved
    print(f"Resuming {run['run_id']}: {len(chat)} items")
```

## Queries

### List runs

```python
# All runs for a session (most recent first)
runs = store.list_runs("agent", "session")
# [{"run_id": "r3", "status": "completed", "updated_at": ...},
#  {"run_id": "r2", "status": "failed", ...}, ...]

# Filter by status
completed = store.list_runs("agent", "session", status="completed")

# Limit results
latest = store.list_runs("agent", "session", limit=5)
```

### Load latest run

```python
# Resume the most recent run
state = store.load_latest_run("agent", "session")
if state:
    chat = mf.ChatMessages()
    chat._hydrate_state(state["messages"])
```

### Find incomplete runs

```python
incomplete = store.find_incomplete_runs("agent", "session")
```

### Delete a run

```python
store.delete_run("agent", "session", "run_1")
# Events are cascade-deleted (SQLite)
```

## Cleanup

```python
# Remove all runs
store.clear()

# Remove runs in a namespace
store.clear(namespace="agent")

# Remove runs in a specific session
store.clear(namespace="agent", session_id="session_1")

# Remove runs older than 1 hour
store.clear(older_than=3600)
```

## Composite Key

The three-part key provides natural isolation:

| Key | Purpose | Example |
|-----|---------|---------|
| `namespace` | Component identity | `"weather_bot"`, `"inline:{hash}"` |
| `session_id` | User or conversation | `"user_42"`, `"default"` |
| `run_id` | Individual execution | `"run_001"`, UUID |

```python
# Same table, different namespaces
store.save_state("agent_a", "session", "run_1", {...})
store.save_state("agent_b", "session", "run_1", {...})

# Isolated — different agents don't see each other's state
store.list_runs("agent_a", "session")  # only agent_a runs
```

## Custom Implementations

Extend `CheckpointStore` for other backends (Redis, PostgreSQL, etc.):

```python
from msgflux.data.stores import CheckpointStore

class RedisCheckpointStore(CheckpointStore):
    def save_state(self, namespace, session_id, run_id, state):
        ...

    def load_state(self, namespace, session_id, run_id):
        ...

    # ... implement all abstract methods
```

For async backends, extend `AsyncCheckpointStore`:

```python
from msgflux.data.stores import AsyncCheckpointStore

class AsyncRedisCheckpointStore(AsyncCheckpointStore):
    async def asave_state(self, namespace, session_id, run_id, state):
        ...
```
