# Inline DSL

`Inline` is a first-class pipeline orchestrator. It composes callables into a pipeline using a lightweight DSL expression, with optional checkpoint-per-step durability.

## Quick Start

```python
import msgflux as mf

def preprocess(msg):
    return {"preprocessed": True}

def analyze(msg):
    return {"result": "Analysis complete"}

pipeline = mf.Inline(
    "preprocess -> analyze",
    modules={"preprocess": preprocess, "analyze": analyze},
)

result = pipeline(mf.dotdict())
print(result["result"])  # "Analysis complete"
```

## Module Return Patterns

Modules receive a `dotdict` message and can return results in two ways:

### Delta pattern (recommended)

Return a `dict` with the fields to merge. The pipeline merges it into the message automatically via `dotdict.apply()`:

```python
def enrich(msg):
    return {"enriched": True, "score": 0.95}
```

This is the preferred pattern — it makes modules **pure functions** that are easy to test, compose, and reason about. Deltas also enable [signals](#signals) (`$break`, `$stop`) for flow control.

### In-place mutation (legacy)

Mutate the message directly and return `None`:

```python
def enrich(msg):
    msg["enriched"] = True
    msg["score"] = 0.95
```

!!! warning "Deprecated"
    In-place mutation still works for backwards compatibility, but it is **deprecated**. It does not support signals and makes modules harder to test in isolation. Prefer the delta pattern for new code.

---

## Syntax Overview

| Pattern | Description | Example |
|---------|-------------|---------|
| `->` | Sequential | `"a -> b -> c"` |
| `[...]` | Parallel | `"a -> [b, c] -> d"` |
| `{c1 ? a, c2 ? b, default}` | Multi-branch conditional | `"{tier == 'vip' ? vip, standard}"` |
| `@{cond}: actions;` | While loop | `"@{count < 5}: increment;"` |

---

## Sequential Execution

Use `->` to chain modules:

```python
import msgflux as mf

def step1(msg):
    return {"step1": "done"}

def step2(msg):
    return {"step2": "done"}

modules = {"step1": step1, "step2": step2}

pipeline = mf.Inline("step1 -> step2", modules=modules)
result = pipeline(mf.dotdict())

print(result["step1"])  # "done"
print(result["step2"])  # "done"
```

---

## Parallel Execution

Use `[...]` to run modules concurrently:

```python
import msgflux as mf

def fetch_a(msg):
    return {"data_a": "result_a"}

def fetch_b(msg):
    return {"data_b": "result_b"}

def combine(msg):
    return {"combined": f"{msg['data_a']} + {msg['data_b']}"}

modules = {"fetch_a": fetch_a, "fetch_b": fetch_b, "combine": combine}

pipeline = mf.Inline("[fetch_a, fetch_b] -> combine", modules=modules)
result = pipeline(mf.dotdict())

print(result["combined"])  # "result_a + result_b"
```

!!! warning "Key Conflicts"
    When parallel modules write the **same key**, a warning is logged and the last writer wins. Use distinct keys per module.

---

## Conditionals

### If-Else (binary)

```python
import msgflux as mf

def adult(msg):
    return {"greeting": "Welcome, adult"}

def child(msg):
    return {"greeting": "Hi, young one"}

modules = {"adult": adult, "child": child}

pipeline = mf.Inline("{age > 18 ? adult, child}", modules=modules)
result = pipeline(mf.dotdict({"age": 21}))

print(result["greeting"])  # "Welcome, adult"
```

### Multi-branch

Route to the first matching condition. The last item without `?` is the default:

```python
"{tier == 'premium' ? premium_handler, tier == 'standard' ? standard_handler, basic_handler}"
```

You can also use `_` as an explicit wildcard default:

```python
"{score >= 90 ? grade_a, score >= 70 ? grade_b, _ ? grade_c}"
```

If no branch matches and there is no default, nothing executes.

---

## Signals

Modules can emit **signals** to control pipeline flow by including special keys in their return dict:

| Signal | Effect |
|--------|--------|
| `$stop` | Halt the entire pipeline immediately |
| `$break` | Exit the current while loop (remaining steps after the loop continue) |

```python
import msgflux as mf

def check(msg):
    if msg.get("counter", 0) >= 3:
        return {"$stop": True, "reason": "threshold reached"}
    return {"counter": msg.get("counter", 0) + 1}

pipeline = mf.Inline(
    "@{counter < 100}: check;",
    modules={"check": check},
)

result = pipeline(mf.dotdict({"counter": 0}))
print(result["counter"])  # 3
print(result["reason"])   # "threshold reached"
```

Signal keys are stripped from the delta before merging into the message.

---

## While Loops

Execute repeatedly while condition is true:

```python
import msgflux as mf

def increment(msg):
    return {"counter": msg.get("counter", 0) + 1}

def finalize(msg):
    return {"done": True}

modules = {"increment": increment, "finalize": finalize}

pipeline = mf.Inline(
    "@{counter < 5}: increment; -> finalize",
    modules=modules,
)

result = pipeline(mf.dotdict({"counter": 0}))
print(result["counter"])  # 5
print(result["done"])     # True
```

!!! warning "Infinite Loops"
    While loops have a maximum iteration limit (default: 1000). A `RuntimeError` is raised if exceeded. Customize with `max_iterations`.

---

## Durable Execution

Pass a `CheckpointStore` to enable checkpoint-per-step durability. If the process crashes, re-running with the same `session_id` and `run_id` resumes from the last completed step.

```python
import msgflux as mf

store = mf.InMemoryCheckpointStore()

pipeline = mf.Inline(
    "extract -> enrich -> summarize",
    modules={"extract": extract, "enrich": enrich, "summarize": summarize},
)

# Durable run
result = pipeline(
    mf.dotdict({"input": "data"}),
    store=store,
    session_id="user_42",
    run_id="run_1",
)
```

### Crash and Resume

```python
store = mf.InMemoryCheckpointStore()

pipeline = mf.Inline("step_a -> step_b -> step_c", modules=modules)

# First run: crashes at step_b
try:
    pipeline(mf.dotdict(), store=store, session_id="s1", run_id="r1")
except RuntimeError:
    pass  # step_a is checkpointed

# Fix the issue, re-run with same session/run — resumes from step_b
result = pipeline(mf.dotdict(), store=store, session_id="s1", run_id="r1")
```

### Durable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `store` | `None` | `CheckpointStore` instance. `None` disables durability. |
| `session_id` | Inherited or `"default"` | Session identifier (inherits from parent context). |
| `run_id` | Auto-generated UUID | Unique execution identifier. |
| `namespace` | `"inline"` | Checkpoint namespace. |
| `max_retries` | `0` | Per-step retry limit. |
| `retry_delay` | `1.0` | Seconds between retries. |

### Event Audit Trail

Durable runs emit events for debugging and replay:

```python
events = store.load_events("inline", "user_42", "run_1")
# [{"type": "run_started", ...},
#  {"type": "step_completed", "step_name": "extract", ...},
#  {"type": "step_completed", "step_name": "enrich", ...},
#  {"type": "step_completed", "step_name": "summarize", ...},
#  {"type": "run_completed", ...}]
```

### Signal Behavior with Durability

- **`$stop`** saves with `status="stopped"` (terminal). Resume starts a fresh run.
- **`$break`** saves with `status="running"` and exits the while loop. Remaining steps continue.
- **Failed steps** save with `status="failed"` and re-raise the exception.

---

## Async Execution

Use `acall` for async modules:

```python
import asyncio
import msgflux as mf

async def async_fetch(msg):
    await asyncio.sleep(0.1)
    return {"data": "fetched"}

async def process(msg):
    return {"processed": True}

modules = {"fetch": async_fetch, "process": process}

pipeline = mf.Inline("fetch -> process", modules=modules)
result = asyncio.run(pipeline.acall(mf.dotdict()))
```

Durable async works the same way — pass `store` to `acall`:

```python
result = await pipeline.acall(
    mf.dotdict(),
    store=store,
    session_id="user_42",
)
```

---

## Session Context

`Inline` propagates session context via `msgflux.context.session_context`. Nested `Inline` pipelines inherit the parent session automatically:

```python
import msgflux as mf

sub = mf.Inline("inner_step", modules={"inner_step": inner_fn})

def run_sub(msg):
    sub(msg)  # inherits parent session_id

pipeline = mf.Inline("outer -> run_sub", modules={"outer": outer_fn, "run_sub": run_sub})
pipeline(mf.dotdict(), session_id="parent_session")
```

---

## Logical Operators

Combine conditions with logical operators:

| Operator | Description |
|----------|-------------|
| `&` | AND |
| `\|\|` | OR |
| `!` | NOT |

```python
# Grant access if active AND not banned
"{user.is_active == True & !user.is_banned == True ? grant, deny}"
```

---

## Comparison Operators

| Operator | Description |
|----------|-------------|
| `==` | Equal |
| `!=` | Not equal |
| `>` | Greater than |
| `<` | Less than |
| `>=` | Greater or equal |
| `<=` | Less or equal |
| `is None` | Is None |
| `is not None` | Is not None |

```python
"{score >= 0.9 ? high_quality, review}"
"{status != 'completed' ? process}"
"{user.name is None ? ask_name, greet}"
```

---

## Message Access

The DSL accesses message fields using dot notation:

```python
message = mf.dotdict()
message.set("user.age", 25)
message.set("config.tier", "premium")

"{user.age > 18 ? adult}"
"{config.tier == 'premium' ? vip}"
```

---

## With nn.Module

Compose `Inline` inside custom modules:

```python
import msgflux as mf
import msgflux.nn as nn

class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()

        self.transcriber = nn.Transcriber(...)
        self.extractor = nn.Agent(...)

        self.components = nn.ModuleDict({
            "transcriber": self.transcriber,
            "extractor": self.extractor,
        })

        self.flux = mf.Inline(
            "{user_audio is not None ? transcriber} -> extractor",
            modules=self.components,
        )

    def forward(self, msg):
        return self.flux(msg)

    async def aforward(self, msg):
        return await self.flux.acall(msg)
```

---

## Complete Example

```python
import msgflux as mf

def classify(msg):
    score = msg.get("score", 0)
    if score >= 90:
        return {"tier": "premium"}
    if score >= 50:
        return {"tier": "standard"}
    return {"tier": "basic"}

def enrich(msg):
    return {"enriched": True}

def validate(msg):
    return {"validated": True}

def premium_handler(msg):
    return {"handler": "premium", "discount": 0.2}

def standard_handler(msg):
    return {"handler": "standard", "discount": 0.1}

def basic_handler(msg):
    return {"handler": "basic", "discount": 0.0}

pipeline = mf.Inline(
    "classify -> [enrich, validate] -> "
    "{tier == 'premium' ? premium_handler, "
    "tier == 'standard' ? standard_handler, "
    "basic_handler}",
    modules={
        "classify": classify,
        "enrich": enrich,
        "validate": validate,
        "premium_handler": premium_handler,
        "standard_handler": standard_handler,
        "basic_handler": basic_handler,
    },
)

result = pipeline(mf.dotdict({"score": 75}))

print(result["tier"])      # "standard"
print(result["enriched"])  # True
print(result["handler"])   # "standard"
print(result["discount"])  # 0.1
```
