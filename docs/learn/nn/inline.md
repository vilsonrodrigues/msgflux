# inline DSL

The `inline` function allows you to orchestrate modules using a **declarative workflow language**. Change your workflow at runtime without modifying code.

## Quick Start

```python
import msgflux.nn.functional as F
from msgflux import dotdict

def preprocess(msg):
    msg.preprocessed = True

def analyze(msg):
    msg.result = "Analysis complete"

modules = {
    "preprocess": preprocess,
    "analyze": analyze
}

message = dotdict()
F.inline("preprocess -> analyze", modules, message)

print(message.result)  # "Analysis complete"
```

!!! info "Modules modify in-place"
    Modules receive the `message` and modify it **in-place**. They do not need to return anything — the `message` object is shared across all steps.

---

## Syntax Overview

| Pattern | Description | Example |
|---------|-------------|---------|
| `->` | Sequential | `"a -> b -> c"` |
| `[...]` | Parallel | `"a -> [b, c] -> d"` |
| `{cond? a}` | If | `"{has_data? process}"` |
| `{cond? a, b}` | If-else | `"{is_premium? vip, standard}"` |
| `@{cond}: a;` | While loop | `"@{count < 5}: increment;"` |

---

## Sequential Execution

Use `->` to chain modules:

```python
def step1(msg):
    msg.step1 = "done"

def step2(msg):
    msg.step2 = "done"

def step3(msg):
    msg.step3 = "done"

modules = {"step1": step1, "step2": step2, "step3": step3}
message = dotdict()

F.inline("step1 -> step2 -> step3", modules, message)
```

---

## Parallel Execution

Use `[...]` to run modules in parallel:

```python
def fetch_a(msg):
    msg.data_a = "result_a"

def fetch_b(msg):
    msg.data_b = "result_b"

def combine(msg):
    msg.combined = f"{msg.data_a} + {msg.data_b}"

modules = {
    "fetch_a": fetch_a,
    "fetch_b": fetch_b,
    "combine": combine
}

message = dotdict()
F.inline("[fetch_a, fetch_b] -> combine", modules, message)

print(message.combined)  # "result_a + result_b"
```

!!! warning "Race Conditions"
    In parallel execution, each module modifies the shared `message` in-place. Avoid writing to the **same key** from multiple parallel modules, as concurrent writes may produce unpredictable results.

---

## Conditionals

### If

Execute only if condition is true:

```python
def transcribe(msg):
    msg.transcription = "Hello world"

def process(msg):
    msg.processed = True

modules = {"transcribe": transcribe, "process": process}

message = dotdict()
message.audio = "audio.mp3"

F.inline("{audio is not None? transcribe} -> process", modules, message)
```

### If-Else

Execute one branch or the other:

```python
def adult_flow(msg):
    msg.result = "Welcome, adult"

def child_flow(msg):
    msg.result = "Hi, young one"

modules = {"adult_flow": adult_flow, "child_flow": child_flow}

message = dotdict()
message.set("user.age", 21)

F.inline("{user.age > 18 ? adult_flow, child_flow}", modules, message)
print(message.result)  # "Welcome, adult"
```

---

## Logical Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `&` | AND | `"is_admin == true & is_active == true"` |
| `\|\|` | OR | `"is_premium == true \|\| has_coupon == true"` |
| `!` | NOT | `"!(is_banned == true)"` |

```python
message = dotdict()
message.set("user.is_active", True)
message.set("user.is_banned", False)

F.inline(
    "{user.is_active == True & !user.is_banned == True ? grant, deny}",
    modules,
    message
)
```

---

## None Verification

Check if a field is None:

```python
# Check if None
"{user.name is None ? ask_name, greet}"

# Check if not None
"{user.audio is not None ? transcribe}"
```

---

## While Loops

Execute repeatedly while condition is true:

```python
async def prep(msg):
    msg.counter = 0

async def increment(msg):
    msg.counter = msg.get("counter", 0) + 1

async def finalize(msg):
    msg.done = True

modules = {
    "prep": prep,
    "increment": increment,
    "finalize": finalize
}

message = dotdict()
await F.ainline(
    "prep -> @{counter < 5}: increment; -> finalize",
    modules,
    message
)

print(message.counter)  # 5
```

---

## Async Execution

Use `ainline` for async modules:

```python
import msgflux.nn.functional as F

async def async_fetch(msg):
    await asyncio.sleep(0.1)
    msg.data = "fetched"

async def process(msg):
    msg.processed = True

modules = {"fetch": async_fetch, "process": process}

message = dotdict()
await F.ainline("fetch -> process", modules, message)
```

---

## With nn.Module

Store workflow as a buffer:

```python
import msgflux.nn as nn
import msgflux.nn.functional as F

class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.transcriber = nn.Transcriber(...)
        self.extractor = nn.Agent(...)
        
        self.components = nn.ModuleDict({
            "transcriber": self.transcriber,
            "extractor": self.extractor
        })
        
        # Workflow stored as buffer
        self.register_buffer(
            "flux", 
            "{user_audio is not None? transcriber} -> extractor"
        )
    
    def forward(self, msg):
        return F.inline(self.flux, self.components, msg)
    
    async def aforward(self, msg):
        return await F.ainline(self.flux, self.components, msg)
```

---

## Complex Example

```python
# Full workflow with all patterns
workflow = """
    prep 
    -> {has_audio is not None? transcribe}
    -> [analyze_sentiment, extract_entities]
    -> @{confidence < 0.8}: refine;
    -> {is_urgent == true? priority_handler, standard_handler}
    -> finalize
"""

F.inline(workflow, modules, message)
```

---

## Message Access

The DSL accesses message fields using dot notation:

```python
message = dotdict()
message.set("user.age", 25)
message.set("config.is_premium", True)

# Access in conditions
"{user.age > 18 ? adult}"
"{config.is_premium == true ? vip}"
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
"{items.0.price > 100 ? expensive}"
```
