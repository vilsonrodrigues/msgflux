# Inline

`Inline` executes a declarative workflow over a [`dotdict`](dotdict.md) message.
The pipeline string is parsed once at construction; the same `Inline` object can
be called many times with different messages.

```python
from msgflux import Inline, dotdict
```

---

## How It Works

Every module in an `Inline` pipeline **mutates the message in place**. Modules
receive the shared `dotdict` as their only argument and write results directly
onto it. Return values are ignored.

```python
# Correct — write to msg, return nothing
def enrich(msg):
    msg.score = compute_score(msg.text)

# Also correct — return value is silently discarded
def tag(msg):
    msg.tag = "urgent"
    return msg  # has no effect
```

`Inline` reads the message's current state before each step, so every module
sees all changes written by the modules that ran before it.

---

## Constructor

```python
Inline(expression, modules, *, max_iterations=1000)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `expression` | `str` | Pipeline string. Parsed once at construction. |
| `modules` | `Mapping[str, Callable]` | Name-to-callable mapping. Callables may be plain functions, async functions, or `nn.Module` instances. |
| `max_iterations` | `int` | Maximum number of iterations a `@{…}` while loop may execute before a `RuntimeError` is raised. Default: `1000`. |

The expression is compiled at `__init__` time, so syntax errors surface
immediately — not during execution.

```python
flux = Inline(
    "prep -> [feat_a, feat_b] -> combine",
    {"prep": prep, "feat_a": feat_a, "feat_b": feat_b, "combine": combine},
)
```

### Reusing a `flux` object

Because parsing happens at construction, the same `flux` can be applied to many
messages without re-parsing the expression each time.

```python
flux = Inline("validate -> enrich -> rank", modules)

for request in batch:
    msg = dotdict(text=request)
    flux(msg)
    results.append(msg.rank)
```

---

## Execution

| Method | When to use |
|--------|-------------|
| `flux(msg)` | Synchronous execution |
| `await flux.acall(msg)` | Asynchronous execution |

Both return the same `msg` object after all steps have run.

---

## Syntax Overview

| Pattern | Description | Example |
|---------|-------------|---------|
| `->` | Sequential | `"a -> b -> c"` |
| `[…]` | Parallel | `"[b, c]"` |
| `{cond? a}` | Conditional (if) | `"{score > 0.9? accept}"` |
| `{cond? a, b}` | Conditional (if-else) | `"{is_vip? vip, standard}"` |
| `@{cond}: …;` | While loop | `"@{retries < 3}: fetch;"` |

---

## Sequential Execution

Modules run in the order listed, each seeing the message as left by the
previous step.

???+ example

    ```python
    from msgflux import Inline, dotdict

    def load(msg):
        msg.raw = "hello world"

    def tokenize(msg):
        msg.tokens = msg.raw.split()

    def count(msg):
        msg.n_tokens = len(msg.tokens)

    flux = Inline("load -> tokenize -> count", {
        "load": load,
        "tokenize": tokenize,
        "count": count,
    })

    msg = dotdict()
    flux(msg)

    print(msg.tokens)    # ["hello", "world"]
    print(msg.n_tokens)  # 2
    ```

---

## Parallel Execution

Modules inside `[…]` run concurrently in a thread pool. All of them receive
**the same message object**. Each module must write to a **different path** —
writing to the same key from two concurrent modules is a data race.

???+ example

    ```python
    from msgflux import Inline, dotdict

    def fetch_weather(msg):
        msg.weather = "sunny"          # writes to msg.weather

    def fetch_news(msg):
        msg.news = ["headline_1"]      # writes to msg.news

    def summarize(msg):
        msg.summary = f"{msg.weather} | {msg.news[0]}"

    flux = Inline(
        "[fetch_weather, fetch_news] -> summarize",
        {"fetch_weather": fetch_weather, "fetch_news": fetch_news, "summarize": summarize},
    )

    msg = dotdict()
    flux(msg)

    print(msg.summary)  # "sunny | headline_1"
    ```

!!! warning "Race Conditions"
    Parallel modules share the same `dotdict`. Writing to the **same key** from
    two modules simultaneously will produce unpredictable results. Design
    parallel stages so each one owns a distinct subtree of the message.

    ```python
    # Safe — each module owns its own key
    def feat_a(msg): msg.feat_a = ...
    def feat_b(msg): msg.feat_b = ...

    # Unsafe — both append to the same list
    def feat_a(msg): msg.results.append("a")
    def feat_b(msg): msg.results.append("b")
    ```

---

## Conditionals

Conditions are evaluated against the current message state at runtime.

### If

```
{field operator value ? true_module}
```

???+ example

    ```python
    from msgflux import Inline, dotdict

    def flag_urgent(msg):
        msg.label = "urgent"

    def flag_normal(msg):
        msg.label = "normal"

    flux = Inline(
        "{priority > 7 ? flag_urgent, flag_normal}",
        {"flag_urgent": flag_urgent, "flag_normal": flag_normal},
    )

    msg = dotdict(priority=9)
    flux(msg)
    print(msg.label)  # "urgent"
    ```

### None checks

```python
"{user.audio is not None ? transcribe}"
"{user.name is None ? ask_name, greet}"
```

### Logical operators

| Operator | Description |
|----------|-------------|
| `&` | AND |
| `\|\|` | OR |
| `!` | NOT |

```python
"{user.active == true & !user.banned == true ? allow, deny}"
"{plan == 'premium' || credits > 100 ? vip_flow, standard_flow}"
```

---

## Comparison Operators

Conditions follow the format `key_path operator value`. The key path uses dot
notation to access nested fields (e.g. `user.profile.age`).

| Operator | Description |
|----------|-------------|
| `==` | Equal |
| `!=` | Not equal |
| `>` | Greater than |
| `<` | Less than |
| `>=` | Greater or equal |
| `<=` | Less or equal |
| `is None` | Value is None or key is absent |
| `is not None` | Value is present and not None |

---

## While Loops

Syntax: `@{condition}: actions;`

The body (`actions`) is any valid pipeline expression. The loop re-evaluates
the condition against the current message before each iteration.

???+ example

    ```python
    from msgflux import Inline, dotdict

    def init(msg):
        msg.counter = 0
        msg.total = 0

    def step(msg):
        msg.counter += 1
        msg.total += msg.counter     # 1 + 2 + 3 + 4 + 5

    def done(msg):
        msg.finished = True

    flux = Inline(
        "init -> @{counter < 5}: step; -> done",
        {"init": init, "step": step, "done": done},
    )

    msg = dotdict()
    flux(msg)

    print(msg.counter)   # 5
    print(msg.total)     # 15
    print(msg.finished)  # True
    ```

### `max_iterations`

`max_iterations` caps the number of times a while body may run. If the
condition never becomes false, `Inline` raises a `RuntimeError` instead of
looping forever.

```python
# Default limit: 1000 iterations
flux = Inline("@{active}: poll;", modules)

# Raise after 10 iterations (useful for tests or tight retry budgets)
flux = Inline("@{active}: poll;", modules, max_iterations=10)
```

```python
# RuntimeError: While loop exceeded maximum iterations (10).
# Possible infinite loop detected. Condition: active
```

!!! tip
    Set `max_iterations` to a small value when the loop represents a **retry
    budget** — this makes the limit explicit and surfaces bugs early.

---

## Async Execution

Use `await flux.acall(msg)` when modules are async functions or `nn.Module`
instances with an `acall` method. Sync modules inside an async pipeline are
called directly (no thread pool).

???+ example

    ```python
    import asyncio
    from msgflux import Inline, dotdict

    async def fetch(msg):
        # simulate I/O
        await asyncio.sleep(0)
        msg.data = "fetched"

    async def process(msg):
        msg.result = msg.data.upper()

    flux = Inline("fetch -> process", {"fetch": fetch, "process": process})

    msg = dotdict()
    asyncio.run(flux.acall(msg))

    print(msg.result)  # "FETCHED"
    ```

Parallel stages in async mode use `ascatter_gather` under the hood, so each
branch runs as a separate coroutine.

---

## With `nn.Module`

Store the workflow string as a buffer so it travels with the module's state.
Instantiate `Inline` inside `forward` / `aforward` — it is lightweight because
parsing is fast.

???+ example

    ```python
    import msgflux.nn as nn
    from msgflux import Inline

    class TranscriptionPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.transcriber = nn.Transcriber(...)
            self.extractor = nn.Agent(...)
            self.components = nn.ModuleDict({
                "transcriber": self.transcriber,
                "extractor": self.extractor,
            })
            self.register_buffer(
                "workflow",
                "{audio is not None? transcriber} -> extractor",
            )

        def forward(self, msg):
            flux = Inline(self.workflow, self.components)
            return flux(msg)

        async def aforward(self, msg):
            flux = Inline(self.workflow, self.components)
            return await flux.acall(msg)
    ```

---

## Complex Workflow

All constructs compose freely.

???+ example

    ```python
    from msgflux import Inline, dotdict

    workflow = """
        ingest
        -> {audio is not None? transcribe}
        -> [extract_entities, analyze_sentiment]
        -> @{confidence < 0.8}: refine;
        -> {is_urgent == true? priority_handler, standard_handler}
        -> finalize
    """

    flux = Inline(workflow, modules)
    flux(msg)
    ```

    Execution order:

    1. `ingest` — always runs
    2. `transcribe` — only if `msg.audio` is not None
    3. `extract_entities` and `analyze_sentiment` — run in parallel
    4. `refine` — loops until `msg.confidence >= 0.8` (capped at 1000 iterations)
    5. `priority_handler` or `standard_handler` — branch on `msg.is_urgent`
    6. `finalize` — always runs
