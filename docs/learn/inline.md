# Inline

`Inline` is a small domain-specific language (DSL) for defining
**message-centric workflows**.

Each pipeline step executes a module that **mutates a shared `dotdict` message
in place**. The DSL describes *control flow* — sequence, branching, parallelism,
and loops — while the modules implement the actual logic.

The pipeline expression is parsed once at construction and reused across
executions.

```python
from msgflux import Inline
```

---

## How It Works

Every module in an `Inline` pipeline receives the shared `dotdict` as its only
argument and writes results directly onto it. **Return values are ignored.**

Modules should treat the message as the **single source of truth**. This design
ensures that pipelines behave consistently in sequential, parallel, and async
execution — the caller always reads results from `msg`, never from a return
value.

```python
# Correct — write to msg, return nothing
def enrich(msg):
    msg.score = compute_score(msg.text)

# Return value is silently discarded — still works, but misleading
def tag(msg):
    msg.tag = "urgent"
    return msg  # has no effect
```

`Inline` reads the message's current state before each step, so every module
sees all changes written by the modules that ran before it.

### Compiled execution

Internally, the pipeline expression is parsed into a list of execution steps
representing an abstract syntax tree (AST). This structure is built once at
`__init__` and reused on every `__call__` or `acall`, so execution does not
require re-parsing the DSL.

---

## Constructor

```python
Inline(expression, modules, *, max_iterations=1000)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `expression` | `str` | Pipeline string. Parsed once at construction. Syntax errors surface immediately. |
| `modules` | `Mapping[str, Callable]` | Name-to-callable mapping. Values may be plain functions, async functions, or `nn.Module` instances. |
| `max_iterations` | `int` | Maximum number of iterations any `@{…}` while loop may run before a `RuntimeError` is raised. Default: `1000`. |

```python
flux = Inline(
    "prep -> [feat_a, feat_b] -> combine",
    {"prep": prep, "feat_a": feat_a, "feat_b": feat_b, "combine": combine},
)
```

### Reusing a pipeline

Because parsing happens at construction, the same `flux` object can be applied
to many messages without re-parsing the expression each time.

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
| `flux(msg)` | Synchronous |
| `await flux.acall(msg)` | Asynchronous |

Both return the same `msg` object after all steps have run.

---

## Syntax Overview

| Pattern | Description | Example |
|---------|-------------|---------|
| `->` | Sequential | `"a -> b -> c"` |
| `[…]` | Parallel | `"[b, c]"` |
| `{msg.key op val? a}` | Conditional (if) | `"{msg.score > 0.9? accept}"` |
| `{msg.key op val? a, b}` | Conditional (if-else) | `"{msg.is_vip == true? vip, standard}"` |
| `@{msg.key op val}: …;` | While loop | `"@{msg.retries < 3}: fetch;"` |

`msg.key` represents a **field path read from the current message** at runtime
(e.g. `score`, `user.age`, `output.confidence`). Conditions are never
free-form Python — they are always a comparison between a message field and a
literal value.

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
**the same message object**.

### Design rule

Parallel modules must write to **disjoint message paths**. Writing to the same
key from two concurrent modules is a data race.

```python
# Safe — each module owns its own subtree
def fetch_weather(msg): msg.weather = "sunny"
def fetch_news(msg):    msg.news = ["headline_1"]

# Unsafe — both mutate the same list
def feat_a(msg): msg.results.append("a")
def feat_b(msg): msg.results.append("b")
```

???+ example

    ```python
    from msgflux import Inline, dotdict

    def fetch_weather(msg):
        msg.weather = "sunny"

    def fetch_news(msg):
        msg.news = ["headline_1"]

    def summarize(msg):
        msg.summary = f"{msg.weather} | {msg.news[0]}"

    flux = Inline(
        "[fetch_weather, fetch_news] -> summarize",
        {
            "fetch_weather": fetch_weather,
            "fetch_news": fetch_news,
            "summarize": summarize,
        },
    )

    msg = dotdict()
    flux(msg)

    print(msg.summary)  # "sunny | headline_1"
    ```

!!! warning "Race Conditions"
    Parallel modules share the same `dotdict`. Writing to the **same key** from
    two modules simultaneously produces unpredictable results. Design parallel
    stages so each one owns a distinct subtree of the message.

---

## Conditionals

Conditions are evaluated against the current message state at runtime.

### Syntax

```
{key_path operator value ? true_module}
{key_path operator value ? true_module, false_module}
```

If the false branch is omitted, nothing is executed when the condition is false.

```python
# Only runs accept when score > 0.9; does nothing otherwise
"{score > 0.9 ? accept}"

# Equivalent to: if score > 0.9: accept(msg) else: reject(msg)
"{score > 0.9 ? accept, reject}"
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

### Key Paths

Conditions access message fields using dot notation resolved via
`message.get(key_path)`.

```
user.profile.age
request.headers.authorization
output.agent
```

If a key path does not exist in the message, its value is treated as `None`.
This means `is None` checks work naturally for absent keys.

### None Checks

```python
"{user.audio is not None ? transcribe}"
"{user.name is None ? ask_name, greet}"
```

### Comparison Operators

| Operator | Description |
|----------|-------------|
| `==` | Equal |
| `!=` | Not equal |
| `>` | Greater than |
| `<` | Less than |
| `>=` | Greater or equal |
| `<=` | Less or equal |
| `is None` | Value is `None` or key is absent |
| `is not None` | Value is present and not `None` |

### Type Coercion

Comparison values are automatically coerced to the type of the expected value
when possible.

```python
msg.count = "10"

"{count > 5 ? ...}"       # "10" is coerced to int → True
"{enabled == true ? ...}" # string "true" is coerced to bool
```

If coercion fails, values are compared as strings.

### Logical Operators

| Operator | Description |
|----------|-------------|
| `!` | NOT (highest precedence) |
| `&` | AND |
| `\|\|` | OR (lowest precedence) |

Precedence order (highest to lowest): `!`, `&`, `||`. Use parentheses to make
intent explicit when combining operators.

```python
# ! binds tightest: reads as (active == true) AND (NOT banned == true)
"{active == true & !banned == true ? allow, deny}"

# Parentheses make precedence unambiguous
"{(plan == 'premium' & credits > 0) || trial == true ? access, block}"
```

---

## While Loops

Syntax: `@{condition}: pipeline;`

The body is any valid pipeline expression. The condition is re-evaluated against
the current message before each iteration.

???+ example

    ```python
    from msgflux import Inline, dotdict

    def init(msg):
        msg.counter = 0
        msg.total = 0

    def step(msg):
        msg.counter += 1
        msg.total += msg.counter     # accumulates: 1 + 2 + 3 + 4 + 5

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

`max_iterations` caps the number of times a while body may run. When the limit
is reached, `Inline` raises a `RuntimeError` instead of looping forever.

```python
# Default: raises after 1000 iterations
flux = Inline("@{active}: poll;", modules)

# Explicit retry budget — raises after 10 attempts
flux = Inline("@{active}: poll;", modules, max_iterations=10)
```

```
RuntimeError: While loop exceeded maximum iterations (10).
Possible infinite loop detected. Condition: active
```

!!! tip
    Set `max_iterations` to a small, intentional value when the loop represents
    a **retry budget**. This makes the limit explicit and surfaces bugs early
    instead of hanging for 1 000 iterations.

---

## Error Handling

If a module raises an exception, `Inline` wraps it and raises a `RuntimeError`
with the module name included.

In parallel stages, all branches run to completion before errors are reported,
so all failures are surfaced together.

```
RuntimeError: Parallel execution failed for:
  `feat_a`: ValueError('invalid input'),
  `feat_b`: TimeoutError('connection timed out')
```

For sequential steps, the error is raised immediately and execution stops.

```
RuntimeError: Execution failed for `enrich`: KeyError('text')
```

---

## Async Execution

Use `await flux.acall(msg)` when modules are async functions or `nn.Module`
instances with an `acall` method.

Sync modules inside an async pipeline are executed directly instead of being
offloaded to a thread pool, avoiding unnecessary scheduling overhead.

Parallel stages in async mode run each branch as a separate coroutine via
`ascatter_gather`.

???+ example

    ```python
    import asyncio
    from msgflux import Inline, dotdict

    async def fetch(msg):
        await asyncio.sleep(0)   # simulate I/O
        msg.data = "fetched"

    async def process(msg):
        msg.result = msg.data.upper()

    flux = Inline("fetch -> process", {"fetch": fetch, "process": process})

    msg = dotdict()
    asyncio.run(flux.acall(msg))

    print(msg.result)  # "FETCHED"
    ```

---

## Complex Workflow

All constructs compose freely. The expression below combines every DSL feature
in a single pipeline, reading from a message with `audio`, `confidence`, and
`is_urgent` fields.

```python
workflow = """
    ingest
    -> {msg.audio is not None? transcribe}
    -> [extract_entities, analyze_sentiment]
    -> @{msg.confidence < 0.8}: refine;
    -> {msg.is_urgent == true? priority_handler, standard_handler}
    -> finalize
"""
```

Execution order for `msg = dotdict(audio="call.wav", confidence=0.4, is_urgent=True)`:

1. `ingest` — always runs
2. `transcribe` — runs because `msg.audio` is not `None`
3. `extract_entities` and `analyze_sentiment` — run in parallel
4. `refine` — loops until `msg.confidence >= 0.8` (capped at 1 000 iterations)
5. `priority_handler` — runs because `msg.is_urgent == True`
6. `finalize` — always runs

---

## DSL Grammar

```
pipeline     ::= step ("->" step)*

step         ::= module
               | parallel
               | conditional
               | while_loop

module       ::= IDENTIFIER

parallel     ::= "[" module ("," module)* "]"

conditional  ::= "{" condition "?" branch ("," branch)? "}"

branch       ::= module ("," module)*

while_loop   ::= "@{" condition "}:" pipeline ";"

condition    ::= logical_expr

logical_expr ::= logical_or

logical_or   ::= logical_and ("||" logical_and)*

logical_and  ::= logical_not  ("&"  logical_not)*

logical_not  ::= "!" logical_not | primary

primary      ::= "(" logical_expr ")" | comparison

comparison   ::= key_path operator value
               | key_path "is" "None"
               | key_path "is" "not" "None"

key_path     ::= IDENTIFIER ("." IDENTIFIER)*

operator     ::= "==" | "!=" | "<" | ">" | "<=" | ">="
```
