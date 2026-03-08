# Inline DSL

The `inline` function allows you to orchestrate modules using a **declarative workflow language**. Define complex workflows as strings that can be modified at runtime without changing code.

## Syntax Overview

| Pattern | Description | Example |
|---------|-------------|---------|
| `->` | Sequential execution | `"a -> b -> c"` |
| `[...]` | Parallel execution | `"a -> [b, c] -> d"` |
| `{cond? a}` | Conditional (if) | `"{has_data? process}"` |
| `{cond? a, b}` | Conditional (if-else) | `"{is_premium? vip, standard}"` |
| `@{cond}: a;` | While loop | `"@{count < 5}: increment;"` |

## Sequential Execution

Use `->` to chain modules in order:

???+ example

    ```python
    import msgflux.nn.functional as F
    from msgflux import dotdict

    def step1(msg):
        msg.step1 = "done"
        return msg

    def step2(msg):
        msg.step2 = "done"
        return msg

    def step3(msg):
        msg.step3 = "done"
        return msg

    modules = {"step1": step1, "step2": step2, "step3": step3}
    message = dotdict()

    F.inline("step1 -> step2 -> step3", modules, message)

    print(message.step1)  # "done"
    print(message.step2)  # "done"
    print(message.step3)  # "done"
    ```

## Parallel Execution

Use `[...]` to run modules concurrently:

???+ example

    ```python
    import msgflux.nn.functional as F
    from msgflux import dotdict

    def fetch_a(msg):
        return {"data": "result_a"}

    def fetch_b(msg):
        return {"data": "result_b"}

    def combine(msg):
        msg.combined = f"{msg.fetch_a.data} + {msg.fetch_b.data}"
        return msg

    modules = {
        "prep": lambda msg: msg,
        "fetch_a": fetch_a,
        "fetch_b": fetch_b,
        "combine": combine
    }

    message = dotdict()
    F.inline("prep -> [fetch_a, fetch_b] -> combine", modules, message)

    print(message.combined)  # "result_a + result_b"
    ```

!!! warning "Race Conditions"
    In parallel execution, do **not modify** the message directly inside functions—return values instead. Results are automatically saved as `msg.set("module_name", response)`.

## Conditionals

### If

Execute a module only if the condition is true:

???+ example

    ```python
    import msgflux.nn.functional as F
    from msgflux import dotdict

    def transcribe(msg):
        msg.transcription = "Hello world"
        return msg

    modules = {"transcribe": transcribe, "process": lambda msg: msg}

    message = dotdict()
    message.audio = "audio.mp3"

    F.inline("{audio is not None? transcribe} -> process", modules, message)

    print(message.transcription)  # "Hello world"
    ```

### If-Else

Execute one branch or the other:

???+ example

    ```python
    import msgflux.nn.functional as F
    from msgflux import dotdict

    def adult_flow(msg):
        msg.result = "Welcome, adult"
        return msg

    def child_flow(msg):
        msg.result = "Hi, young one"
        return msg

    modules = {"adult_flow": adult_flow, "child_flow": child_flow}

    message = dotdict()
    message.set("user.age", 21)

    F.inline("{user.age > 18 ? adult_flow, child_flow}", modules, message)
    print(message.result)  # "Welcome, adult"
    ```

## Logical Operators

Combine conditions with logical operators:

| Operator | Description | Example |
|----------|-------------|---------|
| `&` | AND | `"is_admin & is_active"` |
| `\|\|` | OR | `"is_premium \|\| has_coupon"` |
| `!` | NOT | `"!is_banned"` |

???+ example

    ```python
    import msgflux.nn.functional as F
    from msgflux import dotdict

    modules = {"grant": lambda m: m.update(access=True) or m,
               "deny": lambda m: m.update(access=False) or m}

    message = dotdict()
    message.set("user.is_active", True)
    message.set("user.is_banned", False)

    F.inline(
        "{user.is_active == True & !user.is_banned == True ? grant, deny}",
        modules,
        message
    )

    print(message.access)  # True
    ```

## None Verification

Check if a field is None or not None:

```python
# Check if None
"{user.name is None ? ask_name, greet}"

# Check if not None
"{user.audio is not None ? transcribe}"
```

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

???+ example

    ```python
    "{score >= 0.9 ? high_quality, review}"
    "{status != 'completed' ? process}"
    "{items.0.price > 100 ? expensive}"
    ```

## While Loops

Execute repeatedly while condition is true:

???+ example

    ```python
    import msgflux.nn.functional as F
    from msgflux import dotdict

    async def increment(msg):
        msg.counter = msg.get("counter", 0) + 1
        return msg

    async def finalize(msg):
        msg.done = True
        return msg

    modules = {
        "prep": lambda msg: msg.update(counter=0) or msg,
        "increment": increment,
        "finalize": finalize
    }

    message = dotdict()
    result = await F.ainline(
        "prep -> @{counter < 5}: increment; -> finalize",
        modules,
        message
    )

    print(message.counter)  # 5
    print(message.done)     # True
    ```

!!! warning "Infinite Loops"
    While loops have a maximum iteration limit to prevent infinite loops. A `RuntimeError` is raised if the limit is exceeded.

## Message Access

The DSL accesses message fields using dot notation:

???+ example

    ```python
    from msgflux import dotdict

    message = dotdict()
    message.set("user.age", 25)
    message.set("config.is_premium", True)

    # Access in conditions
    "{user.age > 18 ? adult}"
    "{config.is_premium == true ? vip}"
    ```

## With nn.Module

Store workflows as buffers in custom modules:

???+ example

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

## Complex Workflow Example

???+ example

    ```python
    import msgflux.nn.functional as F

    # Full workflow combining all patterns
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

**Async version:** `ainline`
