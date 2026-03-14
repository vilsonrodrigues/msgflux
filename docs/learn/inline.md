# Inline

`Inline` orchestrates modules with a declarative workflow string. The workflow
is configured once in `__init__`, and execution happens through `__call__` or
`acall`.

```python
from msgflux import Inline
```

## Syntax Overview

| Pattern | Description | Example |
|---------|-------------|---------|
| `->` | Sequential execution | `"a -> b -> c"` |
| `[...]` | Parallel execution | `"a -> [b, c] -> d"` |
| `{cond? a}` | Conditional (if) | `"{has_data? process}"` |
| `{cond? a, b}` | Conditional (if-else) | `"{is_premium? vip, standard}"` |
| `@{cond}: a;` | While loop | `"@{count < 5}: increment;"` |

## Sequential Execution

???+ example

    ```python
    from msgflux import dotdict, Inline

    def step1(msg):
        msg.step1 = "done"

    def step2(msg):
        msg.step2 = "done"

    def step3(msg):
        msg.step3 = "done"

    modules = {"step1": step1, "step2": step2, "step3": step3}
    message = dotdict()

    Inline("step1 -> step2 -> step3", modules)(message)

    print(message.step1)  # "done"
    print(message.step2)  # "done"
    print(message.step3)  # "done"
    ```

## Parallel Execution

Parallel stages mutate the same message in place. Keep each module writing to a
different path.

???+ example

    ```python
    from msgflux import dotdict, Inline

    def prep(msg):
        msg.base = 10

    def feat_a(msg):
        msg.features = msg.get("features", {})
        msg.features["a"] = msg.base + 1

    def feat_b(msg):
        msg.features = msg.get("features", {})
        msg.features["b"] = msg.base + 2

    def combine(msg):
        msg.total = msg.features["a"] + msg.features["b"]

    modules = {
        "prep": prep,
        "feat_a": feat_a,
        "feat_b": feat_b,
        "combine": combine,
    }

    message = dotdict()
    Inline("prep -> [feat_a, feat_b] -> combine", modules)(message)

    print(message.total)  # 23
    ```

!!! warning "Race Conditions"
    Parallel modules should not write to the same message path concurrently.
    `Inline` validates `TaskError` results, but it does not serialize writes.

## Conditionals

???+ example "If / Else"

    ```python
    from msgflux import dotdict, Inline

    def adult_flow(msg):
        msg.result = "Welcome, adult"

    def child_flow(msg):
        msg.result = "Hi, young one"

    modules = {"adult_flow": adult_flow, "child_flow": child_flow}
    message = dotdict()
    message.set("user.age", 21)

    Inline("{user.age > 18 ? adult_flow, child_flow}", modules)(message)
    print(message.result)  # "Welcome, adult"
    ```

## Logical Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `&` | AND | `"is_admin & is_active"` |
| `\|\|` | OR | `"is_premium \|\| has_coupon"` |
| `!` | NOT | `"!is_banned"` |

## None Verification

```python
"{user.name is None ? ask_name, greet}"
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

## While Loops

???+ example

    ```python
    from msgflux import dotdict, Inline

    async def prep(msg):
        msg.counter = 0

    async def increment(msg):
        msg.counter += 1

    async def finalize(msg):
        msg.done = True

    modules = {
        "prep": prep,
        "increment": increment,
        "finalize": finalize,
    }

    message = dotdict()
    await Inline(
        "prep -> @{counter < 5}: increment; -> finalize",
        modules,
    ).acall(message)

    print(message.counter)  # 5
    print(message.done)     # True
    ```

!!! warning "Infinite Loops"
    While loops enforce `max_iterations` to avoid infinite execution.

## With `nn.Module`

???+ example

    ```python
    import msgflux as mf
    import msgflux.nn as nn
    from msgflux import Inline

    class Pipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.transcriber = nn.Transcriber(...)
            self.extractor = nn.Agent(...)
            self.components = nn.ModuleDict({
                "transcriber": self.transcriber,
                "extractor": self.extractor,
            })
            self.register_buffer(
                "flux",
                "{user_audio is not None? transcriber} -> extractor",
            )

        def forward(self, msg):
            return Inline(self.flux, self.components)(msg)

        async def aforward(self, msg):
            return await Inline(self.flux, self.components).acall(msg)
    ```

## Complex Workflow Example

???+ example

    ```python
    from msgflux import Inline

    workflow = """
        prep
        -> {has_audio is not None? transcribe}
        -> [analyze_sentiment, extract_entities]
        -> @{confidence < 0.8}: refine;
        -> {is_urgent == true? priority_handler, standard_handler}
        -> finalize
    """

    Inline(workflow, modules)(message)
    ```
