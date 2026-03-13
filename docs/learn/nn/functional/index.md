# nn.functional

The `msgflux.nn.functional` module provides concurrent execution primitives. For workflow orchestration, use the `Inline` runtime from `msgflux.dsl.inline`.

## Overview

The functional API offers both **synchronous** and **asynchronous** interfaces for concurrent task execution, inspired by MPI scatter-gather patterns and PyTorch's functional API.

### Key Features

- **Concurrent Execution**: Thread pools and async event loops for parallel processing
- **Gather Patterns**: Map, scatter, and broadcast primitives for different use cases
- **Message Passing**: `dotdict` works directly with the generic gather helpers
- **Inline DSL**: `Inline` provides declarative workflow orchestration
- **Zero Overhead**: No performance penalty for single-task execution
- **Error Handling**: Typed `TaskError` results for failed tasks — no silent `None`


## Quick Start

???+ example "Parallel Execution"

    ```python
    import msgflux.nn.functional as F

    def process(x):
        return x * 2

    # Run process(1), process(2), process(3) in parallel
    results = F.map_gather(process, args_list=[(1,), (2,), (3,)])
    print(results)  # (2, 4, 6)
    ```

???+ example "Workflow DSL"

    ```python
    from msgflux import dotdict
    from msgflux.dsl.inline import Inline

    def step1(msg):
        msg.data = "processed"

    def step2(msg):
        msg.result = f"{msg.data} -> done"

    modules = {"step1": step1, "step2": step2}
    message = dotdict()

    Inline("step1 -> step2", modules)(message)
    print(message.result)  # "processed -> done"
    ```

## Pattern Comparison

The three gather patterns serve different use cases:

```
MAP GATHER              SCATTER GATHER          BROADCAST GATHER
──────────────          ──────────────          ────────────────
input1 ──┐              input1 ──> f1 ──┐               ┌──> f1 ──> r1
input2 ──┼──> f ──>     input2 ──> f2 ──┼──>    input ──├──> f2 ──> r2
input3 ──┘              input3 ──> f3 ──┘               └──> f3 ──> r3

Same function           Different functions     Multiple functions
Multiple inputs         Paired inputs/funcs     Same input
```

| Pattern | When to Use |
|---------|-------------|
| `map_gather` | Apply the same function to multiple inputs |
| `scatter_gather` | Route different inputs to different functions |
| `bcast_gather` | Fan-out one input to multiple functions |

## Async Equivalents

All core functions have async counterparts prefixed with `a`:

| Sync Function | Async Equivalent |
|---------------|------------------|
| `map_gather` | `amap_gather` |
| `scatter_gather` | `ascatter_gather` |
| `wait_for_event` | `await_for_event` |
| `fire_and_forget` | `afire_and_forget` |

???+ example "Async Usage"

    ```python
    import msgflux.nn.functional as F
    import asyncio

    async def main():
        async def async_square(x):
            await asyncio.sleep(0.01)
            return x * x

        results = await F.amap_gather(
            async_square,
            args_list=[(2,), (3,), (4,)]
        )
        print(results)  # (4, 9, 16)

    asyncio.run(main())
    ```

## Contents

| Topic | Description |
|-------|-------------|
| [Gather Functions](gather.md) | map_gather, scatter_gather and bcast_gather |
| [Utility Functions](utility.md) | wait_for, wait_for_event, fire_and_forget |
| [Inline DSL](inline-dsl.md) | Declarative workflow language with `Inline` |
| [Best Practices](best-practices.md) | Patterns, performance, and error handling |
