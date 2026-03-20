# Durable Gather

Durable gather adds **Erlang-style one-for-one supervision** to all gather functions. When a worker fails, only that worker is retried with exponential backoff — other workers continue unaffected.

## Overview

| Concept | Description |
|---------|-------------|
| **Activation** | Pass `max_retries` to any gather function. `None` (default) uses the original fast path. |
| **Supervision** | Each worker is monitored independently (one-for-one). |
| **Backoff** | Exponential delay with jitter: `base * 2^(attempt-1) + random(0, 30%)`. |
| **Checkpoint** | Optional `store` parameter persists partial results for crash recovery. |
| **Zero overhead** | When `max_retries` is `None` and `store` is `None`, the exact original code path runs. |

```
Worker 0 ──────── ok ──────────────────────► result[0]
Worker 1 ── fail ── retry ── fail ── retry ─► result[1]  (recovered)
Worker 2 ── fail ── retry ── retry ── fail ─► TaskError   (exhausted)
Worker 3 ──────── ok ──────────────────────► result[3]
```

## Quick Start

### Retry on failure

```python
import msgflux.nn.functional as F

def flaky_api(query):
    """Simulates an API that sometimes fails."""
    import random
    if random.random() < 0.3:
        raise ConnectionError("timeout")
    return f"result for {query}"

results = F.map_gather(
    flaky_api,
    args_list=[("query1",), ("query2",), ("query3",)],
    max_retries=3,
    retry_delay=1.0,
)
# Each failed call retries up to 3 times with exponential backoff
```

### Checkpoint and resume

```python
import msgflux as mf
import msgflux.nn.functional as F

store = mf.InMemoryCheckpointStore()

results = F.scatter_gather(
    [fetch_users, fetch_posts, fetch_comments],
    args_list=[("team_a",), ("team_a",), ("team_a",)],
    max_retries=2,
    store=store,
    namespace="data_pipeline",
    session_id="daily_sync",
    run_id="run_001",
)

# If the process crashes mid-execution, completed workers are saved.
# Re-running with the same store/namespace/session_id/run_id resumes
# from where it left off — only pending workers are re-executed.
```

## Supported Functions

All gather functions accept `max_retries` and `retry_delay`:

| Function | `max_retries` | `retry_delay` | `store` | Notes |
|----------|:---:|:---:|:---:|-------|
| `scatter_gather` | yes | yes | yes | Full durable support |
| `ascatter_gather` | yes | yes | yes | Async equivalent |
| `map_gather` | yes | yes | yes | Full durable support |
| `amap_gather` | yes | yes | yes | Async equivalent |
| `bcast_gather` | yes | yes | no | No store (uses `**kwargs`) |
| `wait_for` | yes | yes | no | Single task, inline retry |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_retries` | `int \| None` | `None` | Max retry attempts per worker. `None` disables durable mode. |
| `retry_delay` | `float` | `1.0` | Base delay in seconds for exponential backoff. |
| `store` | `CheckpointStore \| None` | `None` | Checkpoint store for crash recovery. |
| `namespace` | `str` | `"gather"` | Namespace for store partitioning. |
| `session_id` | `str` | `"default"` | Session identifier for store partitioning. |
| `run_id` | `str \| None` | `None` | Run identifier. Auto-generated UUID if not provided. |

## Retry Behavior

### Exponential backoff with jitter

Each retry waits longer than the previous one, with random jitter to prevent thundering herd:

```
Attempt 1: 1.0s  + jitter (0-30%)  → 1.0 - 1.3s
Attempt 2: 2.0s  + jitter (0-30%)  → 2.0 - 2.6s
Attempt 3: 4.0s  + jitter (0-30%)  → 4.0 - 5.2s
Attempt 4: 8.0s  + jitter (0-30%)  → 8.0 - 10.4s
```

### Exhausted retries

When a worker exhausts all retries, it returns a `TaskError` instead of raising. Other workers are unaffected.

???+ example "Handling mixed results"

    ```python
    import msgflux.nn.functional as F
    from msgflux.exceptions import TaskError

    def reliable(x):
        return x * 2

    def unreliable(x):
        raise RuntimeError("permanent failure")

    results = F.scatter_gather(
        [reliable, unreliable, reliable],
        args_list=[(10,), (20,), (30,)],
        max_retries=2,
        retry_delay=0.1,
    )

    print(results[0])  # 20
    print(results[1])  # TaskError(index=1, exception=RuntimeError(...))
    print(results[2])  # 60

    # Filter successes
    successes = [r for r in results if not isinstance(r, TaskError)]
    errors = [r for r in results if isinstance(r, TaskError)]
    ```

## Checkpoint Store

When a `store` is provided, partial results are persisted after each worker completes. This enables crash recovery — if the process dies, re-running with the same keys resumes from the last checkpoint.

### How it works

```
Start
  │
  ├─ Check store for existing run
  │    ├─ Found incomplete → load partial results, skip completed workers
  │    └─ Not found → start fresh
  │
  ├─ Submit pending workers
  │
  ├─ Monitor loop (FIRST_COMPLETED)
  │    ├─ Worker completes → save result to store
  │    └─ Worker fails → retry or mark as TaskError, save to store
  │
  └─ All done → save final state (status="completed")
```

### Resume from crash

???+ example "Crash recovery"

    ```python
    import msgflux as mf
    import msgflux.nn.functional as F

    store = mf.SQLiteCheckpointStore(path=".msgflux/checkpoints.sqlite3")

    # Run 1: workers 0 and 1 complete, process crashes before worker 2
    try:
        results = F.scatter_gather(
            [fast_task, fast_task, slow_task],
            args_list=[(1,), (2,), (3,)],
            max_retries=0,
            store=store,
            namespace="pipeline",
            session_id="batch_1",
            run_id="run_001",
        )
    except Exception:
        print("Process crashed, but workers 0 and 1 are saved")

    # Run 2: workers 0 and 1 are loaded from store, only worker 2 re-runs
    results = F.scatter_gather(
        [fast_task, fast_task, slow_task],
        args_list=[(1,), (2,), (3,)],
        max_retries=0,
        store=store,
        namespace="pipeline",
        session_id="batch_1",
        run_id="run_001",
    )
    ```

### Auto-resume without run_id

When `run_id` is not provided, the system checks for incomplete runs via `store.find_incomplete_runs()` and resumes the first one found:

```python
# No run_id — auto-detects and resumes any incomplete run
results = F.scatter_gather(
    [task_a, task_b, task_c],
    args_list=[(1,), (2,), (3,)],
    max_retries=0,
    store=store,
    namespace="pipeline",
    session_id="batch_1",
)
```

### Checkpoint state format

Each checkpoint is a full snapshot:

```python
{
    "status": "running",       # or "completed"
    "results": [42, {"__pending__": True}, {"__task_error__": ...}],
    "total": 3,
}
```

| Status | Meaning |
|--------|---------|
| `running` | Execution in progress (saved after each worker completes) |
| `completed` | All workers finished |

## Examples

### Parallel LLM calls with retry

???+ example "Multiple summarizers"

    ```python
    import msgflux as mf
    import msgflux.nn as nn
    import msgflux.nn.functional as F

    mf.load_dotenv()

    groq = mf.Model.chat_completion("groq/llama-3.3-70b-versatile")

    summarizer = nn.Agent(
        "summarizer",
        groq,
        system_message="Summarize the topic in one sentence in Portuguese.",
    )

    topics = [
        ("Fotossíntese",),
        ("Buracos negros",),
        ("Computação quântica",),
    ]

    results = F.map_gather(
        summarizer,
        args_list=topics,
        max_retries=3,
        retry_delay=1.0,
    )

    for topic, result in zip(topics, results):
        print(f"{topic[0]}: {result}")
    ```

### Multi-provider comparison

???+ example "Same question to Groq and OpenAI"

    ```python
    import msgflux as mf
    import msgflux.nn as nn
    import msgflux.nn.functional as F

    mf.load_dotenv()

    groq_agent = nn.Agent(
        "groq",
        mf.Model.chat_completion("groq/llama-3.3-70b-versatile"),
        system_message="Answer in one sentence.",
    )
    openai_agent = nn.Agent(
        "openai",
        mf.Model.chat_completion("openai/gpt-4.1-nano"),
        system_message="Answer in one sentence.",
    )

    results = F.bcast_gather(
        [groq_agent, openai_agent],
        "What is quantum computing?",
        max_retries=2,
        retry_delay=1.0,
    )

    print(f"Groq:   {results[0]}")
    print(f"OpenAI: {results[1]}")
    ```

### Parallel translation with checkpoint

???+ example "Fan-out translation"

    ```python
    import msgflux as mf
    import msgflux.nn as nn
    import msgflux.nn.functional as F

    mf.load_dotenv()
    model = mf.Model.chat_completion("openai/gpt-4.1-nano")
    store = mf.InMemoryCheckpointStore()

    def make_translator(lang):
        return nn.Agent(
            f"translator_{lang}",
            model,
            system_message=f"Translate the text to {lang}. Reply only with the translation.",
        )

    translators = [
        make_translator("English"),
        make_translator("French"),
        make_translator("Spanish"),
    ]
    text = "A inteligência artificial está transformando o mundo."

    results = F.scatter_gather(
        translators,
        args_list=[(text,)] * 3,
        max_retries=1,
        store=store,
        namespace="translation",
        session_id="demo",
        run_id="translate_1",
    )

    for lang, result in zip(["EN", "FR", "ES"], results):
        print(f"[{lang}] {result}")

    # State is persisted — re-running skips completed translations
    state = store.load_state("translation", "demo", "translate_1")
    print(f"Status: {state['status']}")  # "completed"
    ```

### Single task retry with wait_for

???+ example "Retry a single function"

    ```python
    import msgflux.nn.functional as F
    from msgflux.exceptions import TaskError

    def flaky_api_call(query):
        import random
        if random.random() < 0.5:
            raise ConnectionError("timeout")
        return f"result: {query}"

    result = F.wait_for(
        flaky_api_call, "important query",
        max_retries=5,
        retry_delay=0.5,
    )

    if isinstance(result, TaskError):
        print(f"All retries exhausted: {result.exception}")
    else:
        print(f"Success: {result}")
    ```

## Async Support

All durable features work with async functions. The async path uses `asyncio.Task` per worker, each with its own retry loop.

???+ example "Async durable gather"

    ```python
    import asyncio
    import msgflux.nn.functional as F

    async def fetch_data(url):
        await asyncio.sleep(0.1)  # simulate I/O
        return f"data from {url}"

    async def main():
        results = await F.ascatter_gather(
            [fetch_data, fetch_data, fetch_data],
            args_list=[
                ("https://api.example.com/users",),
                ("https://api.example.com/posts",),
                ("https://api.example.com/comments",),
            ],
            max_retries=3,
            retry_delay=1.0,
        )
        print(results)

    asyncio.run(main())
    ```

Sync callables also work in async gather — they are automatically dispatched via `run_in_executor`.

## Architecture

The durable gather implementation lives in `msgflux._private.supervision` and provides two core functions:

- `gather_durable_sync` — Thread pool with `concurrent.futures.wait(FIRST_COMPLETED)` monitoring
- `gather_durable_async` — `asyncio.Task` per worker with `asyncio.wait(FIRST_COMPLETED)` monitoring

Both follow the same one-for-one supervision pattern:

1. **Resume**: Check store for partial results, skip completed workers
2. **Submit**: Launch pending workers concurrently
3. **Monitor**: Wait for the first completion, handle result or retry
4. **Persist**: Save state to store after each worker completes
5. **Finalize**: Mark run as `completed` when all workers finish

!!! note "Erlang inspiration"
    In Erlang/OTP, a **one-for-one supervisor** monitors child processes independently. If one crashes, only that one is restarted. This is the model used here — each worker is an independent unit of supervision.
