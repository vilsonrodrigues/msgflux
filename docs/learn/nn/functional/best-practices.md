# Best Practices

## Choose the Right Pattern

```python
# Use map_gather: Same function, different inputs
results = F.map_gather(process, args_list=[(1,), (2,), (3,)])

# Use scatter_gather: Different functions, different inputs
results = F.scatter_gather([f1, f2, f3], args_list=[(a,), (b,), (c,)])

# Use bcast_gather: Multiple functions, same input
results = F.bcast_gather([f1, f2, f3], input)
```

## Handle Errors Gracefully

```python
from msgflux import TaskError

# Failed tasks return a TaskError instance
results = F.map_gather(
    divide,
    args_list=[(10, 2), (10, 0), (10, 5)]
)
print(results)  # (5.0, TaskError(index=1, ...), 2.0)

# Check for errors — TaskError is falsy, so `if not result` works
for i, result in enumerate(results):
    if isinstance(result, TaskError):
        print(f"Task {i} failed: {result.exception}")
    else:
        print(f"Task {i}: {result}")

# Filter successes and errors
successes = [r for r in results if not isinstance(r, TaskError)]
errors = [r for r in results if isinstance(r, TaskError)]
```

## Use Timeouts in Production

```python
# Prevent indefinite blocking
results = F.bcast_gather(
    [slow_task1, slow_task2],
    input_data,
    timeout=5.0
)
```

## Modify Different Message Paths

```python
# Good - Different paths, safe for parallel
def add_user_data(msg):
    msg.set("user.name", "Alice")
    return msg

def add_timestamp(msg):
    msg.set("meta.timestamp", "2024-01-15")
    return msg

F.msg_bcast_gather([add_user_data, add_timestamp], message)
```

## Common Patterns

### Pipeline with Parallel Stages

???+ example

    ```python
    import msgflux as mf
    import msgflux.nn.functional as F

    def prepare(msg):
        msg.data = [1, 2, 3, 4, 5]
        return msg

    def filter_even(msg):
        msg.set("results.even", [x for x in msg.data if x % 2 == 0])
        return msg

    def filter_odd(msg):
        msg.set("results.odd", [x for x in msg.data if x % 2 != 0])
        return msg

    message = mf.dotdict()
    prepare(message)
    F.msg_bcast_gather([filter_even, filter_odd], message)

    print(message.results.even)  # [2, 4]
    print(message.results.odd)   # [1, 3, 5]
    ```

### Parallel Data Processing

???+ example

    ```python
    import msgflux.nn.functional as F

    def process_chunk(data):
        return sum(data)

    # Split data into chunks
    data = list(range(1000))
    chunk_size = 100
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

    # Process chunks in parallel
    results = F.map_gather(
        process_chunk,
        args_list=[(chunk,) for chunk in chunks]
    )

    total = sum(results)
    print(total)  # 499500
    ```

### Concurrent API Calls

???+ example

    ```python
    import msgflux.nn.functional as F

    def fetch_user(user_id):
        return {"id": user_id, "name": f"User {user_id}"}

    def fetch_posts(user_id):
        return [f"Post {i}" for i in range(3)]

    def fetch_comments(user_id):
        return [f"Comment {i}" for i in range(5)]

    # Fetch all data for a user in parallel
    user_data, posts, comments = F.bcast_gather(
        [fetch_user, fetch_posts, fetch_comments],
        user_id=123
    )

    print(f"User: {user_data['name']}")
    print(f"Posts: {len(posts)}")
    print(f"Comments: {len(comments)}")
    ```

## Performance Considerations

### When to Use Parallel Execution

**Good candidates for parallelization:**

- I/O-bound operations (API calls, file I/O, database queries)
- Independent computations
- Multiple data transformations
- Batch processing

**Poor candidates:**

- CPU-bound operations (use `multiprocessing` instead)
- Very fast operations (overhead > benefit)
- Operations with shared mutable state
- Sequential dependencies

### Overhead vs Benefit

???+ example

    ```python
    import msgflux.nn.functional as F
    import time

    def fast_op(x):
        return x * 2

    def slow_op(x):
        time.sleep(0.1)
        return x * 2

    # Fast operations: sequential might be faster
    start = time.time()
    results = [fast_op(i) for i in range(100)]
    print(f"Sequential (fast): {time.time() - start:.4f}s")

    start = time.time()
    results = F.map_gather(fast_op, args_list=[(i,) for i in range(100)])
    print(f"Parallel (fast): {time.time() - start:.4f}s")

    # Slow operations: parallel is much faster
    start = time.time()
    results = [slow_op(i) for i in range(10)]
    print(f"Sequential (slow): {time.time() - start:.4f}s")  # ~1.0s

    start = time.time()
    results = F.map_gather(slow_op, args_list=[(i,) for i in range(10)])
    print(f"Parallel (slow): {time.time() - start:.4f}s")    # ~0.1s
    ```
