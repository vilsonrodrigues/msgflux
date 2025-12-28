# Functional

The `msgflux.nn.functional` module provides a set of functions for concurrent task execution and message passing patterns. These utilities enable parallel processing, broadcasting, and asynchronous coordination.


## Overview

The Functional API offers both **synchronous** and **asynchronous** interfaces for:

- **Parallel mapping**: Apply a function to multiple inputs concurrently
- **Scatter-gather**: Distribute different tasks across inputs
- **Broadcast-gather**: Send the same input to multiple functions
- **Message passing**: Concurrent message processing with `dotdict`
- **Background tasks**: Fire-and-forget task execution
- **Event coordination**: Wait for async events in sync contexts

### Key Features

- **Concurrent execution** using thread pools and event loops
- **Zero-overhead** when executing single tasks
- **Error handling** with graceful degradation
- **Timeout support** for all operations
- **Message-first design** for workflow orchestration

## Pattern Comparison

```
MAP GATHER           SCATTER GATHER        BROADCAST GATHER
───────────          ──────────────        ────────────────
input1 ──┐           input1 ──┐            input ──┬──> f1 ──> r1
input2 ──┼─> f ──>   input2 ──┼─> f1 ──>           ├──> f2 ──> r2
input3 ──┘           input3 ──┘─> f2 ──>           └──> f3 ──> r3

Same function         Different functions   Multiple functions
Multiple inputs       Paired inputs/funcs   Same input
```

## Core Functions

### `map_gather()`

Apply the same function to multiple inputs concurrently.

**Pattern:**
```
input1 ──┐
input2 ──┼─> function ──> (result1, result2, result3)
input3 ──┘
```

**Usage:**

```python
import msgflux.nn.functional as F

# Example 1: Simple mapping
def square(x):
    return x * x

results = F.map_gather(square, args_list=[(2,), (3,), (4,)])
print(results)  # (4, 9, 16)

# Example 2: With multiple arguments
def add(x, y):
    return x + y

results = F.map_gather(add, args_list=[(1, 2), (3, 4), (5, 6)])
print(results)  # (3, 7, 11)

# Example 3: With kwargs
def multiply(x, y=2):
    return x * y

results = F.map_gather(
    multiply,
    args_list=[(1,), (3,), (5,)],
    kwargs_list=[{"y": 3}, {"y": 4}, {"y": 5}]
)
print(results)  # (3, 12, 25)
```

**Async version:** `amap_gather()`

### `scatter_gather()`

Distribute different functions across corresponding inputs.

**Pattern:**
```
input1 ──> function1 ──┐
input2 ──> function2 ──┼──> (result1, result2, result3)
input3 ──> function3 ──┘
```

**Usage:**

```python
import msgflux.nn.functional as F

def double(x):
    return x * 2

def triple(x):
    return x * 3

def square(x):
    return x ** 2

# Each function gets its corresponding input
results = F.scatter_gather(
    to_send=[double, triple, square],
    args_list=[(5,), (5,), (5,)]
)
print(results)  # (10, 15, 25)

# With different inputs
results = F.scatter_gather(
    to_send=[double, triple, square],
    args_list=[(2,), (3,), (4,)]
)
print(results)  # (4, 9, 16)
```

**Async version:** `ascatter_gather()`

### `bcast_gather()`

Broadcast the same arguments to multiple functions.

**Pattern:**
```
              ┌──> function1 ──> result1
input ────────┼──> function2 ──> result2
              └──> function3 ──> result3
```

**Usage:**

```python
import msgflux.nn.functional as F

def square(x):
    return x * x

def cube(x):
    return x * x * x

def double(x):
    return x * 2

# Same input to all functions
results = F.bcast_gather([square, cube, double], 5)
print(results)  # (25, 125, 10)

# With timeout
results = F.bcast_gather([square, cube, double], 3, timeout=1.0)
print(results)  # (9, 27, 6)

# Error handling - returns None for failed tasks
def fail(x):
    raise ValueError("Intentional error")

results = F.bcast_gather([square, fail, cube], 2)
print(results)  # (4, None, 8)
```

## Message-Based Functions

These functions work specifically with `msgflux.dotdict` for message passing patterns.

### `msg_scatter_gather()`

Scatter messages to functions and gather updated messages.

**Pattern:**
```
message1 ──> function1 ──┐
message2 ──> function2 ──┼──> (msg1, msg2, msg3)
message3 ──> function3 ──┘
```

**Usage:**

```python
import msgflux as mf
import msgflux.nn.functional as F

def process_user(msg):
    msg.type = "user"
    msg.processed = True
    return msg

def process_admin(msg):
    msg.type = "admin"
    msg.permissions = ["read", "write", "delete"]
    return msg

def process_guest(msg):
    msg.type = "guest"
    msg.permissions = ["read"]
    return msg

# Create messages
msg1 = mf.dotdict({"id": 1, "name": "Alice"})
msg2 = mf.dotdict({"id": 2, "name": "Bob"})
msg3 = mf.dotdict({"id": 3, "name": "Charlie"})

# Scatter to different processors
results = F.msg_scatter_gather(
    to_send=[process_user, process_admin, process_guest],
    messages=[msg1, msg2, msg3]
)

for msg in results:
    print(f"{msg.name}: {msg.type} - {msg.get('permissions', [])}")
# Alice: user - []
# Bob: admin - ['read', 'write', 'delete']
# Charlie: guest - ['read']
```

### `msg_bcast_gather()`

Broadcast a message to multiple modules for concurrent processing.

**Pattern:**
```
              ┌──> module1(msg) ──┐
message ──────┼──> module2(msg) ──┼──> message (modified)
              └──> module3(msg) ──┘
```

**Important:** Modules modify the message directly. Return values are ignored.

**Usage:**

```python
import msgflux as mf
import msgflux.nn.functional as F

def add_timestamp(msg):
    from datetime import datetime
    msg.timestamp = datetime.now().isoformat()
    return msg

def add_metadata(msg):
    msg.set("metadata.version", "1.0")
    msg.set("metadata.source", "api")
    return msg

def validate(msg):
    msg.validated = True
    return msg

# Broadcast message to all modules
message = mf.dotdict({"data": "important"})

F.msg_bcast_gather([add_timestamp, add_metadata, validate], message)

print(message.timestamp)  # 2024-01-15T10:30:00.123456
print(message.metadata.version)  # 1.0
print(message.validated)  # True
```

**Async version:** `amsg_bcast_gather()`

## Utility Functions

### `wait_for()`

Execute a single callable and wait for the result with optional timeout.

**Usage:**

```python
import msgflux.nn.functional as F
import time

def slow_computation(x):
    time.sleep(0.1)
    return x * x

# Simple execution
result = F.wait_for(slow_computation, 5)
print(result)  # 25

# With timeout
result = F.wait_for(slow_computation, 10, timeout=0.5)
print(result)  # 100

# Async function
async def async_task(x):
    return x * 2

result = F.wait_for(async_task, 3)
print(result)  # 6
```

### `wait_for_event()`

Wait for an asyncio event in a synchronous context.

**Usage:**

```python
import msgflux.nn.functional as F
import asyncio

# Create an event
event = asyncio.Event()

# Set the event from another thread/task
def set_event():
    event.set()

# Wait for event in sync code
import threading
thread = threading.Thread(target=lambda: (time.sleep(0.1), set_event()))
thread.start()

F.wait_for_event(event)  # Blocks until event is set
print("Event was set!")
```

**Async version:** `await_for_event()`

### `background_task()`

Execute a function in the background without waiting for the result.

**Usage:**

```python
import msgflux.nn.functional as F

def log_event(event_type, user_id):
    # This runs in the background
    print(f"Logging: {event_type} for user {user_id}")

# Fire and forget
F.background_task(log_event, "login", 12345)
# Execution continues immediately

# Background task runs concurrently
print("Main thread continues...")

# With kwargs
F.background_task(log_event, event_type="logout", user_id=67890)
```

**Async version:** `abackground_task()`

## Async Equivalents

All major functions have async equivalents prefixed with `a`:

| Sync Function | Async Equivalent |
|---------------|------------------|
| `map_gather` | `amap_gather` |
| `scatter_gather` | `ascatter_gather` |
| `msg_bcast_gather` | `amsg_bcast_gather` |
| `wait_for_event` | `await_for_event` |
| `background_task` | `abackground_task` |

**Usage:**

```python
import msgflux.nn.functional as F
import asyncio

async def main():
    # Async map gather
    async def async_square(x):
        await asyncio.sleep(0.01)
        return x * x

    results = await F.amap_gather(
        async_square,
        args_list=[(2,), (3,), (4,)]
    )
    print(results)  # (4, 9, 16)

    # Async broadcast
    async def async_double(x):
        await asyncio.sleep(0.01)
        return x * 2

    async def async_triple(x):
        await asyncio.sleep(0.01)
        return x * 3

    results = await F.ascatter_gather(
        [async_double, async_triple],
        [(5,), (5,)]
    )
    print(results)  # (10, 15)

# Run async code
asyncio.run(main())
```

## Best Practices

### 1. Choose the Right Pattern

```python
# Use map_gather when: Same function, different inputs
results = F.map_gather(process, args_list=[(1,), (2,), (3,)])

# Use scatter_gather when: Different functions, different inputs
results = F.scatter_gather([f1, f2, f3], args_list=[(a,), (b,), (c,)])

# Use bcast_gather when: Multiple functions, same input
results = F.bcast_gather([f1, f2, f3], input)
```

### 2. Handle Errors Gracefully

```python
# Functions return None on error
def safe_divide(x, y):
    return x / y

results = F.map_gather(
    safe_divide,
    args_list=[(10, 2), (10, 0), (10, 5)]
)
print(results)  # (5.0, None, 2.0)

# Check for errors
for i, result in enumerate(results):
    if result is None:
        print(f"Task {i} failed")
```

### 3. Use Timeouts for Long Operations

```python
# Prevent indefinite blocking
results = F.bcast_gather(
    [slow_task1, slow_task2],
    input_data,
    timeout=5.0  # 5 second timeout
)
```

### 4. Message Modifications in Parallel

```python
# Good - Modify different message paths
def add_user_data(msg):
    msg.set("user.name", "Alice")
    return msg

def add_timestamp(msg):
    msg.set("meta.timestamp", "2024-01-15")
    return msg

# Both can run in parallel safely
F.msg_bcast_gather([add_user_data, add_timestamp], message)
```

### 5. Background Tasks for Side Effects

```python
# Good use cases for background tasks:
# - Logging
# - Metrics collection
# - Cache updates
# - Notifications

F.background_task(log_to_file, "User logged in", user_id=123)
F.background_task(update_cache, key="user:123", value=user_data)
F.background_task(send_notification, user_id=123, message="Welcome!")
```

## Common Patterns

### Pipeline with Parallel Stages

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

# Sequential then parallel
message = mf.dotdict()
prepare(message)
F.msg_bcast_gather([filter_even, filter_odd], message)

print(message.results.even)  # [2, 4]
print(message.results.odd)   # [1, 3, 5]
```

### Parallel Data Processing

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

```python
import msgflux.nn.functional as F

def fetch_user(user_id):
    # Simulate API call
    return {"id": user_id, "name": f"User {user_id}"}

def fetch_posts(user_id):
    # Simulate API call
    return [f"Post {i}" for i in range(3)]

def fetch_comments(user_id):
    # Simulate API call
    return [f"Comment {i}" for i in range(5)]

# Fetch all data for a user in parallel
user_data, posts, comments = F.bcast_gather(
    [fetch_user, fetch_posts, fetch_comments],
    user_id=123
)

print(f"User: {user_data}")
print(f"Posts: {len(posts)}")
print(f"Comments: {len(comments)}")
```

## Performance Considerations

### When to Use Parallel Execution

**✅ Good candidates for parallelization:**
- I/O-bound operations (API calls, file I/O, database queries)
- Independent computations
- Multiple data transformations
- Batch processing

**❌ Poor candidates:**
- CPU-bound operations (use `multiprocessing` instead)
- Very fast operations (overhead > benefit)
- Operations with shared mutable state
- Sequential dependencies

### Overhead vs Benefit

```python
import msgflux.nn.functional as F
import time

# Fast operation - parallel overhead might not be worth it
def fast_op(x):
    return x * 2

# Slow operation - benefits from parallelization
def slow_op(x):
    time.sleep(0.1)
    return x * 2

# For fast operations, sequential might be faster
start = time.time()
results = [fast_op(i) for i in range(100)]
print(f"Sequential: {time.time() - start:.4f}s")

start = time.time()
results = F.map_gather(fast_op, args_list=[(i,) for i in range(100)])
print(f"Parallel: {time.time() - start:.4f}s")

# For slow operations, parallel is much faster
start = time.time()
results = [slow_op(i) for i in range(10)]
print(f"Sequential: {time.time() - start:.4f}s")

start = time.time()
results = F.map_gather(slow_op, args_list=[(i,) for i in range(10)])
print(f"Parallel: {time.time() - start:.4f}s")
```

## API Reference

For complete API documentation with all parameters and return types, see:

::: msgflux.nn.functional.map_gather

::: msgflux.nn.functional.scatter_gather

::: msgflux.nn.functional.msg_scatter_gather

::: msgflux.nn.functional.bcast_gather

::: msgflux.nn.functional.msg_bcast_gather

::: msgflux.nn.functional.wait_for

::: msgflux.nn.functional.wait_for_event

::: msgflux.nn.functional.background_task

::: msgflux.nn.functional.amap_gather

::: msgflux.nn.functional.ascatter_gather

::: msgflux.nn.functional.amsg_bcast_gather

::: msgflux.nn.functional.await_for_event

::: msgflux.nn.functional.abackground_task
