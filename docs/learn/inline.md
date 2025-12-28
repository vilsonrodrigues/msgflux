# inline

The `inline` function provides a simple, declarative language for orchestrating module workflows at runtime. Define complex pipelines using intuitive string syntax without writing explicit Python code for control flow.

## Overview

`inline` allows you to:

- **Define workflows declaratively** using string syntax
- **Change execution flow at runtime** without modifying code
- **Combine sequential and parallel execution**
- **Add conditional branching** with if-else logic
- **Implement loops** for iterative processing
- **Keep workflow logic separate** from module implementation

## Quick Start

### Basic Sequential Pipeline

```python
import msgflux as mf

# Define simple modules
def prep(msg):
    msg.data_ready = True
    return msg

def process(msg):
    if msg.get("data_ready"):
        msg.processed = "completed"
    return msg

def output(msg):
    print(f"Result: {msg.processed}")
    return msg

# Define workflow with inline
modules = {
    "prep": prep,
    "process": process,
    "output": output
}

message = mf.dotdict()
mf.inline("prep -> process -> output", modules, message)
# Output: Result: completed
```

## Syntax Reference

### 1. Sequential Execution (`->`)

Execute modules in order:

```
"module1 -> module2 -> module3"
```

```python
import msgflux as mf

def step1(msg):
    msg.count = 1
    return msg

def step2(msg):
    msg.count += 1
    return msg

def step3(msg):
    msg.count += 1
    return msg

modules = {"step1": step1, "step2": step2, "step3": step3}
msg = mf.dotdict()

mf.inline("step1 -> step2 -> step3", modules, msg)
print(msg.count)  # 3
```

### 2. Parallel Execution (`[...]`)

Execute modules concurrently:

```
"[module1, module2, module3]"
```

**Important:** Parallel modules receive the **same message** and should modify it directly. Return values from parallel modules are **ignored** - use side effects to update the message.

```python
import msgflux as mf

def setup(msg):
    msg.set("input", 100)
    return msg

def compute_double(msg):
    # Modifies message directly
    msg.set("results.doubled", msg.input * 2)
    return msg

def compute_square(msg):
    # Modifies message directly
    msg.set("results.squared", msg.input ** 2)
    return msg

def combine(msg):
    # Access results from parallel modifications
    total = msg.results.doubled + msg.results.squared
    msg.final_result = total
    return msg

modules = {
    "setup": setup,
    "compute_double": compute_double,
    "compute_square": compute_square,
    "combine": combine
}

msg = mf.dotdict()
mf.inline("setup -> [compute_double, compute_square] -> combine", modules, msg)
print(msg.final_result)  # 10200 (200 + 10000)
```

### 3. Conditional Execution (`{condition ? true, false}`)

Branch based on conditions:

```
"{condition ? true_module, false_module}"
"{condition ? only_if_true}"  # false branch is optional
```

**Supported Operators:**

| Operator | Description | Example |
|----------|-------------|---------|
| `==` | Equal | `status == "active"` |
| `!=` | Not equal | `type != "guest"` |
| `>` | Greater than | `age > 18` |
| `<` | Less than | `count < 10` |
| `>=` | Greater or equal | `score >= 80` |
| `<=` | Less or equal | `level <= 5` |
| `is None` | Check if None | `user.name is None` |
| `is not None` | Check if not None | `data is not None` |

**Logical Operators:**

| Operator | Description | Example |
|----------|-------------|---------|
| `&` | AND | `age >= 18 & verified == true` |
| `\|\|` | OR | `is_premium == true \|\| credits > 100` |
| `!` | NOT | `!(banned == true)` |

```python
import msgflux as mf

def adult_flow(msg):
    msg.access = "granted"
    msg.content = "adult content"
    return msg

def child_flow(msg):
    msg.access = "restricted"
    msg.content = "family-friendly content"
    return msg

modules = {
    "adult_flow": adult_flow,
    "child_flow": child_flow
}

# Age check
msg = mf.dotdict()
msg.set("user.age", 25)

mf.inline("{user.age >= 18 ? adult_flow, child_flow}", modules, msg)
print(msg.content)  # "adult content"

# Try with younger user
msg2 = mf.dotdict()
msg2.set("user.age", 15)

mf.inline("{user.age >= 18 ? adult_flow, child_flow}", modules, msg2)
print(msg2.content)  # "family-friendly content"
```

### 4. While Loops (`@{condition}: actions;`)

Execute actions repeatedly while condition is true:

```
"@{condition}: module1 -> module2;"
```

**Safety:** Loops have a maximum iteration limit (default: 1000) to prevent infinite loops.

```python
import msgflux as mf

def increment(msg):
    msg.counter = msg.get("counter", 0) + 1
    return msg

modules = {"increment": increment}

msg = mf.dotdict()
msg.counter = 0

# Loop while counter < 5
mf.inline("@{counter < 5}: increment;", modules, msg)
print(msg.counter)  # 5
```

## Complex Examples

### Combined Sequential and Parallel

```python
import msgflux as mf

def prepare(msg):
    msg.status = "ready"
    msg.data = [1, 2, 3, 4, 5]
    return msg

def filter_even(msg):
    even = [x for x in msg.data if x % 2 == 0]
    msg.set("filtered.even", even)
    return msg

def filter_odd(msg):
    odd = [x for x in msg.data if x % 2 != 0]
    msg.set("filtered.odd", odd)
    return msg

def sum_all(msg):
    total_even = sum(msg.filtered.even)
    total_odd = sum(msg.filtered.odd)
    msg.results = {
        "even_sum": total_even,
        "odd_sum": total_odd,
        "total": total_even + total_odd
    }
    return msg

modules = {
    "prepare": prepare,
    "filter_even": filter_even,
    "filter_odd": filter_odd,
    "sum_all": sum_all
}

msg = mf.dotdict()
mf.inline("prepare -> [filter_even, filter_odd] -> sum_all", modules, msg)

print(msg.results)
# {'even_sum': 6, 'odd_sum': 9, 'total': 15}
```

### Multi-Condition Logic

```python
import msgflux as mf

def premium_access(msg):
    msg.features = ["basic", "advanced", "premium", "priority_support"]
    return msg

def basic_access(msg):
    msg.features = ["basic"]
    return msg

def denied_access(msg):
    msg.features = []
    msg.reason = "account suspended or inactive"
    return msg

modules = {
    "premium_access": premium_access,
    "basic_access": basic_access,
    "denied_access": denied_access
}

# Complex condition with AND and OR
msg = mf.dotdict()
msg.set("user.is_premium", True)
msg.set("user.is_active", True)
msg.set("user.is_suspended", False)

workflow = "{user.is_active == true & !user.is_suspended == true ? " \
           "{user.is_premium == true ? premium_access, basic_access}, " \
           "denied_access}"

mf.inline(workflow, modules, msg)
print(msg.features)  # ['basic', 'advanced', 'premium', 'priority_support']
```

### Loop with Parallel Processing

```python
import msgflux as mf
import asyncio

async def init_batch(msg):
    msg.batch_count = 0
    msg.processed_items = []
    return msg

async def process_a(msg):
    msg.set("temp.result_a", f"A-{msg.batch_count}")
    return msg

async def process_b(msg):
    msg.set("temp.result_b", f"B-{msg.batch_count}")
    return msg

async def collect_results(msg):
    msg.processed_items.append({
        "a": msg.temp.result_a,
        "b": msg.temp.result_b
    })
    msg.batch_count += 1
    return msg

modules = {
    "init_batch": init_batch,
    "process_a": process_a,
    "process_b": process_b,
    "collect_results": collect_results
}

msg = mf.dotdict()

# Loop with parallel processing
result = await mf.ainline(
    "init_batch -> @{batch_count < 3}: [process_a, process_b] -> collect_results;",
    modules,
    msg
)

print(result.processed_items)
# [
#   {'a': 'A-0', 'b': 'B-0'},
#   {'a': 'A-1', 'b': 'B-1'},
#   {'a': 'A-2', 'b': 'B-2'}
# ]
```

### None Checks

```python
import msgflux as mf

def request_name(msg):
    msg.prompt = "Please enter your name"
    return msg

def greet_user(msg):
    msg.greeting = f"Hello, {msg.user.name}!"
    return msg

def validate(msg):
    if msg.get("user.name"):
        msg.name_provided = True
    return msg

modules = {
    "request_name": request_name,
    "greet_user": greet_user,
    "validate": validate
}

# Case 1: Name is None
msg1 = mf.dotdict()
msg1.set("user.name", None)

mf.inline("{user.name is None ? request_name, greet_user}", modules, msg1)
print(msg1.prompt)  # "Please enter your name"

# Case 2: Name is provided
msg2 = mf.dotdict()
msg2.set("user.name", "Alice")

mf.inline("{user.name is None ? request_name, greet_user}", modules, msg2)
print(msg2.greeting)  # "Hello, Alice!"

# Case 3: Check not None with validation
msg3 = mf.dotdict()
msg3.set("user.name", "Bob")

mf.inline("{user.name is not None ? validate}", modules, msg3)
print(msg3.name_provided)  # True
```

## Async Support

Use `ainline` for asynchronous workflows:

```python
import msgflux as mf
import asyncio

async def async_fetch(msg):
    await asyncio.sleep(0.1)  # Simulate async operation
    msg.data = "fetched"
    return msg

async def async_process(msg):
    await asyncio.sleep(0.1)
    msg.processed = f"{msg.data}_processed"
    return msg

modules = {
    "fetch": async_fetch,
    "process": async_process
}

msg = mf.dotdict()

# Use ainline for async workflows
result = await mf.ainline("fetch -> process", modules, msg)
print(result.processed)  # "fetched_processed"
```

## Best Practices

### 1. Keep Modules Pure

```python
# Good - Pure function, returns modified message
def process(msg):
    msg.result = msg.value * 2
    return msg

# Avoid - Side effects
def process_bad(msg):
    global_var = msg.value  # Don't use global state
    write_to_file(msg)      # Don't perform I/O
    return msg
```

### 2. Use Descriptive Module Names

```python
# Good - Clear, descriptive names
modules = {
    "validate_input": validate_fn,
    "fetch_user_data": fetch_fn,
    "send_notification": notify_fn
}

# Avoid - Vague names
modules = {
    "a": fn1,
    "do_stuff": fn2,
    "handler": fn3
}
```

### 3. Handle Parallel Execution Correctly

```python
# Good - Modify message directly (return values are ignored)
def parallel_task(msg):
    value = msg.get("input_value")
    result = compute(value)
    msg.set("results.task_output", result)  # Store in message
    return msg

# Bad - Returning dict doesn't work (return values are ignored!)
def parallel_task_bad(msg):
    value = msg.get("input_value")
    result = compute(value)
    return {"output": result}  # This is IGNORED in parallel execution!
```

### 4. Design for Readability

```python
# Good - Workflow is easy to understand
workflow = "validate -> fetch_data -> [process_a, process_b] -> combine -> output"

# Acceptable but harder to read
workflow = "{valid ? {premium ? [a,b,c], [d,e]}, fail}"

# Better - Break complex logic into steps
workflow = "validate -> {premium ? premium_flow, basic_flow} -> output"
```

### 5. Set Iteration Limits for Loops

```python
# Good - Bounded loops
workflow = "@{counter < 10}: process;"

# Risky - Could run indefinitely
workflow = "@{status != 'done'}: process;"  # Make sure 'done' is eventually set
```

## Integration with Modules

### Using with nn.Module

```python
import msgflux as mf
import msgflux.nn as nn

class DataProcessor(nn.Module):
    def forward(self, msg):
        msg.processed = True
        return msg

class Validator(nn.Module):
    def forward(self, msg):
        msg.valid = msg.get("data") is not None
        return msg

# Use modules in inline workflow
processor = DataProcessor()
validator = Validator()

modules = {
    "validate": validator,
    "process": processor
}

msg = mf.dotdict({"data": [1, 2, 3]})
mf.inline("validate -> {valid ? process}", modules, msg)
```

### Dynamic Workflow Selection

```python
import msgflux as mf

def get_workflow(user_type):
    """Select workflow based on user type."""
    workflows = {
        "admin": "validate -> process_admin -> audit -> notify",
        "user": "validate -> process_user -> notify",
        "guest": "rate_limit -> validate -> {allowed ? process_guest}"
    }
    return workflows.get(user_type, "reject")

# Use dynamically selected workflow
user_type = "admin"
workflow = get_workflow(user_type)

modules = {
    "validate": validate_fn,
    "process_admin": admin_fn,
    "process_user": user_fn,
    "process_guest": guest_fn,
    "audit": audit_fn,
    "notify": notify_fn,
    "rate_limit": rate_limit_fn,
    "reject": reject_fn
}

msg = mf.dotdict({"user_type": user_type})
mf.inline(workflow, modules, msg)
```


## Syntax Summary

| Feature | Syntax | Example |
|---------|--------|---------|
| Sequential | `->` | `"a -> b -> c"` |
| Parallel | `[...]` | `"[a, b, c]"` |
| Conditional | `{cond ? t, f}` | `"{x > 5 ? yes, no}"` |
| While Loop | `@{cond}: actions;` | `"@{count < 10}: inc;"` |
| AND | `&` | `"{a & b ? yes}"` |
| OR | `\|\|` | `"{a \|\| b ? yes}"` |
| NOT | `!` | `"{!a ? yes}"` |
| Is None | `is None` | `"{x is None ? ask}"` |
| Not None | `is not None` | `"{x is not None ? use}"` |
