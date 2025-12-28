# nn.Module

The `nn.Module` is the base class for all neural network modules in msgflux. It provides a PyTorch-like API with additional features for message passing, state management, and workflow orchestration.

## Overview

`nn.Module` offers a familiar interface for building composable, stateful components:

- **PyTorch-style API**: Similar interface to PyTorch's `nn.Module`
- **State management**: Built-in `state_dict()` for serialization
- **Hooks system**: Pre and post forward hooks for extensibility
- **AutoParams support**: Dataclass-style module definitions (recommended)
- **Async interface**: Automatic async support via `aforward()` and `acall()`
- **OpenTelemetry integration**: Built-in observability
- **Module containers**: `ModuleDict`, `ModuleList`, `Sequential`

## Basic Usage

### Defining a Module

```python
import msgflux as mf
import msgflux.nn as nn

class SimpleModule(nn.Module):
    def __init__(self, greeting="Hello"):
        super().__init__()
        self.greeting = greeting

    def forward(self, name):
        return f"{self.greeting}, {name}!"

# Create and use
module = SimpleModule()
result = module("Alice")
print(result)  # "Hello, Alice!"

# With custom greeting
custom = SimpleModule(greeting="Hi")
result = custom("Bob")
print(result)  # "Hi, Bob!"
```

### Using AutoParams (Recommended)

The **preferred** way to define modules is using the `AutoParams` metaclass:

```python
import msgflux as mf
import msgflux.nn as nn

class GreetingModule(nn.Module, metaclass=mf.AutoParams):
    """Module that generates personalized greetings."""

    def __init__(self, greeting, punctuation):
        super().__init__()
        self.greeting = greeting
        self.punctuation = punctuation

    def forward(self, name):
        return f"{self.greeting}, {name}{self.punctuation}"

# Define variants with different defaults
class FormalGreeting(GreetingModule):
    greeting = "Good morning"
    punctuation = "."

class CasualGreeting(GreetingModule):
    greeting = "Hey"
    punctuation = "!"

# Use with defaults
formal = FormalGreeting()
print(formal("Dr. Smith"))  # "Good morning, Dr. Smith."

casual = CasualGreeting()
print(casual("Alex"))  # "Hey, Alex!"

# Override specific parameters
excited = FormalGreeting(punctuation="!!!")
print(excited("Team"))  # "Good morning, Team!!!"
```

## State Management

### Buffers and Parameters

Modules can register **buffers** for configuration state that should be saved/loaded:

```python
import msgflux as mf
import msgflux.nn as nn

class StatefulModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Register buffers (not trainable, but part of state)
        self.register_buffer("response_template", "Hello, {name}!")
        self.register_buffer("max_length", 50)

        # Use regular attributes for mutable state during execution
        self.call_count = 0

    def forward(self, name):
        # Update call count (regular attribute)
        self.call_count += 1
        response = self.response_template.replace("{name}", name)

        # Use buffer value
        if len(response) > self.max_length:
            response = response[:self.max_length] + "..."

        return f"[Call #{self.call_count}] {response}"

module = StatefulModule()
print(module("Alice"))  # "[Call #1] Hello, Alice!"
print(module("Bob"))    # "[Call #2] Hello, Bob!"

# Buffers are included in state_dict
state = module.state_dict()
print(state["response_template"])  # "Hello, {name}!"
print(state["max_length"])  # 50
```

### Saving and Loading State

```python
import msgflux as mf
import msgflux.nn as nn

class ConfigurableModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("template", "Response: {input}")
        self.register_buffer("prefix", "")

    def forward(self, text):
        formatted = self.template.replace("{input}", text)
        return self.prefix + formatted

# Create and use module
module = ConfigurableModule()
print(module("test"))  # "Response: test"

# Save state
mf.save(module.state_dict(), "module_state.json")

# Load and modify state
state = mf.load("module_state.json")
state["prefix"] = "[INFO] "
state["template"] = "Output: {input}"

# Load modified state
module.load_state_dict(state)
print(module("test"))  # "[INFO] Output: test"
```

**Supported format:**
- `*.json`

## Hooks System

Hooks allow you to intercept and modify module execution without changing the module code.

### Forward Pre-Hooks

Execute code before the forward pass:

```python
import msgflux as mf
import msgflux.nn as nn

class MessageModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("response", "Message received.")

    def forward(self, message, **kwargs):
        user = kwargs.get("user_name", "Guest")
        return f"{user}: {message} -> {self.response}"

# Define pre-hook to enhance context
def add_user_context(module, args, kwargs):
    """Look up user name from ID."""
    user_id = kwargs.get("user_id")
    if user_id == "123":
        kwargs["user_name"] = "Alice"
    elif user_id == "456":
        kwargs["user_name"] = "Bob"
    return args, kwargs

# Register hook
module = MessageModule()
hook_handle = module.register_forward_pre_hook(add_user_context)

# Use with user_id
result = module("Hello", user_id="123")
print(result)  # "Alice: Hello -> Message received."

result = module("Hi", user_id="456")
print(result)  # "Bob: Hi -> Message received."

# Remove hook
hook_handle.remove()
```

### Forward Post-Hooks

Execute code after the forward pass:

```python
import msgflux as mf
import msgflux.nn as nn

class ProcessingModule(nn.Module):
    def forward(self, text):
        return text.upper()

# Define post-hook for logging
def log_output(module, args, kwargs, output):
    """Log all outputs."""
    print(f"[LOG] Output: {output}")
    return output

# Register hook
module = ProcessingModule()
hook_handle = module.register_forward_hook(log_output)

result = module("hello")
# [LOG] Output: HELLO
print(result)  # HELLO

# Remove hook
hook_handle.remove()
```

### Multiple Hooks

Hooks are executed in registration order:

```python
import msgflux as mf
import msgflux.nn as nn

class Module(nn.Module):
    def forward(self, x):
        return x * 2

def hook1(module, args, kwargs, output):
    print("Hook 1")
    return output

def hook2(module, args, kwargs, output):
    print("Hook 2")
    return output

module = Module()
h1 = module.register_forward_hook(hook1)
h2 = module.register_forward_hook(hook2)

module(5)
# Hook 1
# Hook 2

# Access registered hooks
print(module._forward_hooks)  # OrderedDict with hooks
print(module._forward_pre_hooks)  # OrderedDict with pre-hooks
```

## Async Support

Modules automatically support async execution via `acall()`:

```python
import msgflux as mf
import msgflux.nn as nn
import asyncio

class AsyncModule(nn.Module):
    async def aforward(self, text):
        """Async implementation."""
        await asyncio.sleep(0.01)  # Simulate async operation
        return text.upper()

# Sync fallback (optional)
def forward(self, text):
    """Sync implementation."""
    return text.upper()

# Use async
async def main():
    module = AsyncModule()

    # Call async version
    result = await module.acall("hello")
    print(result)  # "HELLO"

    # Sync call also works (falls back to forward if aforward not implemented)
    result = module("world")
    print(result)  # "WORLD"

asyncio.run(main())
```

**Note:** If `aforward()` is not implemented, `acall()` will use `forward()`.

## Module Containers

### ModuleDict

Hold submodules in a dictionary for dynamic routing:

```python
import msgflux as mf
import msgflux.nn as nn

class SalesExpert(nn.Module):
    def forward(self, message):
        return f"Sales: {message} -> Let's discuss pricing!"

class SupportExpert(nn.Module):
    def forward(self, message):
        return f"Support: {message} -> Call our support line!"

class TechExpert(nn.Module):
    def forward(self, message):
        return f"Tech: {message} -> Check the documentation."

class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleDict({
            "sales": SalesExpert(),
            "support": SupportExpert(),
            "tech": TechExpert()
        })

    def forward(self, message, department):
        if department in self.experts:
            return self.experts[department](message)
        return "Department not found."

router = Router()

print(router("I want to buy", "sales"))
# "Sales: I want to buy -> Let's discuss pricing!"

print(router("I need help", "support"))
# "Support: I need help -> Call our support line!"

print(router("How does this work?", "tech"))
# "Tech: How does this work? -> Check the documentation."

# Access state dict
print(router.state_dict())
```

**Key features:**
- Modules properly registered and visible
- Ordered insertion (maintains order)
- Can be indexed like a regular dict

### ModuleList

Hold submodules in a list for sequential processing:

```python
import msgflux as mf
import msgflux.nn as nn

class Uppercase(nn.Module):
    def forward(self, text):
        return text.upper()

class AddPrefix(nn.Module):
    def forward(self, text):
        return f"[PROCESSED] {text}"

class AddSuffix(nn.Module):
    def forward(self, text):
        return f"{text} [DONE]"

class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.steps = nn.ModuleList([
            Uppercase(),
            AddPrefix(),
            AddSuffix()
        ])

    def forward(self, text):
        # Process through all steps
        result = text
        for step in self.steps:
            result = step(result)
        return result

pipeline = Pipeline()
result = pipeline("hello world")
print(result)  # "[PROCESSED] HELLO WORLD [DONE]"

# Access individual steps
print(pipeline.steps[0]("test"))  # "TEST"
```

**Key features:**
- Can be indexed like a list
- Can be iterated over
- Modules properly registered

### Sequential

Chain modules in a cascade:

```python
import msgflux as mf
import msgflux.nn as nn

class Lowercase(nn.Module):
    def forward(self, text):
        return text.lower()

class RemoveSpaces(nn.Module):
    def forward(self, text):
        return text.replace(" ", "")

class AddDashes(nn.Module):
    def forward(self, text):
        return "-".join(text)

# Create sequential pipeline
pipeline = nn.Sequential(
    Lowercase(),
    RemoveSpaces(),
    AddDashes()
)

result = pipeline("Hello World")
print(result)  # "h-e-l-l-o-w-o-r-l-d"
```

**With OrderedDict for named steps:**

```python
from collections import OrderedDict
import msgflux.nn as nn

class Step1(nn.Module):
    def forward(self, x):
        return x + " -> Step1"

class Step2(nn.Module):
    def forward(self, x):
        return x + " -> Step2"

pipeline = nn.Sequential(OrderedDict([
    ("first", Step1()),
    ("second", Step2())
]))

result = pipeline("Start")
print(result)  # "Start -> Step1 -> Step2"

# Access state dict shows named modules
print(pipeline.state_dict())
```

**Difference between Sequential and ModuleList:**
- `Sequential`: Modules are automatically chained (output → input)
- `ModuleList`: Just a container, you control the flow in `forward()`

## Common Patterns

### Configuration Module

```python
import msgflux as mf
import msgflux.nn as nn

class ConfigurableModule(nn.Module, metaclass=mf.AutoParams):
    """Module with configurable behavior."""

    def __init__(self, max_length, add_prefix, prefix_text):
        super().__init__()
        self.max_length = max_length
        self.add_prefix = add_prefix
        self.prefix_text = prefix_text

    def forward(self, text):
        # Truncate
        if len(text) > self.max_length:
            text = text[:self.max_length] + "..."

        # Add prefix
        if self.add_prefix:
            text = f"{self.prefix_text}{text}"

        return text

# Production configuration
class ProductionModule(ConfigurableModule):
    max_length = 100
    add_prefix = True
    prefix_text = "[PROD] "

# Development configuration
class DevModule(ConfigurableModule):
    max_length = 500
    add_prefix = True
    prefix_text = "[DEV] "

# Use
prod = ProductionModule()
dev = DevModule()

text = "A" * 150
print(prod(text))  # "[PROD] AAA...AAA..."
print(dev(text))   # "[DEV] AAAA...AAAA..." (longer)
```

### Conditional Routing

```python
import msgflux as mf
import msgflux.nn as nn

class BasicHandler(nn.Module):
    def forward(self, msg):
        return f"Basic: {msg}"

class PremiumHandler(nn.Module):
    def forward(self, msg):
        return f"Premium: {msg} [VIP Support]"

class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.handlers = nn.ModuleDict({
            "basic": BasicHandler(),
            "premium": PremiumHandler()
        })

    def forward(self, message, user_tier="basic"):
        handler = self.handlers.get(user_tier, self.handlers["basic"])
        return handler(message)

router = Router()
print(router("Help!", "basic"))    # "Basic: Help!"
print(router("Help!", "premium"))  # "Premium: Help! [VIP Support]"
```

### Multi-Step Pipeline with State

```python
import msgflux as mf
import msgflux.nn as nn

class Step1(nn.Module):
    def __init__(self):
        super().__init__()
        # Use regular attribute for runtime state
        self.processed_count = 0

    def forward(self, msg):
        self.processed_count += 1
        msg.set("step1_result", msg.input.upper())
        return msg

class Step2(nn.Module):
    def forward(self, msg):
        msg.set("step2_result", msg.step1_result.replace(" ", "_"))
        return msg

class Step3(nn.Module):
    def forward(self, msg):
        msg.final = f"[{msg.step2_result}]"
        return msg

class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.steps = nn.Sequential(Step1(), Step2(), Step3())

    def forward(self, input_text):
        msg = mf.dotdict({"input": input_text})
        result = self.steps(msg)
        return result.final

pipeline = Pipeline()
print(pipeline("hello world"))  # "[HELLO_WORLD]"
```

## Best Practices

### 1. Use AutoParams for Module Definitions

```python
# Good - AutoParams separates config from logic
class MyModule(nn.Module, metaclass=mf.AutoParams):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def forward(self, x):
        return x * self.param1 + self.param2

class Variant1(MyModule):
    param1 = 2
    param2 = 10

# Bad - Repeating defaults in __init__ signature
class MyModule(nn.Module):
    def __init__(self, param1=2, param2=10):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
```

### 2. Use register_buffer for Non-Trainable State

```python
# Good - State is tracked in state_dict
class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("config", {"key": "value"})

# Bad - State is lost when serializing
class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {"key": "value"}  # Not tracked!
```

### 3. Use Hooks for Cross-Cutting Concerns

```python
# Good - Logging separate from core logic
def logging_hook(module, args, kwargs, output):
    print(f"Output: {output}")
    return output

module.register_forward_hook(logging_hook)

# Bad - Mixing logging with core logic
class Module(nn.Module):
    def forward(self, x):
        result = process(x)
        print(f"Output: {result}")  # Logging mixed in
        return result
```

### 4. Use Sequential for Linear Pipelines

```python
# Good - Clear pipeline structure
pipeline = nn.Sequential(Step1(), Step2(), Step3())

# Acceptable but more verbose
class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.step1 = Step1()
        self.step2 = Step2()
        self.step3 = Step3()

    def forward(self, x):
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        return x
```

### 5. Save and Version State Dicts

```python
# Good - Versioned state management
state = module.state_dict()
state["_version"] = "1.0"
mf.save(state, "module_v1.0.json")

# Load and check version
state = mf.load("module_v1.0.json")
if state.get("_version") != "1.0":
    # Handle migration
    pass
```

## OpenTelemetry Integration

Modules automatically support tracing when telemetry is enabled:

```python
import msgflux as mf
import msgflux.nn as nn

class TracedModule(nn.Module):
    def forward(self, x):
        # Automatically traced
        return x * 2

# Set environment variables to enable tracing:
# MSGTRACE_TELEMETRY_REQUIRES_TRACE=true
# MSGTRACE_TELEMETRY_SPAN_EXPORTER_TYPE=otlp

module = TracedModule()
result = module(5)  # Creates a trace span automatically
```

## API Reference

For complete API documentation, see:

::: msgflux.nn.Module

::: msgflux.nn.ModuleDict

::: msgflux.nn.ModuleList

::: msgflux.nn.Sequential
