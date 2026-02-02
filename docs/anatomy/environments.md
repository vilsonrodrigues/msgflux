# Environments

Code execution environments for LLM agents.

## Architecture Overview

```
environments/
├── base.py              # BaseEnvironment (root class for all environments)
├── pool.py              # EnvironmentPool for reuse and warmup
├── code/
│   ├── base.py          # BaseCodeEnvironment, BasePythonEnvironment
│   ├── registry.py      # Environments factory
│   ├── response.py      # ExecutionResult
│   └── providers/
│       └── deno_pyodide.py  # DenoPyodideSandbox
└── exceptions.py        # Error types
```

## Core Components

### ExecutionResult

Standardized response from code execution:

```python
@dataclass
class ExecutionResult:
    success: bool                    # Execution succeeded
    output: Optional[str]            # stdout
    error: Optional[str]             # Error message (e.g., "NameError: name 'x' is not defined")
    return_value: Optional[Any]      # Last expression value
    variables: Dict[str, Any]        # Variables after execution
    execution_time_ms: Optional[float]
    memory_used_bytes: Optional[int]
    metadata: Dict[str, Any]         # Extensible
```

### BaseCodeEnvironment

Abstract base for all environments:

```python
class BaseCodeEnvironment:
    def __call__(self, action: str, *, timeout=None, vars=None, tools=None) -> ExecutionResult
    async def acall(self, action: str, *, timeout=None, vars=None, tools=None) -> ExecutionResult
    def reset(self) -> None
    def shutdown(self) -> None
```

### Environments Factory

```python
from msgflux.environments import Environments

# Create Python environment
env = Environments.code("python", timeout=30.0)

# With packages pre-installed
env = Environments.code("python", packages=["numpy", "pandas"])

# List available
Environments.list_types()      # ["python"]
Environments.list_providers("python")  # ["deno_pyodide"]
```

## Environment Pooling

For efficient reuse of environments across multiple executions:

```python
from msgflux.environments import EnvironmentPool, Environments

# Create pool with pre-installed packages
pool = EnvironmentPool(
    factory=lambda: Environments.code("python", timeout=60.0),
    packages=["numpy", "pandas"],
    min_size=2,      # Pre-warm 2 environments
    max_size=10,     # Maximum pool size
    idle_timeout=300.0,  # Remove idle envs after 5 min
)

# Acquire and use
env = pool.acquire()
result = env("import numpy; print(numpy.__version__)")
pool.release(env)

# Or use context manager
with pool.acquire() as env:
    result = env("print('hello')")

# Cleanup
pool.close()
```

### Pool Statistics

```python
stats = pool.stats
# {
#     "created": 5,
#     "acquired": 100,
#     "released": 100,
#     "destroyed": 2,
#     "cache_hits": 95,
#     "cache_misses": 5,
#     "pool_size": 3,
#     "in_use": 2,
# }
```

## Package Caching

By default, packages are downloaded on each environment creation. To enable local caching:

```bash
export MSGFLUX_DENO_ALLOW_CACHE_WRITE=true
```

This allows Deno to write package wheels to its cache directory, speeding up subsequent environment creations.

## Integration with Agent

### FlowControl Architecture

```
Agent
  ├── FlowControl (generation_schema)
  │     ├── extract_flow_result() → FlowResult
  │     │     ├── environment_call: EnvironmentCall  # Code execution
  │     │     └── tool_calls: List[Tuple]            # Tool calls
  │     ├── inject_environment_result()
  │     ├── inject_tool_results()
  │     └── build_history()
  └── Environment (nn.Environment)
        └── execute action with tools/vars
```

### EnvironmentCall

Request for code execution:

```python
@dataclass
class EnvironmentCall:
    action: str           # Code to execute
    inject_vars: bool     # Inject context variables
    inject_tools: bool    # Inject tools as functions
```

### Agent Configuration

```python
from msgflux.nn import Agent, Environment
from msgflux.environments import Environments
from msgflux.generation.reasoning import ProgramOfThought

agent = Agent(
    "solver",
    model,
    environment=Environment(environment=Environments.code("python")),
    tools=[search, calculate],
    generation_schema=ProgramOfThought,
)

result = agent("What is 2 + 2?", vars={"context": data})
```

## FlowControl Schemas

### ProgramOfThought

LLM writes Python code iteratively:

```python
class ProgramOfThought(Struct, FlowControl):
    current_step: Optional[ProgramOfThoughtStep]  # thought + code
    final_answer: Optional[str]
```

- Returns `environment_call` with code
- `inject_environment_result()` adds execution output to step

### RLM (Recursive Language Model)

For long-context tasks with code-based data exploration:

```python
class RLM(Struct, FlowControl):
    current_step: Optional[RLMStep]  # reasoning + code
    final_answer: Optional[str]
```

Includes helper tools:
- `LLMQuery(lm)` - Query sub-LLM from sandbox
- `LLMQueryBatched(lm)` - Batch queries with `F.map_gather`

### ReAct

Traditional tool calling (not environment-based):

```python
class ReAct(Struct, FlowControl):
    current_step: Optional[ReActStep]  # thought + actions
    final_answer: Optional[str]
```

- Returns `tool_calls`, not `environment_call`
- Uses `inject_tool_results()`

## nn.Environment Module

Wrapper for code environments in neural module system:

```python
from msgflux.nn import Environment

# With init tools (available in all executions)
env = Environment(
    environment=Environments.code("python"),
    tools={"search": search_fn}
)

# Execute with additional tools
result = env(
    "data = search('query')\nprint(data)",
    tools={"extra": extra_fn},  # Merged with init tools
    vars={"context": "..."}
)

# Async
result = await env.acall(code, tools=tools, vars=vars)
```

Properties:
- `env.name` - Environment identifier (e.g., "execute_code")
- `env.environment_type` - Type (e.g., "python")
- `env.environment` - Underlying implementation
- `env.tools` - Init tools configured at creation

## DenoPyodideSandbox

Secure Python execution via Deno + Pyodide (WebAssembly):

```python
env = Environments.code(
    "python",
    provider="deno_pyodide",
    timeout=30.0,
    allow_network=False,
    allow_read=["/path"],
    allow_write=["/path"],
    packages=["numpy", "pandas"],
)
```

### Security Features

- WebAssembly memory isolation
- Deno permission restrictions
- Thread ownership verification
- Variable size limits (100MB)
- Timeout enforcement

### Tool Communication

Tools execute on host, called from sandbox via JSON-RPC:

```
Sandbox (Pyodide)  ←→  Host (Python)
     │                      │
     │  tool_call request   │
     │ ──────────────────→  │
     │                      │ execute tool
     │  tool_call response  │
     │ ←──────────────────  │
```

## Design Decisions

### Why `action` not `code`?

Generic naming allows future non-Python environments (browser, terminal).

### Why separate `environment_call` from `tool_calls`?

1. Semantic clarity - code execution != tool calling
2. Different execution paths in Agent
3. Enables mixed flows (code + tools) in future

### Why `Environments.code()` factory?

- Consistent with other msgflux patterns
- Supports multiple providers per type
- Easy to add new environment types

### Why FlowControl.inject_environment_result()?

- Schema-specific result formatting
- Decouples Agent from schema internals
- Each schema decides how to present results to LLM

### Why tools per-execution instead of permanent registration?

- Avoids need to "unregister" tools
- Clear scope: tools available for specific execution
- Init tools provide defaults, execution tools override

## Template Variables

FlowControl schemas receive via Jinja:

```python
# In Agent._prepare_model_execution()
inputs = {
    "tool_schemas": [...],
    "tool_choice": "auto",
    "environment_name": "execute_code"  # If environment configured
}
```

Schema template example:

```jinja
## Code Execution Environment
{% if environment_name %}Environment: `{{ environment_name }}`.{% endif %}
{% if tool_schemas %}
## Available Tools
{% for tool in tool_schemas %}
- `{{ tool['function']['name'] }}`: {{ tool['function']['description'] }}
{% endfor %}
{% endif %}
```

## Error Handling

```python
from msgflux.environments.exceptions import (
    SandboxError,           # Base
    SandboxConnectionError, # Process/connection issues
    SandboxTimeoutError,    # Execution timeout
    SandboxSecurityError,   # Thread/permission violation
    SandboxNotReadyError,   # Not initialized
    VariableSizeLimitError, # Variable too large
)
```

Error messages are properly formatted:
```python
result = env("x = undefined_var")
print(result.error)  # "NameError: name 'undefined_var' is not defined"
```

## Adding New Environment Types

1. Create provider in `environments/code/providers/`:

```python
from msgflux.environments.code.base import BaseCodeEnvironment
from msgflux.environments.code.registry import register_environment

@register_environment
class MyEnvironment(BaseCodeEnvironment):
    environment_type = "browser"  # or "python", etc.
    provider = "playwright"
    name = "browser_action"

    def __call__(self, action, *, timeout=None, vars=None, tools=None):
        # Execute action
        return ExecutionResult(...)
```

2. Add to `providers/__init__.py`
3. Use: `Environments.code("browser", provider="playwright")`

## References

- OpenEnv (Meta/HuggingFace): Gymnasium-style agent environments
- E2B: Cloud sandbox infrastructure
- DSPy PythonInterpreter: Code execution pattern
- Program of Thoughts paper: arxiv.org/abs/2211.12588
