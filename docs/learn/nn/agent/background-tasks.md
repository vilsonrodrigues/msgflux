# Background Tasks

Background tasks let a tool start work now and return the final result later.

The current design is intentionally small:

- `background=True` dispatches the tool and returns a `task_id`
- `task_status(task_id)` returns task state and progress
- `task_output(task_id)` returns the final output
- `task_list()` lists tasks visible in the current `ToolLibrary`
- completed and failed tasks can also be delivered back to the agent as a
  passive notification
- `inject_library=True` lets a tool add or remove tools dynamically

This page focuses on the current behavior, not future multi-agent planning.

## Mental Model

```text
tool call
  -> immediate dispatch response with task_id
  -> background execution continues
  -> task state lives in TaskStore
  -> result is consumed later through task tools or notifications
```

## Example 1: Basic Background Tool

```python
import time
import msgflux as mf
import msgflux.nn as nn

mf.load_dotenv()


@mf.tool_config(background=True)
def long_sum(a: int, b: int) -> int:
    """Compute a sum in the background."""
    time.sleep(2)
    return a + b


agent = nn.Agent(
    name="math_assistant",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    instructions="Use tools when needed.",
    tools=[long_sum],
)

dispatch = agent.tool_library([("call_1", "long_sum", {"a": 20, "b": 22})])
print(dispatch.tool_calls[0].result)
# The `long_sum` tool is running in the background with task_id='...'

tasks = agent.tool_library([("call_2", "task_list", {})])
task_id = tasks.tool_calls[0].result[0]["task_id"]

state = agent.tool_library([("call_3", "task_status", {"task_id": task_id})])
print(state.tool_calls[0].result)

result = agent.tool_library([("call_4", "task_output", {"task_id": task_id})])
print(result.tool_calls[0].result)
```

## Example 2: Reporting Progress

Use `inject_task=True` when the tool should update its own progress.

```python
import time
import msgflux as mf


@mf.tool_config(background=True, inject_task=True)
def process_items(items: list[str], task) -> int:
    """Process items and report progress."""
    task.set_running(stage="prepare", message="Preparing work")

    total = len(items)
    for index, item in enumerate(items, 1):
        time.sleep(0.2)
        task.update_progress(
            stage="process",
            message=f"Processed {item}",
            current=index,
            total=total,
        )

    return total
```

While the task is running, `task_status(task_id)` returns something like:

```python
{
    "task_id": "9b8e2f1a",
    "tool_name": "process_items",
    "status": "running",
    "progress": {
        "stage": "process",
        "message": "Processed b.txt",
        "current": 2,
        "total": 3,
        "percent": 66.67,
    },
}
```

## Example 3: Passive Notification Back Into The Agent

Completed and failed tasks are injected back into the next provider call as a
synthetic user message:

```text
<system_note>
<notifications>
<notification source="task" ref="abcd1234" status="completed">
tool=long_sum
hint=Use task_output(task_id='abcd1234') if you need the result.
</notification>
</notifications>
</system_note>
```

That means the model can recover task output without polling manually on every
turn, as long as the prompt tells it what to do with these notifications.

```python
agent = nn.Agent(
    name="assistant",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    instructions=(
        "If you receive a task notification with status=completed, "
        "call task_output for that task before answering."
    ),
    tools=[long_sum],
)
```

## Example 3B: Progress Notifications

The same injected `task` handle can publish lightweight agent-visible updates.

```python
@mf.tool_config(background=True, inject_task=True)
def process_items(items: list[str], task) -> int:
    """Process items and publish progress notifications."""
    total = len(items)
    for index, item in enumerate(items, 1):
        task.notify(
            source="task_progress",
            status="update",
            metadata={"item": item, "current": index, "total": total},
            dedupe_key=f"task_progress:{task.task_id}",
        )
    return total
```

These notifications are persisted like other inbox notifications. `dedupe_key`
keeps the newest progress update for the same task visible to the model.

## Example 4: Dynamic Tool Mutation With `inject_library`

`inject_library=True` exposes a small `tool_library` handle to the tool.

The current handle supports:

- `tool_library.add(tool)`
- `tool_library.remove(tool_name)`
- `tool_library.list_tools()`

```python
import msgflux as mf


def multiply(x: int) -> int:
    """Multiply a number by two."""
    return x * 2


@mf.tool_config(inject_library=True)
def enable_multiplier(tool_library) -> list[str]:
    """Register the multiply tool."""
    tool_library.add(multiply)
    return tool_library.list_tools()


@mf.tool_config(inject_library=True)
def disable_tool(tool_library, name: str) -> list[str]:
    """Remove a tool by name."""
    tool_library.remove(name)
    return tool_library.list_tools()
```

If the injected tool adds a new background tool, the task tools are registered
automatically in the same library.

## Current Limits

This is the current scope of the implementation:

- task state is in-memory
- notifications are injected from `_prepare_model_execution()`
- `inspect_model_execution_params()` peeks notifications without consuming them
- there is no checkpoint resume yet
- there is no mailbox or agent-to-agent messaging yet

## Related Pages

- [Tools](tools.md)
- [Task Runtime](../../../anatomy/task-runtime.md)
