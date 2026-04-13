# Task Runtime

`Task Runtime` is the runtime contract for background tools in msgFlux.

It exists to support three behaviors that do not fit well inside a plain
tool-call result:

- work that continues after the current model turn ends
- progress updates while that work is still running
- result delivery through either polling or passive notification

This page describes the internal shape for `background task`, `progress`, and
`notification` support. It does not cover checkpointing, agent resume, or a
general event stream.

## Mental Model

The runtime should separate execution from task state.

```text
tool call
  -> ToolLibrary
  -> TaskStore.create(...)
  -> background runner
     -> task status updates
     -> optional progress updates
     -> final result or error
  -> NotificationBus.emit(...)
```

The important shift is that a background tool no longer writes directly to a
`Future` registry only. It writes to a durable task record first, and the
future becomes an implementation detail of the runner.

## Core Primitives

The first version should introduce three runtime primitives.

### `TaskStore`

`TaskStore` is the source of truth for task state.

It owns:

- task creation
- task status transitions
- progress snapshots
- result and error persistence
- lookup by `task_id`
- listing tasks by scope

The minimum task shape should be:

```text
task_id
status
tool_name
created_at
updated_at
completed_at
result
error
progress
metadata
```

`progress` should be a structured object, not a string:

```text
stage
message
current
total
percent
```

`percent` stays optional. Some tasks only have stage-based progress.

### `TaskHandle`

`TaskHandle` is the controlled object injected into a background tool.

It exists so the tool can report progress without mutating the store directly.

The first version only needs methods like:

- `set_running(...)`
- `update_progress(...)`
- `complete(result)`
- `fail(error)`

This keeps background progress cooperative. The runtime does not try to infer
progress from stdout, logs, or provider output.

### `NotificationBus`

`NotificationBus` is the delivery layer for task updates.

It should not own task state. It only reacts to task events and decides how
they are delivered.

The first version only needs:

- `publish_task_completed(task_record)`
- `publish_task_failed(task_record)`

## Two Delivery Paths

Background tasks should support both delivery paths from the start.

### Active Path

The active path is explicit polling through tools:

- `task_status(task_id)` returns rich state
- `task_output(task_id)` returns only the final output
- `task_list(...)` returns tasks visible in the current scope

This path is canonical. A passive notification should never be the only way to
observe task state.

### Passive Path

The passive path is automatic delivery when a task changes state.

The runtime materializes that delivery as a synthetic user message wrapped in
`<system_note>` and `<task_notification>` tags.

That choice keeps notifications compatible with the existing message-driven
agent loop without inventing a second message protocol.

## Status Model

The initial status machine should stay small:

```text
queued -> running -> completed
queued -> running -> failed
```

The first implementation does not need retries, child tasks, or checkpoint
resume in the state machine.

## Execution Flow

The background path should look like this:

```text
model emits tool call
  -> ToolLibrary resolves tool
  -> if tool_config.background:
       -> TaskStore.create(status=queued)
       -> ToolLibrary returns task_id immediately
       -> runner executes tool in background
          -> TaskHandle.set_running(...)
          -> tool body updates progress
          -> TaskHandle.complete(...) or fail(...)
          -> NotificationBus publishes completion/failure
```

This means the immediate tool result becomes a lightweight dispatch response,
not the business result itself.

## Tool Config Shape

The first implementation can stay close to the current tool config model.

Recommended additions:

- `background=True`
- `inject_task=True`
- `inject_library=True`

The important detail is that `inject_task=True` should inject a `TaskHandle`,
not the store and not the full `ToolLibrary`.

`inject_library=True` injects a controlled `tool_library` handle. The handle
can add, remove, and list tools without exposing the whole `ToolLibrary`
object.

## Why Both `task_status` And `task_output`

These tools solve different problems.

- `task_status` is for orchestration and polling logic
- `task_output` is for consuming the final payload

If `task_output` is the only reader, the runtime will collapse state, progress,
and final result into one interface. That makes notifications and collaboration
harder later.

## Notification Policy

The runtime should be conservative about progress notifications.

By default:

- always notify on `completed`
- always notify on `failed`
- do not notify every progress update

## Implementation Boundaries

The code should stay split by responsibility:

```text
src/msgflux/runtime/tasks.py
  -> TaskRecord
  -> TaskProgress
  -> TaskStore
  -> TaskHandle

src/msgflux/runtime/notifications.py
  -> Notification
  -> NotificationBus

src/msgflux/nn/modules/tool.py
  -> background dispatch integration
  -> task tools registration
  -> TaskHandle injection

src/msgflux/nn/modules/agent.py
  -> notification delivery into message history
```

`ToolLibrary` remains the execution boundary for tools. It should not absorb
all task and notification logic directly into one class.

## Out Of Scope

The first implementation should explicitly avoid these features:

- checkpointing
- agent resume/reopen
- multi-agent coordination
- free-form event streaming

Those can build on top of this runtime later, but they should not shape the
first task contract.

## Why This Shape Matters

If background work is modeled only as `Future -> string result`, msgFlux gets
dispatch but not orchestration.

If notifications are modeled only as ad-hoc callbacks, msgFlux gets delivery
but not a clean message contract.

This runtime shape keeps the boundaries clear:

- `ToolLibrary` dispatches work
- `TaskStore` owns task state
- `NotificationBus` owns delivery
- `Agent` consumes notifications as messages

That gives msgFlux a base for polling, passive delivery, progress reporting,
and later coordination features without redesigning the first contract.
