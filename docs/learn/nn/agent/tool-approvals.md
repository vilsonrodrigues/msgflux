# Tool Approvals

Tool approvals let an agent pause before executing sensitive tools. The tool
declares that an action requires approval, while the active runtime decides how
to handle that approval.

Use this for operations such as file writes, ticket updates, deployments,
payments, external API mutations, and any tool where a UI, CLI, or operator may
need to review the action before it runs.

## Core Model

There are two separate concepts:

| Concept | API | Purpose |
|---------|-----|---------|
| Approval requirement | `@mf.tool_config(approval={...})` | Marks a tool call as requiring runtime approval and describes the action. |
| Permission mode | `ExecutionScope(permission_mode=...)` or `PermissionManager(default_mode=...)` | Decides whether the approval is bypassed, denied, or sent to a user/operator. |

The tool does not define the policy. A tool only says: "this action needs
approval." The active execution scope or permission manager decides what happens.

!!! warning

    Tool approvals are async-only. Use `agent.acall(...)`,
    `agent.astream_events(...)`, or `ToolLibrary.aforward(...)`. The sync path
    does not wait for approval and will return a tool error when a called tool
    has `approval=...`.

## Permission Modes

msgFlux currently supports three modes:

| Mode | Behavior |
|------|----------|
| `bypass` | Automatically approves the request and executes the tool. |
| `deny` | Automatically denies the request and the tool is not dispatched. |
| `ask_user` | Creates a pending request and waits until another task/thread calls `approve(...)` or `deny(...)`. |

Resolution order:

1. `ExecutionScope.permission_mode`
2. `PermissionManager.default_mode`

This lets one shared agent run with different behavior per session.

```python
import msgflux as mf

manager = mf.PermissionManager(default_mode="deny")

scope = mf.ExecutionScope(
    session_id="user_42",
    run_id="run_001",
    permission_mode="ask_user",
)
```

In this example, `ask_user` wins for the current execution even though the
manager default is `deny`.

## Mark A Tool As Requiring Approval

Use `approval` in `tool_config`.

```python
import msgflux as mf


@mf.tool_config(
    approval={
        "action": "file.write",
        "risk": "high",
        "resource_arg": "path",
        "reason": "This tool writes to the local filesystem.",
        "metadata": {"surface": "workspace"},
    }
)
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    with open(path, "w") as file:
        file.write(content)
    return f"Wrote {path}"
```

Supported approval metadata:

| Key | Purpose |
|-----|---------|
| `action` | Stable action name, for example `file.write` or `ticket.update`. Defaults to the tool name. |
| `risk` | Risk label: `low`, `medium`, or `high`. |
| `resource` | Static resource string to show in the request. |
| `resource_arg` | Name of a tool argument that should be used as the resource. |
| `reason` | Human-readable reason for the approval request. |
| `metadata` | Extra structured data for a UI, CLI, or event consumer. |

Do not put `mode` or `policy` in the tool approval config. Policy belongs to
the runtime scope or permission manager.

## Async Approval Flow

With `ask_user`, the agent waits until another task or thread resolves the
pending request.

```python
import asyncio
import msgflux as mf
import msgflux.nn as nn


@mf.tool_config(approval={"action": "ticket.close", "resource_arg": "ticket_id"})
def close_ticket(ticket_id: str) -> str:
    """Close a ticket."""
    return f"{ticket_id} closed"


agent = nn.Agent(
    name="support_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    tools=[close_ticket],
    permission_manager=mf.PermissionManager(default_mode="ask_user"),
)

scope = mf.ExecutionScope(
    session_id="customer_42",
    run_id="ticket_9001",
    permission_mode="ask_user",
)


async def run_agent():
    task = asyncio.create_task(
        agent.acall("Close ticket MSGFLUX-42.", scope=scope)
    )

    while not agent.permission_manager.list_pending():
        await asyncio.sleep(0.05)

    request = agent.permission_manager.list_pending()[0]
    print(request.request_id, request.action, request.resource)

    agent.permission_manager.approve(
        request.request_id,
        reason="Approved by support operator.",
    )

    return await task


asyncio.run(run_agent())
```

To reject the tool call, resolve the same request with `deny`.

```python
agent.permission_manager.deny(
    request.request_id,
    reason="The customer did not confirm the action.",
)
```

When denied, the tool is not executed. The model receives a tool error and can
explain the denial or choose a safer path.

## Dispatch Guarantees

For async tool execution, msgFlux resolves approvals before dispatching tools.
If a model requests multiple approval-gated tools in one turn, none of those
tools is dispatched until all approvals for that turn have been resolved.

This matters for `spawn=True` and `background=True` tools: approval is checked
before the background process starts.

```python
@mf.tool_config(background=True, approval={"action": "report.generate"})
def generate_report(account_id: str) -> str:
    """Generate a report in the background."""
    return f"report for {account_id}"
```

If the active mode is `deny`, the background task is not created. If the mode is
`ask_user`, the background task is created only after approval.

## Sync Limitation

The sync path is intentionally limited:

```python
response = agent("Close ticket MSGFLUX-42.")
```

If a called tool has `approval=...`, sync execution returns an error for that
tool call instead of waiting for a user decision. This avoids blocking a sync
call on an external approval channel.

Use async instead:

```python
response = await agent.acall("Close ticket MSGFLUX-42.")
```

For CLIs, servers, chat UIs, and long-running sessions, use async execution and
resolve pending requests from the UI or control plane.

## Subagents

Subagents inherit the parent execution scope. This means a coordinator agent can
delegate to a subagent, and tools inside that subagent still use the parent
session and permission mode.

```python
import msgflux as mf
import msgflux.nn as nn


@mf.tool_config(
    approval={
        "action": "ticket.update",
        "risk": "high",
        "resource_arg": "ticket_id",
    }
)
def update_ticket(ticket_id: str, status: str) -> str:
    """Update ticket state."""
    return f"{ticket_id} updated to {status}"


worker = nn.Agent(
    name="ticket_worker",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    tools=[update_ticket],
    instructions="Update tickets when delegated by the coordinator.",
)

coordinator = nn.Agent(
    name="support_coordinator",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    tools=[worker],
    permission_manager=mf.PermissionManager(default_mode="deny"),
)

scope = mf.ExecutionScope(
    session_id="customer_42",
    run_id="ticket_9001",
    permission_mode="ask_user",
)

response = await coordinator.acall(
    "Ask ticket_worker to update MSGFLUX-42 to approved.",
    scope=scope,
)
```

The pending request will include the subagent as the caller:

```python
request = coordinator.permission_manager.list_pending()[0]

print(request.tool_name)     # update_ticket
print(request.caller_name)   # ticket_worker
print(request.resource)      # MSGFLUX-42
```

This lets a UI show both the tool being called and which agent requested it.

## Events

Permission activity is also emitted through runtime events:

| Event | Meaning |
|-------|---------|
| `permission.requested` | A tool approval was requested. |
| `permission.granted` | The request was approved or bypassed. |
| `permission.denied` | The request was denied or timed out. |

```python
with mf.EventStream() as stream:
    response = await agent.acall("Close ticket MSGFLUX-42.", scope=scope)
    stream.close()

for event in stream.events:
    if event.name.startswith("permission."):
        print(event.name, event.attributes)
```

These events are useful for CLI logs, UI timelines, audit trails, and future
streaming integrations.

## Runnable Example

The repository includes an offline example with a coordinator agent, a subagent,
and an approval-gated tool:

```bash
uv run python examples/tool_approval_runtime_demo.py
```

Approve the request:

```bash
uv run python examples/tool_approval_runtime_demo.py --mode ask_user --decision approve
```

Deny the request:

```bash
uv run python examples/tool_approval_runtime_demo.py --mode ask_user --decision deny
```

Run without user approval:

```bash
uv run python examples/tool_approval_runtime_demo.py --mode bypass
uv run python examples/tool_approval_runtime_demo.py --mode deny
```
