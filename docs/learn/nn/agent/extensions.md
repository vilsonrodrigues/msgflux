# Agent Extensions

An `AgentExtension` packages hooks and tools behind one removable capability.
Use a plain `Hook` for one interception point; use an extension when a feature
owns several pieces that should be installed and removed together.

## Define An Extension

```python
from dataclasses import replace

import msgflux.nn as nn
from msgflux.nn.hooks import Hook, SystemPromptContext


class TenantContext(nn.AgentExtension):
    def __init__(self, tenant_client):
        super().__init__("tenant_context")
        self.tenant_client = tenant_client

    async def add_context(self, ctx: SystemPromptContext):
        profile = await self.tenant_client.get_profile(ctx.vars["tenant_id"])
        return replace(ctx, prompt=f"{ctx.prompt}\nTenant profile: {profile}")

    def hooks(self):
        return (
            Hook(event="transform_system_prompt", handler=self.add_context),
        )
```

Install it when constructing the Agent:

```python
agent = nn.Agent(
    name="support",
    model=model,
    extensions=[TenantContext(tenant_client)],
)

response = await agent.acall(
    "Summarize the open requests.",
    vars={"tenant_id": "acme"},
)
```

Async handlers are awaited in registration order. The active
`ExecutionScope.abort_signal` is observed while a handler waits, so aborting a
run cancels its in-flight handler instead of waiting for network I/O to finish.

## Add Tools

Return ordinary msgFlux tools from `tools()`:

```python
class CustomerActions(nn.AgentExtension):
    def __init__(self, client):
        super().__init__("customer_actions")
        self.client = client

    def tools(self):
        async def get_customer(customer_id: str) -> dict:
            """Get a customer by id."""
            return await self.client.get_customer(customer_id)

        return (get_customer,)
```

The Agent registers the tool in its existing `ToolLibrary`, preserving the
normal schema, injection, telemetry, tool events, and abort behavior.

## Register And Remove

Registration returns an ownership handle:

```python
handle = agent.register_extension(
    "customer_actions",
    CustomerActions(client),
)

handle.remove()
```

`remove()` disables the extension for new runs immediately. A run that already
started keeps the extension snapshot it began with; cleanup waits until those
runs finish. This prevents one thread from losing a hook or tool because
another thread changed the Agent concurrently.

Use async removal only when the extension itself owns asynchronous cleanup:

```python
await handle.aremove()
```

Registration and removal are intentionally not context managers. Keep the
handle when ownership is temporary, or remove by name with
`agent.remove_extension("customer_actions")`.

## Hooks And Extensions

`hooks=` remains a public low-level API. Extensions compose that API; they do
not replace it. This separation keeps stable lifecycle boundaries reusable by
application hooks, guards, skills, and future features such as compaction.

The `transform_system_prompt` event receives `SystemPromptContext`, including
the rendered prompt, execution scope, runtime `vars`, and active tool names.
Return a replaced dataclass to modify it. Avoid storing per-run data on the
extension instance because one Agent may execute multiple threads
concurrently.

Use `self.state()` when hooks in the same run need to share temporary data:

```python
class ContextAnalysis(nn.AgentExtension):
    def __init__(self, analyzer):
        super().__init__("context_analysis")
        self.analyzer = analyzer

    async def analyze(self, event):
        self.state()["analysis"] = await self.analyzer.acall(event.message)
        return event
```

The state is namespaced by Agent and extension, inherited by nested work in the
same execution context, and discarded when the run ends. It is not persisted in
checkpoints and must not hold durable conversation state.
