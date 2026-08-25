# Tool Search

Tool search lets a catalog grow without placing every tool schema in every
request. The initial context stays approximately stable as rarely used tools
are added; mark those tools with `defer_loading=True`:

```python
import msgflux as mf
import msgflux.nn as nn


@mf.tool_config(defer_loading=True)
def query_finance_report(company: str) -> str:
    """Query archived finance reports for a company."""
    return f"Finance report for {company}"


agent = nn.Agent(
    name="analyst",
    model=mf.Model.chat_completion("openai/gpt-5.6-luna"),
    tools=[query_finance_report],
)
```

The name follows the OpenAI and Anthropic convention: the tool remains in the
logical `ToolCatalog`, but its full callable definition is deferred until the
model searches for it.

## Hosted And Portable Modes

OpenAI GPT-5.6 models using Responses compile deferred tools to the hosted
protocol:

```python
[
    {"type": "tool_search"},
    {
        "type": "function",
        "name": "query_finance_report",
        "defer_loading": True,
        # description and parameters omitted here
    },
]
```

OpenAI performs the search inside the same Responses request. Its
`tool_search_call` and `tool_search_output` items are retained in
`ChatMessages`, followed by the normal function call that msgFlux executes.

The model implementation checks both the active model and API mode through
`supports_native_tool_search()`. Unknown models, Chat Completions, and providers
without hosted search use the portable fallback: msgFlux exposes its local
`tool_search` function first, then exposes a selected tool on the next model
turn. This works with any msgFlux provider that supports ordinary function
calling, including local tools, agents, and MCP tools.

!!! warning "Portable loading changes the tool prefix"

    Loading a portable deferred tool changes the tool payload on the following
    request. The feature remains correct, but that boundary can reduce provider
    prompt-cache hits. Hosted Responses search keeps discovery within the native
    protocol and avoids this intermediate schema transition.

## Thread-Local Loading

Loaded tools belong to a conversation, not to the shared `ToolLibrary`.
`ChatMessages.metadata` stores the loaded names under the library's catalog ID:

```python
from msgflux.chat_messages import ChatMessages

first = ChatMessages(thread_id="analysis-a")
second = ChatMessages(thread_id="analysis-b")

agent.tool_library(
    [("call_1", "tool_search", {"select": ["query_finance_report"]})],
    messages=first,
)

print(first.get_loaded_tools(agent.tool_library.name))
# {"query_finance_report"}
print(second.get_loaded_tools(agent.tool_library.name))
# set()
```

The executable catalog is not mutated. This means simultaneous threads can
share one agent safely while exposing different tool subsets. The state is also
copied and checkpointed with `ChatMessages`.

## Search And Load Locally

A search returns matching names without loading them:

```python
result = agent.tool_library(
    [("call_2", "tool_search", {"query": "finance report"})],
    messages=first,
).tool_calls[0].result

print(result["matches"])
# ["query_finance_report"]
print(result["loaded"])
# []
```

Use `select` to load exact matches for that thread:

```python
result = agent.tool_library(
    [
        (
            "call_3",
            "tool_search",
            {"select": ["query_finance_report"]},
        )
    ],
    messages=first,
).tool_calls[0].result

print(result["loaded"])
# ["query_finance_report"]
```

`query="select:query_finance_report"` remains supported as a compact selection
syntax. Set `description=True` to include display names, descriptions, usage
guidance, and tool kinds in search results.

After loading, `ToolCatalog.portable_tools()` includes the selected tool for
`first`; it continues to include only the local `tool_search` fallback for
`second`. Local functions, modules, agents, and MCP tools use the same catalog
and thread-state contract.
