# Visualization

Generate a Mermaid diagram of your module:

```python
module.plot()  # Displays a flow diagram
```

!!! warning
    Flow visualization is **experimental** and may be incomplete for complex conditionals.

## Example: Complete Module

```python
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F

class QAWorkflow(nn.Module):
    """A question-answering workflow with retrieval."""

    def __init__(self):
        super().__init__()

        # Models
        chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        # Sub-modules
        self.retriever = nn.Retriever("wikipedia")
        self.agent = nn.Agent(
            "qa-agent",
            chat_model,
            instructions="Answer questions using the provided context."
        )

        # Components for inline DSL
        self.components = nn.ModuleDict({
            "retrieve": self.retriever,
            "answer": self.agent
        })

        # Workflow definition
        self.register_buffer("flux", "retrieve -> answer")

    def forward(self, question: str) -> str:
        msg = mf.dotdict(query=question)
        msg = F.inline(self.flux, self.components, msg)
        return msg.answer

    async def aforward(self, question: str) -> str:
        msg = mf.dotdict(query=question)
        msg = await F.ainline(self.flux, self.components, msg)
        return msg.answer

# Use it
qa = QAWorkflow()
answer = qa("What is quantum entanglement?")
print(answer)

# Save state
mf.save(qa.state_dict(), "qa_workflow.toml")
```
