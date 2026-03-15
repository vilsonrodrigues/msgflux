# Quick Start

Creating an agent in msgFlux takes just three lines: define a model, instantiate the agent, and call it. No boilerplate, no configuration files — just a clean, direct API.

???+ example

    ```python
    # pip install msgflux[openai]
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    agent = nn.Agent("Assistant", model)

    response = agent("What is the capital of France?")
    print(response)  # "The capital of France is Paris."
    ```

## String shorthand

When you don't need to configure extra model parameters (temperature, max_tokens, etc.), you can pass a `"provider/model-id"` string directly as the model argument. msgFlux will call `Model.chat_completion` for you internally.

???+ example

    ```python
    import msgflux.nn as nn

    # Equivalent to Model.chat_completion("openai/gpt-4.1-mini")
    agent = nn.Agent("Assistant", "openai/gpt-4.1-mini")

    response = agent("What is the capital of France?")
    print(response)  # "The capital of France is Paris."
    ```

The string shorthand also works when reassigning the model after construction:

```python
agent.model = "groq/llama-3.1-8b-instant"
```
