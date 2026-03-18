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
