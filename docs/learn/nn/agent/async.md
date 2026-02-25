# Async

For asynchronous workflows, use the `acall` method instead of `__call__`. This allows the agent to run without blocking, making it ideal for concurrent execution or integration with async frameworks.

???+ example

    ```python
    # pip install msgflux[openai]
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    class Assistant(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    agent = Assistant()

    response = await agent.acall("Tell me a story")
    ```
