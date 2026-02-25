# Streaming

When `stream=True` is set, the agent returns an awaitable response object. Use the `consume()` generator to iterate over response chunks as they arrive, enabling real-time output display.

???+ example

    === "Sync"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            config = {"stream": True}

        agent = Assistant()

        response = agent("Tell me a story")

        for chunk in response.consume():
            print(chunk, end="", flush=True)
        ```

    === "Async"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            config = {"stream": True}

        agent = Assistant()

        response = await agent.acall("Tell me a story")

        async for chunk in response.consume():
            print(chunk, end="", flush=True)
        ```
