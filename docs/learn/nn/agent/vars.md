# Vars

`vars` is the **unified execution variable space**. Is a Mapping (dict-like). The data is transferred between Agent, Tools, and templates. It can be read, updated, and overwritten.

For example, if your Tool returns information that is useful to your application but shouldn't be shared with the agent, you can save that information within `vars`.

Similarly, you can inject extra information into templates to enrich the agent's context.

???+ example "Basic Examples"

    === "Templates"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class BusinessAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            instructions = "The customer's name is {{customer_name}}."
            config = {"verbose": True}

        agent = BusinessAgent()

        vars = {"customer_name": "Clark Kent"}

        response = agent(
            "Help me with a purchase",
            vars=vars
        )
        ```

    === "Tools"

        Tools can access the `vars` using the `@mf.tool_config(inject_vars=True)` decorator. See [inject_vars](tools.md#inject_vars) for more details.

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(inject_vars=True)
        def get_customer_discount(**kwargs) -> str:
            """Get the discount for the current customer."""
            vars = kwargs.get("vars")
            customer_name = vars.get("customer_name", "Guest")
            return f"{customer_name} has a 15% loyalty discount."

        class BusinessAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [get_customer_discount]
            config = {"verbose": True}

        agent = BusinessAgent()

        vars = {"customer_name": "Clark Kent"}

        response = agent(
            "What discount do I have?",
            vars=vars
        )
        ```

    TODO: adicionar mais um exemplo que consuma o vars de mf.Message usando message_fields