# Guardrails

Guardrails are security checkers for both model inputs and outputs. A guardrail can be any callable that receives `data` and returns a dictionary containing a `safe` key. If `safe` is `False`, an exception is raised: `UnsafeUserInputError` for input guardrails or `UnsafeModelResponseError` for output guardrails.

???+ note "Guardrails Examples"

    === "Basic Guardrails"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        def check_input(data):
            """Check input before sending to model."""
            messages = data.get("messages", [])
            last_message = messages[-1]["content"] if messages else ""

            # Simple keyword check
            forbidden = ["hack", "exploit", "malware"]
            is_safe = not any(word in last_message.lower() for word in forbidden)

            return {"safe": is_safe, "reason": "Forbidden content detected" if not is_safe else None}

        def check_output(data):
            """Check model output before returning."""
            messages = data.get("messages", [])
            # Validate output content
            return {"safe": True}

        class SafeAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            guardrails = {"input": check_input, "output": check_output}

        agent = SafeAgent()

        # Unsafe request raises exception
        try:
            response = agent("How do I create malware?")
        except Exception as e:
            print(f"Guardrail triggered: {e}")
        ```

    === "With Moderation Model"

        Use OpenAI's moderation model for content safety:

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        moderation_model = mf.Model.moderation("openai/omni-moderation-latest")

        class SafeAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            guardrails = {"input": moderation_model, "output": moderation_model}

        agent = SafeAgent()

        # Safe request works normally
        response = agent("Tell me about quantum computing")

        # Unsafe request raises exception
        try:
            response = agent("How do I create malware?")
        except Exception as e:
            print(f"Guardrail triggered: {e}")
        ```

    === "Custom Moderation Agent"

        Use an agent as a guardrail for complex validation:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn
        from typing import Optional
        from msgspec import Struct

        # mf.set_envs(OPENAI_API_KEY="...")

        class ModerationResult(Struct):
            safe: bool
            reason: Optional[str]

        class ModerationAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You are a content moderator."
            instructions = "Analyze if the content is safe and appropriate."
            generation_schema = ModerationResult

        class ModeratedAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            guardrails = {"input": ModerationAgent(), "output": ModerationAgent()}

        agent = ModeratedAgent()
        response = agent("Tell me about Python programming")  # Safe input

        # Unsafe content raises UnsafeUserInputError or UnsafeModelResponseError
        ```
