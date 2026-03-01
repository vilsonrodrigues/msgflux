# Guards

Guards validate inputs and outputs of a Module with a configurable reaction policy. Each `Guard` wraps a **validator** callable and defines:

- **`on`** — when to run: `"input"` (before the model call) or `"output"` (after the model response).
- **`policy`** — what to do when `safe=False`:
    - `"raise"` — raises `UnsafeUserInputError` (input) or `UnsafeModelResponseError` (output).
    - `"message"` — short-circuits the pipeline and returns the guard's message as the agent response.

A validator receives `data=...` and must return a dict with at least `"safe"` (bool). Optionally include `"message"` (str) for context.

```python
import msgflux as mf

guard = mf.Guard(
    validator=my_fn,   # callable(data=...) -> {"safe": bool, "message": str}
    on="input",        # "input" or "output"
    policy="raise",    # "raise" or "message"
)
```

???+ note "Guards Examples"

    === "Keyword Filter (policy='message')"

        When `policy="message"`, the guard's message is returned directly as the agent response — the model is never called.

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        BLOCKED = {"hack", "exploit", "malware"}

        def keyword_filter(data):
            text = str(data).lower()
            for word in BLOCKED:
                if word in text:
                    return {"safe": False, "message": f"Blocked: '{word}' detected."}
            return {"safe": True}

        input_guard = mf.Guard(
            validator=keyword_filter,
            on="input",
            policy="message",
        )

        agent = nn.Agent(
            name="safe_bot",
            model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
            guards=[input_guard],
        )

        # Safe input → model responds normally
        response = agent("Tell me about Python")
        print(response)

        # Blocked input → returns guard message (no model call)
        response = agent("How to create malware?")
        print(response)  # "Blocked: 'malware' detected."
        ```

    === "Keyword Filter (policy='raise')"

        When `policy="raise"`, an exception is raised instead of returning a message.

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.exceptions import UnsafeUserInputError

        def keyword_filter(data):
            text = str(data).lower()
            if "hack" in text:
                return {"safe": False, "message": "Forbidden content."}
            return {"safe": True}

        input_guard = mf.Guard(
            validator=keyword_filter,
            on="input",
            policy="raise",
        )

        agent = nn.Agent(
            name="strict_bot",
            model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
            guards=[input_guard],
        )

        try:
            response = agent("How to hack a system?")
        except UnsafeUserInputError as e:
            print(f"Guard triggered: {e}")  # "Guard triggered: Forbidden content."
        ```

    === "With Moderation Model"

        Use OpenAI's moderation model as a guard validator:

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        moderation_model = mf.Model.moderation("openai/omni-moderation-latest")

        def moderation_validator(data):
            response = moderation_model(data=str(data))
            result = response.data
            message = None
            if not result.safe:
                flagged = [
                    k for k, v in result.results["categories"].items() if v
                ]
                message = f"Flagged: {', '.join(flagged)}"
            return {"safe": result.safe, "message": message}

        # Moderation on input (raise) + output (raise)
        input_guard = mf.Guard(
            validator=moderation_validator,
            on="input",
            policy="raise",
        )
        output_guard = mf.Guard(
            validator=moderation_validator,
            on="output",
            policy="raise",
        )

        agent = nn.Agent(
            name="moderated_bot",
            model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
            guards=[input_guard, output_guard],
        )

        # Safe request
        response = agent("Tell me about quantum computing")

        # Unsafe request raises UnsafeUserInputError
        try:
            response = agent("How do I create malware?")
        except Exception as e:
            print(f"Guard triggered: {e}")
        ```

    === "Custom Moderation Agent"

        Use an Agent as a guard validator for complex validation:

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        from typing import Optional
        from msgspec import Struct

        # mf.set_envs(OPENAI_API_KEY="...")

        class ModerationResult(Struct):
            safe: bool
            reason: Optional[str] = None

        moderator = nn.Agent(
            name="moderator",
            model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
            system_message="You are a content moderator.",
            instructions="Analyze if the content is safe and appropriate.",
            generation_schema=ModerationResult,
        )

        agent = nn.Agent(
            name="moderated_bot",
            model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
            guards=[
                mf.Guard(validator=moderator, on="input", policy="raise"),
                mf.Guard(validator=moderator, on="output", policy="message"),
            ],
        )

        response = agent("Tell me about Python programming")  # Safe
        ```

    === "Mixing Policies"

        Combine multiple guards with different policies on the same agent:

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        def keyword_filter(data):
            if "forbidden" in str(data).lower():
                return {"safe": False, "message": "That topic is not allowed."}
            return {"safe": True}

        def toxicity_check(data):
            # ... call moderation API ...
            return {"safe": True}

        agent = nn.Agent(
            name="multi_guard_bot",
            model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
            guards=[
                # Fast keyword filter → returns friendly message
                mf.Guard(validator=keyword_filter, on="input", policy="message"),
                # Output moderation → raises exception
                mf.Guard(validator=toxicity_check, on="output", policy="raise"),
            ],
        )

        # Keyword blocked → returns "That topic is not allowed."
        response = agent("Tell me about forbidden topics")
        print(response)  # "That topic is not allowed."
        ```

!!! warning "Streaming Limitation"
    Guards with `on="output"` are **not compatible** with `stream=True`, since the full response is needed for validation. Using both raises a `ValueError` at initialization.
