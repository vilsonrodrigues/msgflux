# ChatMessages

`ChatMessages` is a unified container for conversation history. It replaces manual list management with a structured abstraction that supports session isolation, turn tracking, dual-format output, and serialization.

## Why ChatMessages?

Managing conversation history as a plain `list[dict]` works for simple cases, but breaks down quickly:

| Problem | Plain list | ChatMessages |
|---------|-----------|--------------|
| Multi-turn tracking | Manual index bookkeeping | `begin_turn()` / `end_turn()` |
| Format conversion | Write adapters yourself | `to_chatml()` / `to_responses_input()` |
| Session isolation | Separate lists per session | Built-in `session_id` + context propagation |
| Branching conversations | Deep copy + slice | `fork(upto_turn=N)` |
| Serialization | Custom JSON logic | `_to_state()` / `_hydrate_state()` |

## Quick Start

```python
import msgflux as mf

# Create and populate
chat = mf.ChatMessages()
chat.add_system("You are a helpful assistant.")
chat.add_user("What is the capital of France?")
chat.add_assistant("The capital of France is Paris.")

# Use with Agent
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4.1-mini")
agent = nn.Agent("Assistant", model)

response = agent("What about Germany?", messages=chat)
```

## Creating ChatMessages

```python
# Empty
chat = mf.ChatMessages()

# From existing ChatML messages
chat = mf.ChatMessages([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
])

# From ChatML (classmethod)
chat = mf.ChatMessages.from_chatml(existing_messages)

# With session metadata
chat = mf.ChatMessages(session_id="user_123", namespace="support")
```

## Adding Content

### Text Messages

```python
chat.add_user("What is quantum computing?")
chat.add_assistant("Quantum computing uses quantum bits...")
chat.add_system("You are a physics tutor.")
chat.add_message("developer", "Focus on practical examples.")
```

### Tool Interactions

```python
# Tool result (from function calling)
chat.add_tool(call_id="call_abc123", content="Temperature: 24°C")
```

### Reasoning

```python
# Explicit reasoning (chain-of-thought)
chat.add_reasoning("The user is asking about weather, I should use the tool.")

# Combined: reasoning + response
chat.add_assistant_response(
    "It's 24°C and sunny in Paris.",
    reasoning_content="I retrieved the weather data using the tool."
)
```

### Bulk Operations

```python
# Add multiple ChatML messages
chat.add_chatml([
    {"role": "user", "content": "First question"},
    {"role": "assistant", "content": "First answer"},
])

# Add Responses API items
chat.add_response_items([
    {"type": "message", "role": "user", "content": "Hello"},
    {"type": "function_call", "call_id": "call_1", "name": "search", "arguments": "{}"},
])
```

## Using with Agent

`ChatMessages` works as a drop-in replacement for `messages` lists. The Agent automatically manages the turn lifecycle and appends the response to the conversation history.

By default, the Agent captures its output into `ChatMessages` (`capture_output=True`). To disable this and manage the response manually:

```python
class MyAgent(nn.Agent):
    model = model
    config = {"capture_output": False}

agent = MyAgent()
chat = mf.ChatMessages()

response = agent("Hello", messages=chat)
# response is NOT added to chat — add it yourself if needed
chat.add_assistant(response)
```

???+ example

    === "Basic Chat"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        class Assistant(nn.Agent):
            model = model
            system_message = "You are a helpful assistant."

        agent = Assistant()
        chat = mf.ChatMessages()

        # Turn 1 — Agent manages turn lifecycle automatically
        response = agent("Hi, I'm Alice.", messages=chat)

        # Turn 2 — agent remembers Alice (history is in chat)
        response = agent("What's my name?", messages=chat)
        # "Your name is Alice."
        ```

    === "ChatBot Loop"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class ChatBot(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You are a helpful assistant."

        agent = ChatBot()
        chat = mf.ChatMessages(session_id="user_42")

        while True:
            user_input = input("You: ")
            if user_input.lower() in ("quit", "exit"):
                break

            response = agent(user_input, messages=chat)
            print(f"Assistant: {response}")
        ```

    === "With Tools"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        def get_weather(city: str) -> str:
            """Get the current weather for a city."""
            return f"Sunny, 24°C in {city}"

        class WeatherBot(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [get_weather]

        agent = WeatherBot()
        chat = mf.ChatMessages()

        # Agent handles turn lifecycle and tool call messages
        response = agent("What's the weather in Paris?", messages=chat)
        ```

## Reasoning with Groq

When using models that produce reasoning (chain-of-thought), `ChatMessages` automatically captures the reasoning alongside the response. The Agent handles `begin_turn` / `end_turn` and reasoning extraction transparently.

```python
import msgflux as mf
import msgflux.nn as nn

# mf.set_envs(GROQ_API_KEY="...")

model = mf.Model.chat_completion(
    "groq/openai/gpt-oss-20b",
    reasoning_effort="low",
    max_tokens=2048,
)

class MathAgent(nn.Agent):
    model = model
    system_message = "You are a concise math assistant. Answer briefly."

agent = MathAgent()
chat = mf.ChatMessages(session_id="demo_session")

# Turn 1
response = agent("What is the integral of x^2?", messages=chat)
# {'answer': '\\int x^2 dx = x^3/3 + C', 'think': 'Need integral of x^2: ...'}

# Turn 2 — follow-up, agent has full context
response = agent("Now differentiate the result to verify.", messages=chat)
# {'answer': 'd/dx(x^3/3 + C) = x^2', 'think': 'We need derivative of x^3/3 + C = x^2.'}

# Inspect turn lifecycle
print(len(chat.turns))        # 2
print(chat.turns[0]["status"]) # 'completed'
print(chat.turns[0]["inputs"]) # 'What is the integral of x^2?'

# Reasoning is captured as a normalized item
for item in chat._items:
    if item.get("type") == "reasoning":
        print(item["reasoning_content"])
        # 'Need integral of x^2: ∫ x^2 dx = x^3/3 + C.'
        break
```

!!! info "How reasoning is captured"
    The Agent extracts reasoning from the model response (fields: `reasoning_content`, `reasoning_text`, or `think`) and stores it as a separate reasoning item via `add_assistant_response`. Both the reasoning and the answer are visible in `to_chatml()` output.

## Dual Format Output

ChatMessages stores items in a normalized internal format and converts on demand.

### ChatML (Chat Completions API)

```python
chatml = chat.to_chatml()
# [
#     {"role": "system", "content": "You are helpful."},
#     {"role": "user", "content": "Hello"},
#     {"role": "assistant", "content": "Hi there!"},
# ]
```

### Responses API

```python
responses = chat.to_responses_input()
# [
#     {"type": "message", "role": "system", "content": "You are helpful."},
#     {"type": "message", "role": "user", "content": "Hello"},
#     {"type": "message", "role": "assistant", "content": "Hi there!"},
# ]
```

Both formats handle automatic conversion of:

- Function calls ↔ `tool_calls` / `function_call`
- Tool results ↔ `function_call_output`
- Reasoning items
- Multimodal content parts

!!! info "Turn Markers"
    Turn markers (`begin_turn` / `end_turn`) are stored in `_items` for tracking but are automatically filtered out in both `to_chatml()` and `to_responses_input()`.

## Turn Lifecycle

Turns track the lifecycle of a user-agent interaction: what was the input, what was the output, and what happened in between.

```python
chat = mf.ChatMessages(session_id="user_42")

# Start a turn
turn_id = chat.begin_turn(
    inputs={"question": "What is AI?"},
    vars={"temperature": 0.7},
    metadata={"source": "web"},
)

# Add messages during the turn
chat.add_user("What is AI?")
chat.add_assistant("AI is artificial intelligence...")

# End the turn
turn_record = chat.end_turn(
    assistant_output="AI is artificial intelligence...",
    response_type="text",
)

# Access turn history
print(chat.turns)       # List of all turns (deep copies)
print(turn_record)      # The completed turn record
```

### Turn Record Fields

| Field | Description |
|-------|-------------|
| `turn_id` | Unique identifier (auto-generated or custom) |
| `session_id` | Session this turn belongs to |
| `started_at` / `ended_at` | ISO timestamps |
| `status` | `in_progress`, `completed`, or `interrupted` |
| `inputs` | What the user asked |
| `assistant_output` | What the agent answered |
| `vars` | Variables active during this turn |
| `metadata` | Custom metadata |

!!! tip
    If you call `begin_turn()` while another turn is active, the previous turn is automatically ended with `status="interrupted"`.

## Session Management

Sessions group related conversations using `contextvars` for implicit propagation.

### Context Manager

```python
# All ChatMessages created inside inherit the session
with mf.ChatMessages.session_context(session_id="user_42", namespace="support"):
    chat1 = mf.ChatMessages()  # session_id="user_42", namespace="support"
    chat2 = mf.ChatMessages()  # same session

# Nesting works
with mf.ChatMessages.session_context(session_id="outer"):
    with mf.ChatMessages.session_context(session_id="inner"):
        chat = mf.ChatMessages()  # session_id="inner"
    chat = mf.ChatMessages()      # session_id="outer"
```

### Explicit Configuration

```python
chat = mf.ChatMessages()
chat.configure_session(session_id="user_42", namespace="support")
```

### Query Context

```python
with mf.ChatMessages.session_context(session_id="s1"):
    ctx = mf.ChatMessages.get_session_context()
    # {"session_id": "s1", "namespace": None}
```

## Fork

Create a copy of the conversation, optionally truncated to a specific turn.

```python
chat = mf.ChatMessages()

# Turn 0
chat.begin_turn(inputs="What is AI?")
chat.add_user("What is AI?")
chat.add_assistant("Artificial intelligence is...")
chat.end_turn(assistant_output="Artificial intelligence is...")

# Turn 1
chat.begin_turn(inputs="Tell me more about ML")
chat.add_user("Tell me more about ML")
chat.add_assistant("Machine learning is a subset...")
chat.end_turn(assistant_output="Machine learning is a subset...")

# Fork up to turn 0 — only the first Q&A
branch = chat.fork(upto_turn=0)
len(branch.turns)  # 1
# Continue the branch in a different direction
branch.add_user("What about deep learning?")

# Full fork (independent copy)
full_copy = chat.fork()
```

## Serialization

`ChatMessages` supports serialization for persistence and recovery via `_to_state()` and `_hydrate_state()`.

```python
# Save state
state = chat._to_state()
# state is a dict with: items, metadata, turns, session_id, namespace, etc.

# Restore state
restored = mf.ChatMessages()
restored._hydrate_state(state)
```

!!! info "Future: Checkpoint Integration"
    In a future release, `ChatMessages` will integrate with `CheckpointStore` for automatic persistence. The `_to_state()` / `_hydrate_state()` API is designed to support this.

## Generating Examples

Convert conversation turns into `Example` objects for few-shot prompting or evaluation.

```python
chat = mf.ChatMessages()

chat.begin_turn(inputs={"question": "2+2?"})
chat.add_user("2+2?")
chat.add_assistant("4")
chat.end_turn(assistant_output="4")

chat.begin_turn(inputs={"question": "3+3?"})
chat.add_user("3+3?")
chat.add_assistant("6")
chat.end_turn(assistant_output="6")

examples = chat.to_examples()
# [
#     Example(inputs={"question": "2+2?", "history": [...]}, labels={"response": "4"}),
#     Example(inputs={"question": "3+3?", "history": [...]}, labels={"response": "6"}),
# ]

# Without conversation history
examples = chat.to_examples(include_history=False)

# Custom field names
examples = chat.to_examples(history_key="context", output_key="answer")
```

## Multimodal Content

Build multimodal messages with images, audio, video, and files.

```python
# Add multimodal user message
chat.add_user_multimodal(
    text="Describe this image",
    media={"image": "https://example.com/photo.jpg"},
)

# Build content blocks manually
content = mf.ChatMessages.build_multimodal_content(
    text="What do you see?",
    media={
        "image": ["photo1.jpg", "photo2.jpg"],
        "audio": "recording.wav",
    },
)
```

## List-Like Interface

`ChatMessages` behaves like a list for familiar operations:

```python
chat = mf.ChatMessages([
    {"role": "user", "content": "a"},
    {"role": "assistant", "content": "b"},
])

len(chat)       # 2
chat[0]         # {"role": "user", "content": "a"}
bool(chat)      # True
list(chat)      # iterate over items

# Mutation
chat.append({"role": "user", "content": "c"})
chat.insert(0, {"role": "system", "content": "Be helpful."})
chat.extend([{"role": "user", "content": "d"}])
copied = chat.copy()

# Metadata
chat.update_metadata({"model": "gpt-4.1", "tokens": 150})
chat.set_response_id("resp_abc123")
```

## Normalization

Items are normalized on insertion. Key behaviors:

- **Reasoning extraction**: Assistant messages with `reasoning_content`, `reasoning_text`, or `think` fields are split into a separate reasoning item + the clean message
- **Reasoning type**: Items with `type="reasoning"` are normalized to `{type, role, reasoning_content}`
- **Empty reasoning**: Items with empty reasoning content are dropped
- **Deep copy**: All items are deep-copied on insertion to prevent external mutation

```python
# This single append produces TWO items
chat.append({
    "role": "assistant",
    "content": "The answer is 42.",
    "reasoning_content": "Let me think step by step..."
})

len(chat)  # 2
chat[0]    # {"type": "reasoning", "role": "assistant", "reasoning_content": "Let me think..."}
chat[1]    # {"role": "assistant", "content": "The answer is 42."}
```
