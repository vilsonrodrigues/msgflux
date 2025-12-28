# Chat Completion

The `chat_completion` model is the most versatile model type for natural language interactions. It processes messages in conversational format and supports advanced features like multimodal input/output, structured generation, and tool calling.

**All code examples use the recommended import pattern:**

```python
import msgflux as mf
```

## Overview

Chat completion models are stateless - they don't maintain conversation history between calls. You must provide all context (previous messages, system prompt, etc.) in each request.

### Quick Start

```python
import msgflux as mf

# Create model
model = mf.Model.chat_completion("openai/gpt-4o")

# Simple completion
response = model(messages=[{"role": "user", "content": "Hello!"}])
print(response.consume())  # "Hello! How can I help you today?"

# With system prompt
response = model(
    messages=[{"role": "user", "content": "What is AI?"}],
    system_prompt="You are a helpful assistant."
)
print(response.consume())
```

## Model Initialization

### Basic Parameters

```python
import msgflux as mf

model = mf.Model.chat_completion(
    "openai/gpt-4o",
    temperature=0.7,        # Randomness (0-2)
    max_tokens=1000,        # Maximum output tokens
    top_p=0.9,              # Nucleus sampling
    enable_cache=True,      # Enable response caching
    cache_size=128          # Cache size
)
```

### Model Information

```python
# Get model metadata
print(model.get_model_info())
# {'model_id': 'gpt-4o', 'provider': 'openai'}

print(model.instance_type())
# {'model_type': 'chat_completion'}

# Serialize model state
state = model.serialize()
print(state)
# {
#     'msgflux_type': 'model',
#     'provider': 'openai',
#     'model_type': 'chat_completion',
#     'state': {...}
# }
```

## Model Profiles

Model profiles provide metadata about capabilities, pricing, and limits from [models.dev](https://models.dev):

### Getting Profile Information

```python
import msgflux as mf
from msgflux.models.profiles import get_model_profile

# Get profile for a model
profile = get_model_profile("gpt-4o", provider_id="openai")

if profile:
    # Check capabilities
    print(f"Tool calling: {profile.capabilities.tool_call}")
    print(f"Structured output: {profile.capabilities.structured_output}")
    print(f"Reasoning: {profile.capabilities.reasoning}")

    # Check modalities
    print(f"Input: {profile.modalities.input}")   # ['text', 'image']
    print(f"Output: {profile.modalities.output}") # ['text']

    # Check limits
    print(f"Context window: {profile.limits.context}")  # 128000
    print(f"Max output: {profile.limits.output}")       # 16384

    # Check pricing
    print(f"Input: ${profile.cost.input_per_million}/M tokens")
    print(f"Output: ${profile.cost.output_per_million}/M tokens")
```

### Cost Calculation

```python
from msgflux.models.profiles import get_model_profile

profile = get_model_profile("gpt-4o", provider_id="openai")

if profile:
    # Calculate cost for a request
    cost = profile.cost.calculate(
        input_tokens=1000,
        output_tokens=500
    )
    print(f"Estimated cost: ${cost:.4f}")
```

### Profile-Based Model Selection

```python
import msgflux as mf
from msgflux.models.profiles import get_model_profile

def select_model_by_budget(max_cost_per_million_output):
    """Select cheapest model within budget that supports tools."""
    models = [
        ("gpt-4o", "openai"),
        ("claude-3-5-sonnet-20241022", "anthropic"),
        ("gemini-2.0-flash-exp", "google")
    ]

    for model_id, provider in models:
        profile = get_model_profile(model_id, provider_id=provider)
        if profile:
            if (profile.capabilities.tool_call and
                profile.cost.output_per_million <= max_cost_per_million_output):
                return f"{provider}/{model_id}"

    return None

# Select model under $5 per million output tokens
model_path = select_model_by_budget(5.0)
if model_path:
    model = mf.Model.chat_completion(model_path)
```

## Response Caching

Response caching avoids redundant API calls by caching identical requests:

### Enabling Cache

```python
import msgflux as mf

# Enable cache on initialization
model = mf.Model.chat_completion(
    "openai/gpt-4o",
    enable_cache=True,   # Enable caching
    cache_size=128       # Cache up to 128 responses
)

# First call - hits API
response1 = model(messages=[{"role": "user", "content": "Hello"}])
print(response1.consume())

# Second identical call - returns cached response (no API call)
response2 = model(messages=[{"role": "user", "content": "Hello"}])
print(response2.consume())

# Different call - hits API again
response3 = model(messages=[{"role": "user", "content": "Hi"}])
print(response3.consume())
```

### Cache Statistics

```python
import msgflux as mf

model = mf.Model.chat_completion(
    "openai/gpt-4o",
    enable_cache=True,
    cache_size=128
)

# Make some calls
model(messages=[{"role": "user", "content": "Test 1"}])
model(messages=[{"role": "user", "content": "Test 1"}])  # Cache hit
model(messages=[{"role": "user", "content": "Test 2"}])

# Check cache stats
if model._response_cache:
    stats = model._response_cache.cache_info()
    print(stats)
    # {
    #     'hits': 1,
    #     'misses': 2,
    #     'maxsize': 128,
    #     'currsize': 2
    # }

    # Clear cache
    model._response_cache.cache_clear()
```

### Cache Behavior

The cache is sensitive to:
- Message content
- System prompt
- Temperature and sampling parameters
- Generation schema
- Tool schemas

Changing any of these creates a new cache entry.

## Message Formats

### Simple String

```python
response = model(
    messages="What is Python?",
    system_prompt="You are a programming expert."
)
```

### ChatML Format

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Tell me a joke."}
]

response = model(messages=messages)
```

### ChatBlock Format

```python
import msgflux as mf

# Text only
messages = [
    mf.ChatBlock.user("What's in this image?")
]

# With images
messages = [
    mf.ChatBlock.user(
        "Describe this image",
        media=mf.ChatBlock.image("https://example.com/image.jpg")
    )
]

# Multiple media
messages = [
    mf.ChatBlock.user(
        "Compare these images",
        media=[
            mf.ChatBlock.image("https://example.com/image1.jpg"),
            mf.ChatBlock.image("https://example.com/image2.jpg")
        ]
    )
]

response = model(messages=messages)
```

## Streaming

Stream tokens as they're generated:

### Basic Streaming

```python
import msgflux as mf

model = mf.Model.chat_completion("openai/gpt-4o")

response = model(
    messages=[{"role": "user", "content": "Count to 10"}],
    stream=True
)

# Consume stream
async for chunk in response.consume():
    print(chunk, end="", flush=True)
```

### Streaming with FastAPI

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import msgflux as mf

app = FastAPI()
model = mf.Model.chat_completion("openai/gpt-4o")

@app.get("/chat")
async def chat(query: str):
    response = model(
        messages=[{"role": "user", "content": query}],
        stream=True
    )

    return StreamingResponse(
        response.consume(),
        media_type="text/plain"
    )
```

## Async Support

Async version for concurrent operations:

```python
import msgflux as mf
import asyncio

model = mf.Model.chat_completion("openai/gpt-4o")

async def get_completion(prompt):
    response = await model.acall(messages=[{"role": "user", "content": prompt}])
    return response.consume()

# Run multiple completions concurrently
async def main():
    results = await asyncio.gather(
        get_completion("What is AI?"),
        get_completion("What is ML?"),
        get_completion("What is DL?")
    )
    for result in results:
        print(result)

asyncio.run(main())
```

## Multimodal Inputs

Modern models support multiple input modalities:

### Image Understanding

```python
import msgflux as mf

model = mf.Model.chat_completion("openai/gpt-4o")

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.jpg"
            }
        }
    ]
}]

response = model(messages=messages)
print(response.consume())
```

### Using ChatBlock Helper

```python
import msgflux as mf

model = mf.Model.chat_completion("openai/gpt-4o")

messages = [
    mf.ChatBlock.user(
        "Describe this image",
        media=mf.ChatBlock.image("https://example.com/image.jpg")
    )
]

response = model(messages=messages)
print(response.consume())
```

### Base64 Images

```python
import msgflux as mf
import base64

# Read and encode image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data}"
            }
        }
    ]
}]

model = mf.Model.chat_completion("openai/gpt-4o")
response = model(messages=messages)
```

## Structured Generation

Generate structured data conforming to a schema:

### Basic Schema

```python
import msgflux as mf
from msgspec import Struct

class CalendarEvent(Struct):
    name: str
    date: str
    participants: list[str]

model = mf.Model.chat_completion("openai/gpt-4o")

response = model(
    messages="Alice and Bob are going to a science fair on Friday.",
    system_prompt="Extract the event information.",
    generation_schema=CalendarEvent
)

event = response.consume()
print(event)
# {'name': 'science fair', 'date': 'Friday', 'participants': ['Alice', 'Bob']}
```

### Nested Schemas

```python
import msgflux as mf
from msgspec import Struct

class Address(Struct):
    street: str
    city: str
    country: str

class Person(Struct):
    name: str
    age: int
    address: Address

model = mf.Model.chat_completion("openai/gpt-4o")

response = model(
    messages="John Doe, 30 years old, lives at 123 Main St, New York, USA.",
    system_prompt="Extract person information.",
    generation_schema=Person
)

person = response.consume()
print(person)
# {
#     'name': 'John Doe',
#     'age': 30,
#     'address': {
#         'street': '123 Main St',
#         'city': 'New York',
#         'country': 'USA'
#     }
# }
```

### Generation Schemas for Planning

```python
import msgflux as mf

# Access built-in planning schemas
ChainOfThoughts = mf.generation.plan.ChainOfThoughts
ReAct = mf.generation.plan.ReAct
SelfConsistency = mf.generation.plan.SelfConsistency

model = mf.Model.chat_completion("openai/gpt-4o")

# Use Chain of Thoughts
response = model(
    messages="What is 25 * 4 + 17?",
    generation_schema=ChainOfThoughts
)

result = response.consume()
print(result)
```

## Typed Parsers (XML)

Alternative to JSON for structured output using typed XML:

### Setup

```bash
pip install msgflux[xml]
```

### Using Typed XML

```python
import msgflux as mf
from jinja2 import Template

# Get typed XML parser
typed_xml = mf.dsl.typed_parsers.typed_parser_registry["typed_xml"]

# Define instructions
instructions = """Extract the event information:
name: str
date: str
participants: list[str]
"""

# Create system prompt
template = Template(typed_xml.template)
system_prompt = template.render({"instructions": instructions}).strip()

# Generate with XML output
model = mf.Model.chat_completion("openai/gpt-4o")

response = model(
    messages="Alice and Bob are going to a science fair on Friday.",
    system_prompt=system_prompt,
    typed_parser="typed_xml"
)

# Automatically parsed to dict
result = response.consume()
print(result)
# {'name': 'science fair', 'date': 'Friday', 'participants': ['Alice', 'Bob']}
```

## Tool Calling

Models can suggest calling functions (tools) to gather information:

### Defining Tools

```python
import msgflux as mf

# Define tool schema
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g. Paris, France"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"],
            "additionalProperties": False
        }
    }
}]

model = mf.Model.chat_completion("openai/gpt-4o")

response = model(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tool_schemas=tools
)

# Get tool calls
tool_call_agg = response.consume()
calls = tool_call_agg.get_calls()

for call in calls:
    print(f"Tool: {call['function']['name']}")
    print(f"Arguments: {call['function']['arguments']}")
# Tool: get_weather
# Arguments: {'location': 'Paris, France', 'unit': 'celsius'}
```

### Tool Choice

Control when and which tools are called:

```python
import msgflux as mf

model = mf.Model.chat_completion("openai/gpt-4o")

# Auto - model decides
response = model(
    messages=[{"role": "user", "content": "What's the weather?"}],
    tool_schemas=tools,
    tool_choice="auto"  # Default
)

# Required - must call at least one tool
response = model(
    messages=[{"role": "user", "content": "What's the weather?"}],
    tool_schemas=tools,
    tool_choice="required"
)

# Specific function - must call this exact function
response = model(
    messages=[{"role": "user", "content": "Paris weather"}],
    tool_schemas=tools,
    tool_choice="get_weather"
)
```

### Tool Call Flow

```python
import msgflux as mf

def get_weather(location, unit="celsius"):
    """Simulate weather API call."""
    return f"The weather in {location} is 22°{unit[0].upper()}"

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]

model = mf.Model.chat_completion("openai/gpt-4o")

# Initial request
messages = [{"role": "user", "content": "What's the weather in Paris?"}]

response = model(messages=messages, tool_schemas=tools)
tool_call_agg = response.consume()

# Execute tool calls
tool_functions = {"get_weather": get_weather}
calls = tool_call_agg.get_calls()

for call in calls:
    func_name = call['function']['name']
    func_args = call['function']['arguments']

    # Execute function
    result = tool_functions[func_name](**func_args)

    # Add result to aggregator
    tool_call_agg.insert_results(call['id'], result)

# Get messages with tool results
tool_messages = tool_call_agg.get_messages()
messages.extend(tool_messages)

# Final response with tool results
final_response = model(messages=messages)
print(final_response.consume())
# "The weather in Paris is currently 22°C."
```

### Streaming with Tools

```python
import msgflux as mf

model = mf.Model.chat_completion("openai/gpt-4o")

response = model(
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tool_schemas=tools,
    stream=True
)

# Tool calls are aggregated during streaming
tool_call_agg = response.consume()

# After stream completes, get calls
calls = tool_call_agg.get_calls()
print(calls)
```

## Prefilling

Force the model to start its response with specific text:

### Basic Prefilling

```python
import msgflux as mf

model = mf.Model.chat_completion("anthropic/claude-3-5-sonnet-20241022")

response = model(
    messages=[{"role": "user", "content": "What is 30 * 3 + 33?"}],
    prefilling="Let's think step by step:"
)

print(response.consume())
# Let's think step by step:
# First, calculate 30 × 3 = 90.
# Then, add 33 to that: 90 + 33 = 123.
# So, the answer is 123.
```

### Prefilling for Format Control

```python
import msgflux as mf

model = mf.Model.chat_completion("anthropic/claude-3-5-sonnet-20241022")

# Force JSON output
response = model(
    messages=[{"role": "user", "content": "List 3 colors"}],
    prefilling="{"
)

print(response.consume())
# {"colors": ["red", "blue", "green"]}
```

## Best Practices

### 1. Use Response Caching for Repeated Queries

```python
# Good - Enable cache for applications with repeated queries
model = mf.Model.chat_completion(
    "openai/gpt-4o",
    enable_cache=True,
    cache_size=256  # Adjust based on usage patterns
)
```

### 2. Check Model Profiles Before Use

```python
from msgflux.models.profiles import get_model_profile

# Good - Verify capabilities before use
profile = get_model_profile("gpt-4o", provider_id="openai")
if profile and profile.capabilities.tool_call:
    model = mf.Model.chat_completion("openai/gpt-4o")
    response = model(messages=messages, tool_schemas=tools)
```

### 3. Reuse Model Instances

```python
# Good - Create once, use many times
model = mf.Model.chat_completion("openai/gpt-4o")

for query in queries:
    response = model(messages=[{"role": "user", "content": query}])
    results.append(response.consume())

# Bad - Creating new instance each time
for query in queries:
    model = mf.Model.chat_completion("openai/gpt-4o")
    response = model(messages=[{"role": "user", "content": query}])
```

### 4. Use Async for Concurrent Requests

```python
import asyncio

async def process_queries(queries):
    model = mf.Model.chat_completion("openai/gpt-4o")

    tasks = [
        model.acall(messages=[{"role": "user", "content": q}])
        for q in queries
    ]

    responses = await asyncio.gather(*tasks)
    return [r.consume() for r in responses]
```

### 5. Monitor Cache Performance

```python
model = mf.Model.chat_completion("openai/gpt-4o", enable_cache=True)

# Periodically check cache stats
if model._response_cache:
    stats = model._response_cache.cache_info()
    hit_rate = stats['hits'] / (stats['hits'] + stats['misses'])
    print(f"Cache hit rate: {hit_rate:.2%}")

    # Clear if hit rate is too low
    if hit_rate < 0.1:
        model._response_cache.cache_clear()
```

## Response Metadata

All responses include metadata with usage information:

```python
import msgflux as mf

model = mf.Model.chat_completion("openai/gpt-4o")

response = model(messages=[{"role": "user", "content": "Hello"}])

# Access metadata
print(response.metadata)
# {
#     'usage': {
#         'completion_tokens': 9,
#         'prompt_tokens': 19,
#         'total_tokens': 28
#     }
# }

# Calculate cost using profile
from msgflux.models.profiles import get_model_profile

profile = get_model_profile("gpt-4o", provider_id="openai")
if profile:
    usage = response.metadata.usage
    cost = profile.cost.calculate(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens
    )
    print(f"Request cost: ${cost:.4f}")
```

## Error Handling

Handle common errors gracefully:

```python
import msgflux as mf

model = mf.Model.chat_completion("openai/gpt-4o")

try:
    response = model(messages=[{"role": "user", "content": "Hello"}])
    result = response.consume()
except ImportError:
    print("Provider not installed")
except ValueError as e:
    print(f"Invalid parameters: {e}")
except Exception as e:
    print(f"API error: {e}")
```

## See Also

- [Model](model.md) - Model factory and registry
- [Text Embeddings](text_embedder.md) - Text embedding models
- [Tool Usage](../tools.md) - Working with tools
- [Generation Schemas](../generation.md) - Planning schemas
