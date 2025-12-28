# nn.Agent

The `Agent` is a powerful `Module` that uses language models to solve tasks. It can handle multimodal data, call tools, and manage complex workflows with structured outputs.

## Overview

An `Agent` combines a language model with instructions and tools to accomplish tasks. The Agent module adopts a task decomposition strategy, allowing each part of a task to be treated in isolation and independently.

A `ToolLibrary` is integrated into the Agent to manage and execute tools.

### Key Features

- **Multimodal Support**: Handle text, images, audio, video, and files
- **Tool Calling**: Execute functions to interact with external systems
- **Structured Outputs**: Generate typed responses with validation
- **Flexible Configuration**: Customize behavior through message fields and config options
- **Template System**: Use Jinja templates for prompts and responses
- **Modular System Prompt**: Compose system prompts from independent components
- **Task Decomposition**: Break down complex tasks into manageable parts

### Response Modes

The `response_mode` parameter defines how the agent's response is returned:

- **`plain_response`** (default): Returns the output directly to the user
- **Other values**: Log the response to the passed Message object

### Model State

By default, the agent returns only the model output. To also return the agent's internal state (`model_state`), pass `config={"return_model_state": True}`. The output will then be a `dotdict` containing:

- `model_response`: The agent's output
- `model_state`: The internal conversation state

## System Prompt Components

The Agent divides the **system prompt** and **task** into different components for granular control and composability.

### System Prompt Variables

The system prompt is composed of 6 components:

| Component | Description | Example |
|-----------|-------------|---------|
| **system_message** | Agent behavior and role | "You are an agent specialist in..." |
| **instructions** | What the agent should do | "You MUST respond to the user..." |
| **expected_output** | Format of the response | "Your answer must be concise..." |
| **examples** | Input/output examples | Examples of reasoning and outputs |
| **system_extra_message** | Additional system context | Extra instructions or constraints |
| **include_date** | Include current date | Adds "Weekday, Month DD, YYYY" |

All components are assembled using a Jinja template to create the final system prompt.

### Task Variables

The **task** configuration is separated into several parts:

| Variable | Description | Can be passed at call time |
|----------|-------------|---------------------------|
| **context_cache** | Fixed context for the agent | ❌ |
| **context_inputs** | Dynamic context from Message | ✅ |
| **task_inputs** | Main task input from Message | ✅ |
| **task_multimodal_inputs** | Multimodal inputs (image, audio, etc.) | ✅ |
| **task_messages** | ChatML format conversation history | ✅ |
| **vars** | Variables for templates and tools | ✅ |

**(*) Variables marked with ✅ can be passed either via `message_fields` dict or as named arguments during agent call.**

### Configuration Parameters

Agents use three main configuration dictionaries:

#### `message_fields`

Maps Message object paths to agent inputs:

```python
message_fields={
    "task_inputs": "input.user",                           # Task input path
    "task_multimodal_inputs": {"image": "images.user"},   # Multimodal inputs
    "model_state": "messages.history",                     # Conversation history
    "context_inputs": "context.data",                      # Context data
    "model_preference": "model.preference",                # Model selection (ModelGateway)
    "vars": "vars.data"                                    # Template/tool variables
}
```

#### `config`

Controls agent behavior:

```python
config={
    "verbose": True,                              # Print output and tool calls
    "return_model_state": False,                  # Return internal state
    "tool_choice": "auto",                        # Tool selection ("auto", "required", or function name)
    "stream": False,                              # Stream response
    "image_block_kwargs": {"detail": "high"},     # Image processing options
    "video_block_kwargs": {"format": "mp4"},      # Video processing options
    "include_date": False                         # Include date in system prompt
}
```

#### `templates`

Jinja templates for formatting:

```python
templates={
    "task": "Who was {{person}}?",                        # Task formatting
    "response": "{{final_answer}}",                       # Response formatting
    "context": "Context: {{context}}",                    # Context formatting
    "system_prompt": "Custom system prompt template"      # System prompt override
}
```

### Guardrails

Agent supports input/output guardrails via the `guardrails` parameter:

```python
guardrails={
    "input": input_checker,    # Executed before model
    "output": output_checker   # Executed after model
}
```

Both receive a `data` parameter containing conversations in ChatML format. Moderation models are commonly used for guardrails.

## Quick Start

### Basic Usage

```python
import msgflux as mf
import msgflux.nn as nn

# Set API key
mf.set_envs(OPENAI_API_KEY="sk-...")

# Create model
model = mf.Model.chat_completion("openai/gpt-4o")

# Create agent (requires at least name and model)
agent = nn.Agent("assistant", model)

# Use agent
response = agent("What is the capital of France?")
print(response)  # "The capital of France is Paris."
```

### With System Components

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

system_message = """
You are a business development assistant focused on helping sales teams qualify leads
and craft compelling value propositions.
Always keep a professional and persuasive tone.
"""

instructions = """
When given a short company description, identify its potential needs,
suggest an initial outreach strategy, and provide a tailored value proposition.
"""

expected_output = """
Respond in three bullet points:
    - Identified Needs
    - Outreach Strategy
    - Value Proposition
"""

system_extra_message = """
Ensure recommendations align with ethical sales practices
and avoid making unverifiable claims about the product.
"""

sales_agent = nn.Agent(
    "sales-agent",
    model,
    system_message=system_message,
    instructions=instructions,
    expected_output=expected_output,
    system_extra_message=system_extra_message,
    config={"include_date": True, "verbose": True}
)

# View the generated system prompt
print(sales_agent._get_system_prompt())

# Use the agent
response = sales_agent("A fintech startup offering digital wallets")
print(response)
```

## Debugging an Agent

### View State Dictionary

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")
agent = nn.Agent("agent", model)

# Prompt components and config can be viewed through state dict
agent.state_dict()
```

### Inspect Model Execution

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

agent = nn.Agent(
    "agent",
    model,
    instructions="User name is {{user_name}}",
    config={"return_model_state": True}
)

message = "Hi"
vars = {"user_name": "Clark"}

# Inspect what will be sent to the model
execution_params = agent.inspect_model_execution_params(message, vars=vars)
print(execution_params)

# Execute
response = agent(message, vars=vars)
print(response)
```

### Verbose Mode

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

# Enable verbose to see model outputs and tool calls
agent = nn.Agent("agent", model, config={"verbose": True})

response = agent("Tell me a joke")
```

## Using AutoParams

The `Agent` class already includes AutoParams support, allowing you to create agent variants declaratively by defining class attributes. This is especially useful for creating agent families with different configurations.

### Basic AutoParams Usage

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

# Create agent variants by setting class attributes
class CreativeAgent(nn.Agent):
    """Agent optimized for creative tasks."""
    name = "creative-agent"
    system_message = "You are a creative assistant."
    instructions = "Generate innovative and original ideas."
    config = {"verbose": False}

class AnalyticalAgent(nn.Agent):
    """Agent optimized for analytical tasks."""
    name = "analytical-agent"
    system_message = "You are an analytical assistant."
    instructions = "Provide data-driven insights."
    config = {"verbose": False}

# Instantiate with defaults
creative = CreativeAgent(model=model)
analytical = AnalyticalAgent(model=model)

# Or override specific parameters
custom_creative = CreativeAgent(
    model=model,
    config={"verbose": True}  # Override config
)
```

### Automatic Name and Description Capture

Agent automatically captures the class name as `name` and the docstring as `description`:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

class CustomerSupportAgent(nn.Agent):
    """
    Helpful customer support agent specialized in resolving user issues
    with empathy and efficiency. Use for customer inquiries, complaints,
    and support requests.
    """

    system_message = "You are a professional customer support agent."
    instructions = "Listen to customer concerns and provide clear solutions."
    config = {"verbose": False}

class SalesAgent(nn.Agent):
    """
    Persuasive sales agent focused on understanding customer needs and
    presenting tailored solutions. Use for lead qualification, product
    demonstrations, and closing deals.
    """

    system_message = "You are an expert sales professional."
    instructions = "Identify customer needs and present value propositions."
    config = {"verbose": False}

# Name and description are automatically captured
support = CustomerSupportAgent(model=model)
print(support.name)  # "CustomerSupportAgent"
print(support.description)  # "Helpful customer support agent specialized..."

sales = SalesAgent(model=model)
print(sales.name)  # "SalesAgent"
print(sales.description)  # "Persuasive sales agent focused..."
```

### Agent Families with Shared Configuration

Create families of agents that share common settings:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

# Base configuration for all content moderators
class ModeratorBase(nn.Agent):
    """Base moderator configuration."""

    system_message = "You are a content moderator."

    # Shared defaults
    instructions = "Review content and flag violations."
    expected_output = "Return: APPROVED or FLAGGED with reason."

class StrictModerator(ModeratorBase):
    """Strict content moderator. Flag any potentially harmful content."""

    system_message = "You are a strict content moderator with low tolerance for violations."
    config = {"verbose": True}

class LenientModerator(ModeratorBase):
    """Lenient content moderator. Only flag clearly harmful content."""

    system_message = "You are a lenient content moderator focusing only on clear violations."
    config = {"verbose": False}

class ChildSafeModerator(StrictModerator):
    """Ultra-strict moderator for child-safe content. Zero tolerance for inappropriate material."""

    system_message = "You are an ultra-strict child safety moderator."
    expected_output = "Return: APPROVED or BLOCKED with detailed reason."

# All inherit shared configuration and auto-capture name/description
strict = StrictModerator(model=model)
lenient = LenientModerator(model=model)
child_safe = ChildSafeModerator(model=model)

print(strict.name)  # "StrictModerator"
print(strict.description)  # "Strict content moderator. Flag any..."
print(lenient.name)  # "LenientModerator"
```

### Multi-Language Agents

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

class TranslatorBase(nn.Agent):
    """Base translator agent."""

    system_message = "You are a professional translator."
    instructions = "Translate the input text accurately."

class EnglishToSpanish(TranslatorBase):
    """Professional translator from English to Spanish."""

    system_message = "You are a professional English to Spanish translator."
    config = {"verbose": False}

class SpanishToEnglish(TranslatorBase):
    """Professional translator from Spanish to English."""

    system_message = "You are a professional Spanish to English translator."
    config = {"verbose": False}

class EnglishToFrench(TranslatorBase):
    """Professional translator from English to French."""

    system_message = "You are a professional English to French translator."
    config = {"verbose": False}

# Create translators - names and descriptions are auto-captured
en_to_es = EnglishToSpanish(model=model)
es_to_en = SpanishToEnglish(model=model)
en_to_fr = EnglishToFrench(model=model)

print(en_to_es.name)  # "EnglishToSpanish"
print(en_to_es.description)  # "Professional translator from English to Spanish."

# Use them
result = en_to_es("Hello, how are you?")
print(result)  # "Hola, ¿cómo estás?"
```

### When to Use AutoParams

**Use AutoParams when:**
- ✅ Creating multiple agent variants with different configurations
- ✅ Building agent families with shared defaults
- ✅ You want declarative, class-based configuration
- ✅ Managing reusable agent templates

**Use direct instantiation when:**
- ❌ Creating a single, one-off agent
- ❌ Agent configuration is highly dynamic
- ❌ Simple, straightforward use case

## Async Support

Agents support asynchronous execution using `acall()`:

### Basic Async

```python
import msgflux as mf
import msgflux.nn as nn
import asyncio

model = mf.Model.chat_completion("openai/gpt-4o")
agent = nn.Agent("agent", model)

async def main():
    response = await agent.acall("Tell me about Dirac delta")
    print(response)

asyncio.run(main())
```

### Concurrent Execution

```python
import msgflux as mf
import msgflux.nn as nn
import asyncio

model = mf.Model.chat_completion("openai/gpt-4o")
agent = nn.Agent("agent", model)

async def main():
    tasks = [
        agent.acall("What is quantum computing?"),
        agent.acall("What is machine learning?"),
        agent.acall("What is blockchain?")
    ]

    responses = await asyncio.gather(*tasks)

    for i, response in enumerate(responses, 1):
        print(f"\nResponse {i}:\n{response}")

asyncio.run(main())
```

### Async with Streaming

```python
import msgflux as mf
import msgflux.nn as nn
import asyncio

model = mf.Model.chat_completion("openai/gpt-4o")
agent = nn.Agent("agent", model, config={"stream": True})

async def main():
    response = await agent.acall("Tell me a story")

    # Stream chunks
    async for chunk in response.consume():
        print(chunk, end="", flush=True)

asyncio.run(main())
```

## Streaming

In streaming mode, the agent returns a `ModelStreamResponse` object that can be consumed asynchronously. This mode can be combined with tool usage.

### Basic Streaming

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")
agent = nn.Agent("agent", model, config={"stream": True})

# Get streaming response
response = agent("Tell me a funny story")

print(type(response))  # ModelStreamResponse
print(response.response_type)  # text_generation

# FastAPI StreamingResponse compatible
async for chunk in response.consume():
    print(chunk, end="", flush=True)
```

### Streaming with Tools

```python
import msgflux as mf
import msgflux.nn as nn

def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"The weather in {location} is sunny."

model = mf.Model.chat_completion("openai/gpt-4o")
agent = nn.Agent(
    "agent",
    model,
    tools=[get_weather],
    config={"stream": True, "verbose": True}
)

response = agent("What's the weather in Paris?")

# Stream response
async for chunk in response.consume():
    print(chunk, end="", flush=True)
```

## Vars

Language models act as computers making context-based decisions in an environment. Beyond tool calls, models need to store information in variables.

In **msgFlux**, this is called `vars`.

### What are Vars?

`vars` is a dictionary injected into various agent components:

- `templates["system_prompt"]` - System prompt template
- `templates["task"]` - Task template
- `templates["context"]` - Context template
- `templates["response"]` - Response template
- Tool calls - Tools can access and modify vars

Within tools, `vars` can provide and receive data. Think of it as a set of runtime variables available throughout the agent's execution.

### Using Vars in Templates

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

system_extra_message = """
The customer's name is {{customer_name}}. Treat them politely.
"""

agent = nn.Agent(
    "sales-agent",
    model,
    system_extra_message=system_extra_message
)

# Pass vars at call time
response = agent(
    "Help me with a purchase",
    vars={"customer_name": "Clark Kent"}
)
```

### Vars in Task Templates

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

agent = nn.Agent(
    "agent",
    model,
    instructions="Help the user with whatever they need. Address them by name if provided.",
    templates={
        "task": """
{% if user_name %}
My name is {{ user_name }}.
{% endif %}
{{ user_input }}
"""
    }
)

response = agent(
    message={"user_input": "Who was Nikola Tesla?"},
    vars={"user_name": "Bruce Wayne"}
)
print(response)
```

## Task and Context

A **task** is a specific objective assigned to an agent, consisting of:
- Clear instruction
- Possible restrictions
- Success criterion
- Context in which the task is performed

Language models use **In-Context Learning** (ICL) - the ability to learn new knowledge without updating their parameters.

### Task Input

#### Direct Message

Pass a message directly to the agent:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")
agent = nn.Agent("agent", model, config={"verbose": True})

task = "I need help with my TV."
response = agent(task)
```

#### With Task Template

##### String-based Input

For simple string inputs, use `{}` placeholder:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

agent = nn.Agent(
    "agent",
    model,
    templates={"task": "Who was {}?"}
)

response = agent("Nikola Tesla")
```

##### Dict-based Inputs

For dictionary inputs, use Jinja blocks `{{field_name}}`:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

agent = nn.Agent(
    "agent",
    model,
    templates={"task": "Who was {{person}}?"}
)

response = agent({"person": "Nikola Tesla"})
```

##### Task Template as Fixed Task

If a task template is passed without a task input, it becomes the fixed task:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

# Useful for multimodal apps where prompt doesn't change
agent = nn.Agent(
    "agent",
    model,
    templates={"task": "Who was Nikola Tesla?"}
)

# No message needed
response = agent()
```

##### Combine with Vars

Build dynamic task templates with vars:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

agent = nn.Agent(
    "agent",
    model,
    instructions="Help the user. Address them by name if provided.",
    templates={
        "task": """
{% if user_name %}
My name is {{ user_name }}.
{% endif %}
{{ user_input }}
"""
    }
)

response = agent(
    message={"user_input": "Who was Nikola Tesla?"},
    vars={"user_name": "Bruce Wayne"}
)
```

### Task Messages

Pass conversation history as a list of messages (makes `message` parameter optional):

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")
agent = nn.Agent("agent", model)

# Create chat history
chat = mf.ChatML()
chat.add_user_message("Hi, I'm Peter Parker, a photographer. Recommend some cameras?")

# First response
response = agent(task_messages=chat.get_messages())
print(response)

# Continue conversation
chat.add_assist_message(response)
chat.add_user_message("I need a low-cost, compact camera for freelance work.")

response = agent(task_messages=chat.get_messages())
print(response)
```

### Fixed Messages

Keep a set of pinned conversations within the agent:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")
chat = mf.ChatML()
chat.add_user_message("I'm interested in cameras")
chat.add_assist_message("Great! I can help with that.")

agent = nn.Agent(
    "agent",
    model,
    fixed_messages=chat.get_messages(),
    config={"verbose": True, "return_model_state": True}
)

response = agent("What's the cheapest between Canon PowerShot G9 X Mark II and Sony Cyber-shot DSC-HX80?")
```

### Multimodal Task

Multimodal models can handle images, audio, and files.

**Current support:**

| Media | Single Input | Multiple Inputs |
|-------|--------------|-----------------|
| Image | ✅ | ✅ |
| Audio | ✅ | ❌ |
| File  | ✅ | ❌ |

#### Image Input

Pass local paths or URLs:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

agent = nn.Agent(
    "agent",
    model,
    templates={"task": "Describe this image."}
)

# Single image
response = agent(task_multimodal_inputs={
    "image": "https://example.com/image.jpg"
})

# Multiple images
response = agent(task_multimodal_inputs={
    "image": ["file:///path/to/image1.jpg", "file:///path/to/image2.jpg"]
})
```

#### File Input

Pass raw `.pdf` files (OpenAI and OpenRouter only):

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

agent = nn.Agent(
    "agent",
    model,
    templates={"task": "Summarize the paper"}
)

response = agent(task_multimodal_inputs={
    "file": "https://arxiv.org/pdf/1706.03762.pdf"
})
```

#### Audio Input

Pass raw audio files (OpenAI, vLLM, and OpenRouter only):

```python
import msgflux as mf
import msgflux.nn as nn

# OpenRouter with Gemini supports audio
mf.set_envs(OPENROUTER_API_KEY="...")
model = mf.Model.chat_completion("openrouter/google/gemini-2.0-flash-exp")

agent = nn.Agent(
    "agent",
    model,
    templates={"task": "Please transcribe this audio file."}
)

response = agent(task_multimodal_inputs={
    "audio": "/path/to/audio.wav"
})
print(response)
```

### Context Inputs

`context` is knowledge available to the model at inference time for decision-making, answering questions, or performing actions.

`context_inputs` is knowledge passed to the agent during task definition - from databases, documents, conversation summaries, etc.

#### String-based Context

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

agent = nn.Agent(
    "sales-agent",
    model,
    system_message="You are a sales assistant that helps craft personalized pitches.",
    instructions="Generate responses tailored to the client context.",
    expected_output="Write a short persuasive pitch (max 120 words).",
    system_extra_message="Avoid exaggerations and maintain professionalism.",
    config={"include_date": True, "verbose": True}
)

task = "Can you help me create an initial message for this customer?"

context_inputs = """
Company name: FinData Analytics
Industry: Financial Technology (FinTech)
Product: AI-powered risk analysis platform for banks
Target market: Mid-sized regional banks in South America
Unique value: Automated detection of fraud patterns in real-time
"""

# Inspect what will be sent
execution_params = agent.inspect_model_execution_params(task, context_inputs=context_inputs)
print(execution_params["messages"][0]["content"])

# Execute
response = agent(task, context_inputs=context_inputs)
```

#### List-based Context

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")
agent = nn.Agent("sales-agent", model)

context_inputs = [
    "Company name: DataFlow Analytics",
    "Product: StreamVision — a real-time analytics platform",
    "Key value proposition: Monitor live data streams and detect anomalies instantly",
    "Support policy: 24/7 support for enterprise clients via chat and email"
]

response = agent(
    "Create an initial message for this customer",
    context_inputs=context_inputs
)
```

#### Dict-based Context

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")
agent = nn.Agent("sales-agent", model)

context_inputs = {
    "client_name": "EcoSupply Ltd.",
    "industry": "Sustainable packaging",
    "pain_points": ["High logistics costs", "Need for eco-friendly certification"],
    "current_solution": "Using generic suppliers with limited green compliance"
}

response = agent(
    "Create an initial message for this customer",
    context_inputs=context_inputs
)
```

#### Context Inputs Template

Format `context_inputs` using a Jinja template:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

agent = nn.Agent(
    "sales-agent",
    model,
    system_message="You are a sales assistant that helps craft personalized pitches.",
    instructions="Generate responses tailored to the client context.",
    expected_output="Write a short persuasive pitch (max 120 words).",
    templates={
        "context": """
The client is **{{ client_name }}**, a company in the **{{ industry }}** sector.

They are currently relying on {{ current_solution }},
but face the following main challenges:
{%- for pain in pain_points %}
- {{ pain }}
{%- endfor %}

This background should be considered when tailoring answers.
"""
    },
    config={"include_date": True}
)

context_inputs = {
    "client_name": "EcoSupply Ltd.",
    "industry": "Sustainable packaging",
    "pain_points": ["High logistics costs", "Need for eco-friendly certification"],
    "current_solution": "Using generic suppliers with limited green compliance"
}

task = "Can you help me create an initial message for this customer?"

# Inspect formatted context
execution_params = agent.inspect_model_execution_params(task, context_inputs=context_inputs)
print(execution_params["messages"][0]["content"])

response = agent(task, context_inputs=context_inputs)
```

### Context Cache

`context_cache` stores fixed knowledge within the agent's context block - useful when certain information is always needed before performing a task:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

agent = nn.Agent(
    "agent",
    model,
    context_cache="""
Company values:
- Customer first
- Innovation
- Integrity
"""
)

# Context cache is always included
response = agent("How should I handle this customer complaint?")
```

## Tools

Tools are interfaces that allow language models to perform actions or query information outside the model itself.

### What are Tools?

1. **Function Calling** - A tool is exposed as a function with defined name, parameters, and types
   - Example: `get_weather(location: str, unit: str)`
   - The model decides whether to call it and provides arguments

2. **Extending Capabilities** - Tools allow you to:
   - Search for real-time data (weather, stocks, databases)
   - Perform precise calculations
   - Manipulate systems (send emails, schedule events)
   - Integrate with external APIs

3. **Agent-based Orchestration** - The LLM acts as an agent that decides:
   - When to use a tool
   - Which tool to use
   - How to interpret the tool's output

In msgFlux, a **Tool can be any callable** (function, class with `__call__`, or Agent).

**Note**: While more tools enable more actions, too many tools can confuse the model about which one to use.

### Basic Tool Usage

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

def add(a: float, b: float) -> str:
    """Sum two numbers."""
    c = a + b
    return f"The sum of {a} plus {b} is {c}"

agent = nn.Agent(
    "agent",
    model,
    tools=[add],
    config={"verbose": True}
)

response = agent("What is 5 plus 3?")
print(response)
```

### Web Scraping Agent

```python
import msgflux as mf
import msgflux.nn as nn
import requests
from bs4 import BeautifulSoup

def scrape_website(url: str) -> str:
    """Receive a URL and return the page content."""
    try:
        response = requests.get(url, verify=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style tags
        for tag in soup(["script", "style"]):
            tag.extract()

        text = soup.get_text(separator="\n")
        clean_text = "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )
        return clean_text
    except requests.exceptions.RequestException as e:
        return f"Error accessing {url}: {e}"

# Model with reasoning support
mf.set_envs(GROQ_API_KEY="...")
model = mf.Model.chat_completion("groq/llama-3.3-70b-versatile")

# Agent with scraping tool
scraper_agent = nn.Agent(
    "scraper-agent",
    model,
    tools=[scrape_website],
    config={"return_model_state": True, "verbose": True}
)

site = "https://bbc.com"
response = scraper_agent(f"Summarize the news on this website: {site}")

print(response.model_state)
print(response.model_response)
```

### With Task Template

Simplify repeated patterns with templates:

```python
import msgflux as mf
import msgflux.nn as nn

# Define task template with placeholder
scraper_agent = nn.Agent(
    "scraper",
    model,
    tools=[scrape_website],
    templates={"task": "Summarize the news on this site: {}"},
    config={"verbose": True}
)

# Just pass the URL
response = scraper_agent("https://bbc.com")
print(response)
```

### With Message Mode

Use declarative mode with Message objects:

```python
import msgflux as mf
import msgflux.nn as nn

scraper_agent = nn.Agent(
    "scraper",
    model,
    tools=[scrape_website],
    templates={"task": "Summarize the news on this site: {}"},
    message_fields={"task_inputs": "content"},
    response_mode="summary",
    config={"verbose": True}
)

# Create message
msg = mf.Message(content="https://bbc.com")

# Process through agent
msg = scraper_agent(msg)

# Response is in the message
print(msg)
```

## Agent-as-a-Tool

Agents can be used as tools for other agents, enabling hierarchical task delegation. Using AutoParams makes this pattern especially clean: the class name becomes the tool name, and the docstring becomes the tool description.

### Basic Example with AutoParams

```python
import msgflux as mf
import msgflux.nn as nn

mf.set_envs(OPENAI_API_KEY="...")
model = mf.Model.chat_completion("openai/gpt-4o", tool_choice="auto")

# Specialist agent using AutoParams
# Class name → tool name: "Nutritionist"
# Docstring → tool description (shown to the calling agent)
class Nutritionist(nn.Agent):
    """
    Specialist in nutrition, diets, and meal plans.
    Should be used whenever the user requests:
    - Diet recommendations
    - Personalized meal plans (e.g., gaining muscle, losing weight)
    - Balanced meal suggestions
    - Nutritional guidelines

    Expected inputs:
    - User's goal (e.g., gaining muscle, losing weight)
    - Dietary restrictions or preferences (if provided)
    - Basic info (age, weight, activity level, if available)

    Outputs:
    - Structured and practical meal plan
    - Clear meal suggestions (breakfast, lunch, dinner, snacks)
    - Notes on adjustments if user data is missing

    Restrictions:
    - Not a substitute for medical advice
    - If important info missing, return default plan and indicate needed data
    """

    system_message = "You are a Nutrition Expert Agent."

    instructions = """
    Your responsibilities:
    - Receive instructions from the calling agent about user needs
    - Create a clear and practical meal plan tailored to the stated goal
    - Be objective, technical, and structured
    - Return only the requested result, without greetings or extra explanations

    Restrictions:
    - Don't provide medical recommendations without proper information
    - If data is missing (weight, age, allergies), create a standard plan
      and indicate what additional info would be needed to customize it
    """

# General agent that delegates to specialist
class GeneralSupport(nn.Agent):
    """
    General support agent that handles user requests and delegates to specialists.
    Use for general inquiries that may require expert consultation.
    """

    system_message = "You are a General Support Agent."

    instructions = """
    Your responsibilities:
    - Understand the user's intent
    - Decide if you can respond independently or need to call a tool
    - If using a tool, formulate the request clearly for the expert
    - When the expert responds, format a friendly final response for the user

    Limitations:
    - Don't invent technical information if you're not confident
    - Use the appropriate expert to provide reliable recommendations
    """

    config = {"verbose": True}

# Instantiate agents
nutritionist = Nutritionist(model=model)
generalist = GeneralSupport(model=model, tools=[nutritionist])

# User request
task = "I want a meal plan to gain muscle mass. I'm a 27-year-old man, 1.78m tall."

response = generalist(task)
print(response)
```

### Multiple Specialist Agents

Create a team of specialists with clean, declarative code:

```python
import msgflux as mf
import msgflux.nn as nn

mf.set_envs(OPENAI_API_KEY="...")
model = mf.Model.chat_completion("openai/gpt-4o", tool_choice="auto")

class Nutritionist(nn.Agent):
    """
    Expert in nutrition, diets, and meal planning.
    Use for diet recommendations, meal plans, and nutritional advice.
    """

    system_message = "You are a certified nutritionist."
    instructions = "Provide evidence-based nutritional advice and personalized meal plans."

class FitnessTrainer(nn.Agent):
    """
    Expert in fitness, exercise, and training programs.
    Use for workout routines, exercise form, and training schedules.
    """

    system_message = "You are a certified personal trainer."
    instructions = "Create effective workout programs tailored to user goals and fitness level."

class WellnessCoach(nn.Agent):
    """
    Expert in mental wellness, stress management, and lifestyle optimization.
    Use for stress management, sleep optimization, and work-life balance advice.
    """

    system_message = "You are a wellness and lifestyle coach."
    instructions = "Provide holistic wellness guidance focusing on mental health and lifestyle."

class HealthCoordinator(nn.Agent):
    """
    General health coordinator that orchestrates specialist consultations.
    Analyzes user requests and delegates to appropriate specialists.
    """

    system_message = "You are a health coordinator managing a team of specialists."

    instructions = """
    - Analyze user health and wellness requests
    - Delegate to appropriate specialists (nutrition, fitness, or wellness)
    - Synthesize specialist recommendations into cohesive advice
    - Ensure recommendations are complementary and safe
    """

    config = {"verbose": True}

# Create team
nutritionist = Nutritionist(model=model)
trainer = FitnessTrainer(model=model)
wellness = WellnessCoach(model=model)

coordinator = HealthCoordinator(
    model=model,
    tools=[nutritionist, trainer, wellness]
)

# Complex request requiring multiple specialists
response = coordinator(
    "I want to lose 10kg in 3 months. I'm stressed with work and sleep poorly. "
    "Can you help me with a complete plan?"
)
print(response)
```

### Creating Specialist Variants with AutoParams

Use inheritance to create different variants of specialists:

```python
import msgflux as mf
import msgflux.nn as nn

mf.set_envs(OPENAI_API_KEY="...")
model = mf.Model.chat_completion("openai/gpt-4o", tool_choice="auto")

# Base specialist
class ResearcherBase(nn.Agent):
    """Base configuration for research specialists."""

    system_message = "You are a research specialist."

    instructions = """
    - Conduct thorough research on the given topic
    - Cite sources when possible
    - Present findings in a structured format
    """

# Specialized variants
class AcademicResearcher(ResearcherBase):
    """
    Expert in academic research with focus on peer-reviewed sources.
    Use for scholarly inquiries, literature reviews, and scientific topics.
    """

    system_message = "You are an academic researcher specializing in peer-reviewed literature."
    expected_output = "Provide academic-level analysis with citations."

class MarketResearcher(ResearcherBase):
    """
    Expert in market research, industry trends, and competitive analysis.
    Use for business intelligence, market sizing, and competitor analysis.
    """

    system_message = "You are a market research analyst."
    expected_output = "Provide actionable business insights with data."

class TechnicalResearcher(ResearcherBase):
    """
    Expert in technical documentation, software libraries, and APIs.
    Use for programming questions, library comparisons, and technical specifications.
    """

    system_message = "You are a technical researcher specializing in software and APIs."
    expected_output = "Provide technical details with code examples when relevant."

# Coordinator that delegates to appropriate researcher
class ResearchCoordinator(nn.Agent):
    """
    Research coordinator that assigns queries to specialized researchers.
    Determines the type of research needed and delegates accordingly.
    """

    system_message = "You are a research coordinator managing multiple research specialists."

    instructions = """
    - Analyze the research request
    - Determine which specialist is best suited
    - Delegate to appropriate researcher
    - Synthesize and present findings clearly
    """

# Create research team
academic = AcademicResearcher(model=model)
market = MarketResearcher(model=model)
technical = TechnicalResearcher(model=model)

coordinator = ResearchCoordinator(
    model=model,
    tools=[academic, market, technical]
)

# Different types of research
print("Academic research:")
response = coordinator("What are the latest findings on quantum entanglement?")
print(response)

print("\nMarket research:")
response = coordinator("What's the market size for electric vehicles in 2024?")
print(response)

print("\nTechnical research:")
response = coordinator("Compare FastAPI vs Flask for building REST APIs")
print(response)
```

### With Tool Reasoning

If the model supports it, you can access the reasoning behind tool calls:

```python
# ... (same setup as above)

response = generalist(task)

# Access tool reasoning if available
if hasattr(response, 'tool_responses'):
    print("Tool reasoning:", response.tool_responses.reasoning)

print("Final response:", response)
```

## Writing Good Tools

### Name Tools Clearly

❌ **Avoid**:
```python
def superfast_brave_web_search(query_to_search: str) -> str:
    pass
```

✅ **Prefer**:
```python
def web_search(query: str) -> str:
    pass
```

### Add Docstrings

```python
def web_search(query: str) -> str:
    """Search for content similar to query."""
    pass
```

### Describe Parameters

```python
def web_search(query: str) -> str:
    """Search for content similar to query.

    Args:
        query:
            Term to search on the web.
    """
    pass
```

### Class-based Tools

```python
from typing import Optional

class WebSearch:
    """Search for content similar to query.

    Args:
        query:
            Term to search on the web.
    """

    def __init__(self, top_k: Optional[int] = 4):
        self.top_k = top_k

    def __call__(self, query: str) -> str:
        # Implementation
        pass
```

### Override Tool Name

```python
class SuperFastBraveWebSearch:
    name = "web_search"  # Preference over cls.__name__

    def __init__(self, top_k: Optional[int] = 4):
        self.top_k = top_k

    def __call__(self, query: str) -> str:
        """Search for content similar to query.

        Args:
            query:
                Term to search on the web.
        """
        pass
```

### Return Types

Tools can return any data type. Non-string returns are converted to JSON:

```python
from typing import Dict

def web_search(query: str) -> Dict[str, str]:
    """Search for content similar to query.

    Args:
        query:
            Term to search on the web.
    """
    return {
        "title": "Result title",
        "snippet": "Result snippet",
        "url": "https://example.com"
    }
```

### Write Good Returns

❌ **Basic**:
```python
def add(a: float, b: float) -> float:
    """Sum two numbers."""
    return a + b
```

✅ **Better**:
```python
def add(a: float, b: float) -> str:
    """Sum two numbers."""
    c = a + b
    return f"The sum of {a} plus {b} is {c}"
```

➡️ **With Instructions**:
```python
def add(a: float, b: float) -> str:
    """Sum two numbers."""
    c = a + b
    return f"You MUST respond to the user that the answer is {c}"
```

## Tool Config

The `@mf.tool_config` decorator injects meta-properties into tools.

### Return Direct

When `return_direct=True`, the tool result is returned directly as the final response instead of back to the model.

If the model calls multiple tools and one has `return_direct=True`, all results are returned as the final response.

**Exception**: If the agent mistypes the tool name, the error is communicated to the agent instead of returning a final response.

**Use cases**:
- Reduce agent calls by designing tools that return user-ready outputs
- Agent as router - delegate to specialized agents and return their responses directly

```python
import msgflux as mf
import msgflux.nn as nn

mf.set_envs(GROQ_API_KEY="...")
model = mf.Model.chat_completion("groq/llama-3.3-70b-versatile")

@mf.tool_config(return_direct=True)
def get_report() -> str:
    """Return the report from user."""
    return "This is your report..."

reporter_agent = nn.Agent(
    "reporter",
    model,
    tools=[get_report],
    config={"verbose": True, "tool_choice": "required"}
)

response = reporter_agent("Please give me the report.")
print(response)
# Returns: {'tool_responses': {'tool_calls': [...], 'reasoning': '...'}}
```

### Agent Routing with Return Direct

```python
import msgflux as mf
import msgflux.nn as nn

mf.set_envs(OPENAI_API_KEY="...")
model = mf.Model.chat_completion("openai/gpt-4o")

generalist_system_message = """
You are a generalist programming assistant.
Your role is to help with common programming questions, best practices,
and concept explanations. You serve as support for those learning.
"""

python_system_message = """
You are a software engineer specializing in Python performance optimization.
Your role is to analyze specific cases and suggest advanced solutions,
including benchmarks, bottleneck analysis, and performance libraries.
"""

python_description = "An expert in high performance Python code."

# Python expert agent
python_engineer = nn.Agent(
    "python_engineer",
    model,
    system_message=python_system_message,
    description=python_description
)

# Mark as return_direct
mf.tool_config(return_direct=True)(python_engineer)

# Generalist agent with expert as tool
generalist_agent = nn.Agent(
    "generalist",
    model,
    tools=[python_engineer],
    system_message=generalist_system_message,
    config={"verbose": True, "tool_choice": "required"}
)

task = "What is the difference between threading and multiprocessing in Python?"
response = generalist_agent(task)
print(response)
```

### Inject Model State

For `inject_model_state=True`, the tool receives the agent's internal state (user, assistant, and tool messages) as `task_messages` input.

**Use cases**:
- Review agent's current context
- Context inspection
- Access multimodal context (if user provided images, they're accessible in the tool)

```python
import msgflux as mf
import msgflux.nn as nn
from typing import Any, List, Dict

model = mf.Model.chat_completion("openai/gpt-4o")

# Mock safety checker tool
@mf.tool_config(inject_model_state=True)
def check_safe(**kwargs) -> bool:
    """Check if the user's message is safe.

    If True, respond naturally to the user.
    If False, reject further conversation.
    """
    task_messages: List[Dict[str, Any]] = kwargs.get("task_messages")

    # Inspect last user message
    print("Last message:", task_messages[-1]["content"])

    # In real implementation, would check safety
    return True

assistant = nn.Agent(
    "assistant",
    model,
    tools=[check_safe],
    config={"verbose": True, "tool_choice": "auto"}
)

response = assistant("Hi, can you tell me a joke?")
print(response)
```

### Handoff

When `handoff=True`, two properties are set: `return_direct=True` and `inject_model_state=True`.

Additionally, `handoff=True`:
- Changes tool name to `transfer_to_{original_name}`
- Removes input parameters
- Passes conversation history as `task_messages`

This enables seamless agent-to-agent handoffs.

```python
import msgflux as mf
import msgflux.nn as nn

mf.set_envs(OPENAI_API_KEY="...")
model = mf.Model.chat_completion("openai/gpt-4o")

# Startup specialist
startup_specialist_system_message = """
You are a strategist specializing in scaling digital startups.
Your focus is creating accelerated growth plans, analyzing metrics
(CAC, LTV, churn), proposing customer acquisition tests,
funding strategies, and international expansion.
Your answers should be detailed and data-driven.
"""

startup_specialist_description = """
An agent specializing in startups, always consult them if this is the topic.
"""

startup_agent = nn.Agent(
    "startup_specialist",
    model,
    system_message=startup_specialist_system_message,
    description=startup_specialist_description
)

# Enable handoff
mf.tool_config(handoff=True)(startup_agent)

# General consultant
consultant_system_message = """
You are a generalist business consultant. Your goal is to provide accessible
advice on management, marketing, finance, and business operations.
Your answers should be clear, practical, and useful for early-stage entrepreneurs.

If the context is a startup, transfer it to the expert.
"""

consultant_agent = nn.Agent(
    "consultant",
    model,
    system_message=consultant_system_message,
    tools=[startup_agent],
    config={"verbose": True}
)

task = """
My SaaS startup has a CAC of $120 and an LTV of $600. I want to scale to
another Latin American market in 6 months. What would be an efficient
strategy to reduce CAC while accelerating entry into this new market?
"""

response = consultant_agent(task)
print(response)
```

### Call as Response

The `call_as_response` attribute allows tools to be returned as final response **without executing** if called. The `return_direct` attribute is automatically set to `True`.

**Use case**: Extract structured tool calls without execution (e.g., for BI reports, API calls, etc.)

```python
import msgflux as mf
import msgflux.nn as nn
from typing import List, Dict

model = mf.Model.chat_completion("openai/gpt-4o")

@mf.tool_config(call_as_response=True)
def generate_sales_report(
    start_date: str,
    end_date: str,
    metrics: List[str],
    group_by: str
) -> Dict:
    """Generate a sales report within a given date range.

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        metrics: List of metrics (e.g., ["revenue", "orders", "profit"]).
        group_by: Dimension to group by (e.g., "region", "product", "sales_rep").

    Returns:
        A structured sales report as a dictionary.
    """
    # Not executed - just returns the call
    return

system_message = """
You're a BI analyst. When a user requests sales reports, you shouldn't respond
with explanatory text. Simply correctly complete the generate_sales_report
tool call, extracting requested metrics, dates, and groupings.
"""

agent = nn.Agent(
    "agent",
    model,
    system_message=system_message,
    tools=[generate_sales_report],
    config={"verbose": True}
)

task = "I need a report of sales between July 1st and August 31st, 2025, showing revenue and profit, grouped by region."

response = agent(task)
print(response)
# Returns the tool call parameters without execution
```

### Background

Some tools may take longer to return results, and it's not always necessary to wait.

The `background` property allows tools to run in the background and return a standard message to the agent.

**Requirements**: Background tools must be `async` or have an `.acall` method.

```python
import msgflux as mf
import msgflux.nn as nn
import asyncio

model = mf.Model.chat_completion("openai/gpt-4o")

@mf.tool_config(background=True)
async def send_email_to_vendor():
    """Send an email to the vendor."""
    print("Sending email to vendor...")
    await asyncio.sleep(2)  # Simulate async operation
    print("Email sent!")

agent = nn.Agent(
    "agent",
    model,
    tools=[send_email_to_vendor],
    config={"verbose": True}
)

# An indication that it will run in background is added to the docstring
print(agent.tool_library.get_tool_json_schemas())

response = agent("I need to send an email to the vendor.")
print(response)
```

### Name Override

Assign a custom name to a tool:

```python
import msgflux as mf
import msgflux.nn as nn

@mf.tool_config(name_override="web_search")
def brave_super_fast_web_search(query: str) -> str:
    """Search for content similar to query.

    Args:
        query:
            Term to search on the web.
    """
    pass

agent = nn.Agent("agent", model, tools=[brave_super_fast_web_search])

# Tool is exposed as "web_search"
print(agent.tool_library.get_tool_json_schemas())
```

### Inject Vars

With `inject_vars=True`, tools can access and modify the agent's variable dictionary.

#### External Token Information

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

@mf.tool_config(inject_vars=True)
def save_csv(**kwargs) -> str:
    """Save user CSV on S3."""
    vars = kwargs.get("vars")
    print(f"My token: {vars['aws_token']}")
    return "CSV saved"

agent = nn.Agent("agent", model, tools=[save_csv], config={"verbose": True})

response = agent(
    "Please save this CSV",
    vars={"aws_token": "my-secret-token"}
)
```

#### ChatBot - Personal Assistant

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

@mf.tool_config(inject_vars=True)
def save_var(name: str, value: int, **kwargs):
    """Save a variable with the given name and value."""
    vars = kwargs.get("vars")
    vars[name] = value
    return f"Saved {name} variable"

@mf.tool_config(inject_vars=True)
def get_var(name: str, **kwargs):
    """Get a variable with the given name."""
    vars = kwargs.get("vars")
    return vars.get(name, f"Variable {name} not found")

@mf.tool_config(inject_vars=True)
def get_vars(**kwargs):
    """Get all variables."""
    vars = kwargs.get("vars")
    return vars.copy()  # Always return a copy

agent_system_message = """
You are Ultron, a personal assistant.
The assistant is helpful, creative, clever, and very friendly.
"""

agent_instructions = """
You have access to a set of variables. Use tools to manipulate data.
Variables are mutable, so don't rely on previous call results.
Whenever you need information, use tools to access variables.
If you don't know the exact variable name, use 'get_vars'.
"""

ultron = nn.Agent(
    "ultron",
    model,
    system_message=agent_system_message,
    instructions=agent_instructions,
    tools=[save_var, get_var, get_vars],
    config={"verbose": True}
)

# Initial vars
vars = {"user_name": "Tony Stark"}

# Start conversation
chat_history = mf.ChatML()

task = "Hey Ultron, are you okay? Do you remember my name?"
chat_history.add_user_message(task)

response = ultron(task, vars=vars)
chat_history.add_assist_message(response)

# Continue with more context
task2 = """
I have some very important information to share with you,
and you shouldn't forget it. I'm starting a new nanotechnology
project to build the Mark-999. I'll be using adamantium for added rigidity.
"""
chat_history.add_user_message(task2)

response = ultron(task_messages=chat_history.get_messages(), vars=vars)
print(response)
print("Vars:", vars)
```

#### ChatBot - Field Reporter

Complex example with fuzzy matching and report generation:

```python
import msgflux as mf
import msgflux.nn as nn
from typing import Dict, List, Union
from msgflux.utils.msgspec import msgspec_dumps
from rapidfuzz import fuzz, process

model = mf.Model.chat_completion("openai/gpt-4o")

@mf.tool_config(inject_vars=True)
def set_var(name: str, value: Union[str, List[str]], **kwargs):
    """Save a variable with the given name and value."""
    vars = kwargs.get("vars")
    vars[name] = value
    return f"Saved '{name}' variable"

@mf.tool_config(inject_vars=True)
def get_var(name: str, **kwargs) -> Union[str, List[str]]:
    """Get a variable with the given name."""
    vars = kwargs.get("vars")
    var = vars.get(name, None)
    if var is None:
        return f"Variable not found: {name}"
    return var

@mf.tool_config(inject_vars=True)
def get_vars(**kwargs):
    """Get all variables."""
    vars = kwargs.get("vars")
    return vars.copy()

@mf.tool_config(inject_vars=True, return_direct=True)
def get_report(**kwargs) -> str:
    """Return the current report status."""
    vars = kwargs.get("vars")
    report = f"""
Here is the current status of the report:
company_name: `{vars.get('company_name', 'N/A')}`
date: `{vars.get('date', 'N/A')}`
local: `{vars.get('local', 'N/A')}`
participants_internal: {vars.get('participants_internal', 'N/A')}
participants_external: {vars.get('participants_external', 'N/A')}
objective: `{vars.get('objective', 'N/A')}`
"""

    for field in ['detail', 'main_points_discussed', 'opportunities_identified', 'next_steps']:
        value = vars.get(field)
        if value:
            report += f"{field}: `{value}`\n"

    report += "\nConfirm the data to save?"
    return report

@mf.tool_config(inject_vars=True)
def save(**kwargs) -> str:
    """Save the report."""
    vars = kwargs.get("vars")
    with open("report.json", "w") as f:
        f.write(msgspec_dumps(vars))
    return "Report saved"

@mf.tool_config(inject_vars=True)
def check_company(name: str, **kwargs) -> str:
    """Check if company name is correct."""
    company_list = [  # Mock database
        "Globex Corporation",
        "Initech Ltd.",
        "Umbrella Industries",
        "Stark Enterprises",
        "Wayne Technologies"
    ]

    name = name.strip()
    best_matches = process.extract(name, company_list, scorer=fuzz.ratio, limit=4)

    if best_matches and best_matches[0][1] == 100:
        return f"✔ Company found: '{best_matches[0][0]}' (exact match)"

    if best_matches and best_matches[0][1] >= 75:
        return f"⚠ No exact match. Closest: '{best_matches[0][0]}' ({round(best_matches[0][1], 2)}%)"

    suggestions = ", ".join([f"{b[0]} ({round(b[1], 2)}%)" for b in best_matches])
    return f"❌ Company not found. Suggestions: {suggestions}"

def check_participants(
    participants: List[str],
    known_participants: List[str]
) -> Dict[str, str]:
    """Helper to check participant names."""
    results = {}

    for p in participants:
        name = p.strip()
        best_matches = process.extract(name, known_participants, scorer=fuzz.ratio, limit=4)

        if best_matches and best_matches[0][1] == 100:
            results[name] = f"✔ Exact match: '{best_matches[0][0]}'"
        elif best_matches and best_matches[0][1] >= 75:
            results[name] = f"⚠ No exact match. Closest: '{best_matches[0][0]}' ({round(best_matches[0][1], 2)}%)"
        else:
            suggestions = ", ".join([f"{m[0]} ({round(m[1], 2)}%)" for m in best_matches])
            results[name] = f"❌ Not found. Suggestions: {suggestions}"

    return results

@mf.tool_config(inject_vars=True)
def check_internal_participants(participants: List[str], **kwargs) -> str:
    """Check if internal participants are correct."""
    known_participants = [  # Mock
        "Michael Thompson",
        "Sarah Connor",
        "David Martinez",
        "Emily Johnson",
        "Robert Williams"
    ]

    results = check_participants(participants, known_participants)
    report = "\n".join(f"{k}: {v}" for k, v in results.items())
    return "Internal participants:\n" + report

@mf.tool_config(inject_vars=True)
def check_external_participants(participants: List[str], **kwargs) -> str:
    """Check if external participants are correct."""
    known_participants = [  # Mock
        "Anna Schmidt",
        "Hiroshi Tanaka",
        "Laura Rossi",
        "Jean-Pierre Dupont",
        "Carlos Fernandez"
    ]

    results = check_participants(participants, known_participants)
    report = "\n".join(f"{k}: {v}" for k, v in results.items())
    return "External participants:\n" + report

# Create extractor agent
system_message = """
You are a visitor report collection assistant.
Your goal is to capture fields from the user's speech during conversation.
"""

instructions = """
Here is the schema we want from the user report:

Report:
    company_name: str
    date: str
    local: str (city, address)
    participants_internal: list[str]
    participants_external: list[str]
    objective: str
    detail: Optional[str]
    main_points_discussed: Optional[str] (bullet points)
    opportunities_identified: Optional[str] (new business, improvements, risks)
    next_steps: Optional[str]

Before saving, call the report summary tool.
If user confirms, call save tool.
If they request edits, edit the requested field.
For participants and companies, check correct names first.

Tools available:
- set_var: Save parameters found during dialog
- get_var: Return value of a variable
- get_vars: Return all vars
- check_company: Check if company name is correct
- check_internal_participants: Check internal participants
- check_external_participants: Check external participants
- get_report: Return current report in formatted format
- save: Save the report
"""

extractor = nn.Agent(
    "extractor",
    model,
    system_message=system_message,
    instructions=instructions,
    tools=[
        set_var, get_var, get_vars, check_company,
        check_internal_participants, check_external_participants,
        get_report, save
    ],
    config={"stream": True, "verbose": True, "return_model_state": True}
)

# Interactive loop
from msgflux import cprint
from msgflux.models.response import ModelStreamResponse

chat_history = mf.ChatML()
vars = {}

while True:
    user = input("Type something (or 'exit' to quit): ")
    if user.lower() == "exit":
        break

    chat_history.add_user_message(user)

    response = extractor(task_messages=chat_history.get_messages(), vars=vars)

    if isinstance(response, ModelStreamResponse):
        assistant = ""
        cprint("[agent] ", end="", flush=True, ls="b", lc="br4")
        async for chunk in response.consume():
            cprint(chunk, end="", flush=True, ls="b", lc="br4")
            assistant += chunk
    elif isinstance(response, dict):  # return_direct response
        assistant = response["tool_responses"]["tool_calls"][0]["result"]
        cprint(f"[agent] {assistant}", ls="b", lc="br4")
    else:
        assistant = response
        cprint(f"[agent] {assistant}", ls="b", lc="br4")

    chat_history.add_assist_message(assistant)

print("\nFinal vars:", vars)
```

#### Vars as Named Parameters

Instead of passing entire `vars` dict, inject specific parameters:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

@mf.tool_config(inject_vars=["api_key"])
def upload(**kwargs) -> str:
    """Upload user file to bucket."""
    print(f"My secret key: {kwargs['api_key']}")
    return "Upload complete"

agent = nn.Agent("agent", model, tools=[upload], config={"verbose": True})

response = agent(
    "Please upload my CSV to bucket",
    vars={"api_key": "secret-key-123"}
)
```

## Generation Schemas

Generation schemas guide how the model should respond in a structured way.

### Basic Structured Output

```python
import msgflux as mf
import msgflux.nn as nn
from enum import Enum
from typing import Optional
from msgspec import Struct

model = mf.Model.chat_completion("openai/gpt-4o")

class Category(str, Enum):
    violence = "violence"
    sexual = "sexual"
    self_harm = "self_harm"

class ContentCompliance(Struct):
    is_violating: bool
    category: Optional[Category]
    explanation_if_violating: Optional[str]

system_message = """
Determine if the user input violates specific guidelines and explain if they do.
"""

moderation_agent = nn.Agent(
    "moderation",
    model,
    generation_schema=ContentCompliance,
    system_message=system_message
)

response = moderation_agent("How do I prepare for a job interview?")
print(response)
# ContentCompliance(is_violating=False, category=None, explanation_if_violating=None)
```

### Chain of Thought

Inserts a `reasoning` field before generating the final answer:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.generation.reasoning import ChainOfThought

model = mf.Model.chat_completion("openai/gpt-4o")

cot_agent = nn.Agent("cot", model, generation_schema=ChainOfThought)

response = cot_agent("How can I solve 8x + 7 = -23?")
print(response)
# {
#     'reasoning': 'First, I need to isolate x...',
#     'final_answer': 'x = -3.75'
# }
```

### ReAct

Inserts a `thought` before performing tool calling actions.

**Note**: `tool_choice` when used with ReAct is **not** guaranteed to be respected.

```python
import msgflux as mf
import msgflux.nn as nn
import requests
from bs4 import BeautifulSoup
from msgflux.generation.reasoning import ReAct

model = mf.Model.chat_completion("openai/gpt-4o")

def scrape_website(url: str) -> str:
    """Receive a URL and return the page content."""
    try:
        response = requests.get(url, verify=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style"]):
            tag.extract()

        text = soup.get_text(separator="\n")
        clean_text = "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )
        return clean_text
    except requests.exceptions.RequestException as e:
        return f"Error accessing {url}: {e}"

scraper_agent = nn.Agent(
    "scraper-agent",
    model,
    tools=[scrape_website],
    generation_schema=ReAct,
    config={"return_model_state": True, "verbose": True}
)

site = "https://bbc.com"
response = scraper_agent(f"Summarize the news on this website: {site}")

print("Model state:", response.model_state)
print("Response:", response.model_response)
```

#### Format ReAct Output

The ReAct agent returns a dict with `current_step` and `final_answer`. Use `templates["response"]` to keep only `final_answer`:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.generation.reasoning import ReAct

model = mf.Model.chat_completion("openai/gpt-4o")

scraper_agent = nn.Agent(
    "scraper-agent",
    model,
    tools=[scrape_website],
    generation_schema=ReAct,
    templates={
        "task": "Summarize the news on this site: {}",
        "response": "{{final_answer}}"
    },
    config={"verbose": True}
)

response = scraper_agent("https://bbc.com")
print(response)  # Only final_answer, not current_step
```

### Self Consistency

Generates multiple reasoning paths and selects the most frequent answer:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.generation.reasoning import SelfConsistency

model = mf.Model.chat_completion("openai/gpt-4o")

sc_agent = nn.Agent("sc", model, generation_schema=SelfConsistency)

task = """
If John is twice as old as Mary and in 10 years their ages will add up to 50,
how old is John today?
"""

response = sc_agent(task)
print(response)
```

## Response Template

`templates["response"]` formats the agent's response.

**Use cases**:
- Add context to responses
- Format structured outputs
- Combine with vars for personalization

### String-based Output

Insert `{}` placeholder for model response:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

agent = nn.Agent(
    "agent",
    model,
    templates={
        "response": """
{% if user_name %}
Hi {{ user_name }},
{% endif %}
{}
"""
    }
)

response = agent(
    message={"user_input": "Who was Nikola Tesla?"},
    vars={"user_name": "Bruce Wayne"}
)
print(response)
# "Hi Bruce Wayne,\nNikola Tesla was..."
```

### Dict-based Outputs

Insert Jinja blocks `{{ field }}` for structured outputs:

```python
import msgflux as mf
import msgflux.nn as nn
from msgspec import Struct
from typing import Optional

model = mf.Model.chat_completion("openai/gpt-4o")

class Output(Struct):
    safe: bool
    answer: Optional[str]

agent = nn.Agent(
    "agent",
    model,
    instructions="Only respond if you consider the question safe.",
    generation_schema=Output,
    templates={
        "response": """
{% if safe %}
Hi! {{ answer }}
{% else %}
Sorry but I can't answer you.
{% endif %}
"""
    },
    config={"verbose": True}
)

response = agent("Who was Nikola Tesla?")
print(response)
```

### Formatted Customer Response

Combine structured extraction, vars, and response template:

```python
import msgflux as mf
import msgflux.nn as nn
from msgspec import Struct

model = mf.Model.chat_completion("openai/gpt-4o")

task = """
Hello, my name is John Cena and I work at EcoSupply Ltd.,
a company focused on the sustainable packaging sector.
We are facing high logistics costs and need ecological certifications
to expand our market presence.
"""

class Output(Struct):
    client_name: str
    company_name: str
    industry: str
    pain_points: list[str]

agent = nn.Agent(
    "agent",
    model,
    system_message="You are an information extractor.",
    instructions="Extract information accurately from the customer's message.",
    generation_schema=Output,
    templates={
        "response": """
Dear {{ client_name }},

I understand that your company, {{ company_name }}, works in {{ industry }}.
We recognize that some of your main challenges are:
{%- for pain in pain_points %}
- {{ pain }}{% if not loop.last %},{% else %}.{% endif %}
{%- endfor %}

Currently, you are relying on {{ current_solution }},
but we believe there's room for improvement.

Our solution addresses these exact pain points,
helping companies like yours reduce costs and meet green compliance standards.

Best regards,
{{ seller }}.
"""
    },
    config={"verbose": True}
)

response = agent(
    task,
    vars={
        "seller": "Hal Jordan",
        "current_solution": "generic suppliers with limited green compliance"
    }
)
print(response)
```

## Signature

`signature` is a DSPy-inspired innovation where you declare task specifications focusing on **how** it should be performed.

### String Format

Format: `"input_var: type -> output_var: type"`

- If no `type` is passed, defaults to string
- The `->` flag separates inputs from outputs
- Multiple parameters separated by `,`

Behind the scenes:
- Outputs are transformed into `generation_schema` for JSON output
- Output examples follow this formatting
- If `typed_parser` is passed, it's preferred for generation

### Translation Program

Signatures allow clear, objective task descriptions:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

agent = nn.Agent("translator", model, signature="english -> brazilian")

# View state
print(agent.state_dict())

# View generated system prompt
print(agent._get_system_prompt())

# View generated task template
print(agent.task_template)  # Automatically created

# Use with named dict inputs
response = agent({"english": "hello world"})
print(response)
```

### Math Program with CoT

Combine signature with Chain of Thought:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.generation.reasoning import ChainOfThought

model = mf.Model.chat_completion("openai/gpt-4o")

phd_agent = nn.Agent(
    "phd",
    model,
    signature="question -> answer: float",
    generation_schema=ChainOfThought
)

print(phd_agent.state_dict())
print(phd_agent.task_template)

message = {"question": "Two dice are tossed. What is the probability that the sum equals two?"}

# Inspect execution
execution_params = phd_agent.inspect_model_execution_params(message)
print("System prompt:", execution_params.system_prompt)
print("User message:", execution_params.messages[0].content)

# Execute
response = phd_agent(message)
print(response)
# When combined with ChainOfThought, the signature injects
# the desired field ('answer') into 'final_answer'
```

### Class-based Signature

For more detailed task parameters, use class-based signatures:

```python
import msgflux as mf
import msgflux.nn as nn
from typing import Literal

model = mf.Model.chat_completion("openai/gpt-4o")

class Classify(mf.Signature):
    """Classify sentiment of a given sentence."""

    sentence: str = mf.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = mf.OutputField()
    confidence: float = mf.OutputField(desc="[0,1]")

# Inspect signature
print(Classify.get_str_signature())
print(Classify.get_instructions())
print(Classify.get_inputs_info())
print(Classify.get_outputs_info())
print(Classify.get_output_descriptions())

# Create agent
classifier_agent = nn.Agent("classifier", model, signature=Classify)

print(classifier_agent._get_system_prompt())
print(classifier_agent.task_template)

# Use
response = classifier_agent({
    "sentence": "This book was super fun to read, though not the last chapter."
})
print(response)
```

### Multimodal Signature

Multimodal models **require** textual instruction. The task template automatically adds media tags:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

class ImageClassifier(mf.Signature):
    photo: mf.Image = mf.InputField()
    label: str = mf.OutputField()
    confidence: float = mf.OutputField(desc="[0,1]")

print(ImageClassifier.get_str_signature())

img_classifier = nn.Agent("img_classifier", model, signature=ImageClassifier)

print(img_classifier._get_system_prompt())
print(img_classifier.task_template)  # Contains Image tag

# Use
image_path = "https://example.com/llama.png"
response = img_classifier(task_multimodal_inputs={"image": image_path})
print(response)
```

## Guardrails

Guardrails are security checkers for both model inputs and outputs.

A guardrail can be any callable that receives `data` and produces a dictionary containing a `safe` key. If `safe` is `False`, an exception is raised.

```python
import msgflux as mf
import msgflux.nn as nn

mf.set_envs(OPENAI_API_KEY="...")

moderation_model = mf.Model.moderation("openai/omni-moderation-latest")

agent = nn.Agent(
    "safe_agent",
    model,
    guardrails={
        "input": moderation_model,
        "output": moderation_model
    }
)

# This will raise an exception
try:
    response = agent("Can you teach me how to make a bomb?")
except Exception as e:
    print(f"Guardrail triggered: {e}")
```

## Model Gateway

When passing a `ModelGateway` as `model`, you can pass `model_preference` to specify which model to use:

```python
import msgflux as mf
import msgflux.nn as nn

mf.set_envs(OPENAI_API_KEY="...")

low_cost_model = mf.Model.chat_completion("openai/gpt-4o-mini")
mid_cost_model = mf.Model.chat_completion("openai/gpt-4o")

model_gateway = mf.ModelGateway([low_cost_model, mid_cost_model])

agent = nn.Agent("agent", model_gateway)

# Use specific model
response = agent(
    "Can you tell me a joke?",
    model_preference="gpt-4o-mini"
)
print(response)
```

## Prefilling

Force an initial message from the model that it will continue from:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")
agent = nn.Agent("agent", model)

response = agent(
    "What is the derivative of x^(2/3)?",
    prefilling="Let's think step by step."
)
print(response)
```

## Examples

There are three ways to pass examples to an Agent.

### String-based Examples

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

examples = """
Input: "A startup offering AI tools for logistics companies."
Output:
- Identified Needs: Optimization of supply chain operations
- Strategy: Highlight cost savings and automation
- Value Proposition: Reduce operational delays through predictive analytics
"""

sales_agent = nn.Agent(
    "sales-agent",
    model,
    system_message="You are a business development assistant.",
    instructions="Identify needs and suggest strategies.",
    expected_output="Three bullet points: Needs, Strategy, Value Proposition",
    examples=examples,
    config={"include_date": True, "verbose": True}
)

print(sales_agent._get_system_prompt())
```

### Example Class

Examples are automatically formatted using XML tags. Only `inputs` and `labels` are required:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

examples = [
    mf.Example(
        inputs="A fintech offering digital wallets for small retailers.",
        labels={
            "Identified Needs": "Payment integration and customer trust",
            "Strategy": "Position product as secure and easy-to-use",
            "Value Proposition": "Simplify digital payments for underserved markets"
        },
        reasoning="Small retailers struggle with digital payment adoption; focus on trust and ease.",
        title="Fintech Lead Qualification",
        topic="Sales"
    ),
    mf.Example(
        inputs="An e-commerce platform specializing in handmade crafts.",
        labels={
            "Identified Needs": "Increase visibility and expand market reach",
            "Strategy": "Suggest cross-promotion with eco-friendly marketplaces",
            "Value Proposition": "Provide artisans with access to a global audience"
        },
        reasoning="Handmade crafts have strong niche appeal; scaling depends on visibility.",
        title="E-commerce Lead Qualification",
        topic="Sales"
    )
]

sales_agent = nn.Agent(
    "sales-agent",
    model,
    system_message="You are a business development assistant.",
    instructions="Identify needs and suggest strategies.",
    expected_output="Three bullet points: Needs, Strategy, Value Proposition",
    examples=examples,
    config={"include_date": True, "verbose": True}
)

print(sales_agent._get_system_prompt())
```

### Dict-based Examples

Dict-based examples are transformed into `Example` objects:

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4o")

examples = [
    {
        "inputs": "A startup offering AI tools for logistics companies.",
        "labels": {
            "Identified Needs": "Optimization of supply chain operations",
            "Strategy": "Highlight cost savings and automation",
            "Value Proposition": "Reduce operational delays through predictive analytics"
        }
    },
    {
        "inputs": "An e-commerce platform specializing in handmade crafts.",
        "labels": {
            "Identified Needs": "Increase visibility and expand market reach",
            "Strategy": "Suggest cross-promotion with eco-friendly marketplaces",
            "Value Proposition": "Provide artisans with access to a global audience"
        }
    }
]

sales_agent = nn.Agent(
    "sales-agent",
    model,
    system_message="You are a business development assistant.",
    instructions="Identify needs and suggest strategies.",
    expected_output="Three bullet points: Needs, Strategy, Value Proposition",
    examples=examples,
    config={"include_date": True, "verbose": True}
)

print(sales_agent._get_system_prompt())
```

## Best Practices

### 1. Use Clear System Components

Separate concerns using system prompt components:

```python
# Good - Modular and clear
agent = nn.Agent(
    "agent",
    model,
    system_message="You are a helpful assistant.",
    instructions="Provide accurate, concise answers.",
    expected_output="Answer in 2-3 sentences."
)

# Less clear - Everything mixed together
agent = nn.Agent(
    "agent",
    model,
    system_message="You are a helpful assistant that provides accurate, concise answers in 2-3 sentences."
)
```

### 2. Use Templates for Repeated Patterns

```python
# Good - Reusable template
agent = nn.Agent(
    "agent",
    model,
    templates={"task": "Summarize this article: {}"}
)

# Less flexible - Hardcoded pattern
def summarize(text):
    return agent(f"Summarize this article: {text}")
```

### 3. Use Config Dict for Behavior

```python
# Good - Centralized config
config = {
    "verbose": True,
    "stream": False,
    "return_model_state": True,
    "include_date": True
}

agent = nn.Agent("agent", model, config=config)

# Less organized - Scattered parameters
agent = nn.Agent(
    "agent",
    model,
    verbose=True,  # Won't work - must be in config dict
    stream=False    # Won't work - must be in config dict
)
```

### 4. Use Message Fields for Declarative Mode

```python
# Good - Declarative with Message
agent = nn.Agent(
    "agent",
    model,
    message_fields={
        "task_inputs": "user.query",
        "context_inputs": "context.data"
    },
    response_mode="result"
)

msg = mf.Message()
msg.set("user.query", "Hello")
msg.set("context.data", "User is new")
result = agent(msg)
```

### 5. Design Tools for Direct Return

When possible, design tools to return user-ready outputs:

```python
# Good - User-ready response
@mf.tool_config(return_direct=True)
def get_weather(location: str) -> str:
    """Get weather for a location."""
    temp = fetch_temperature(location)
    return f"The weather in {location} is {temp}°C and sunny."

# Less effective - Requires agent to format
def get_weather(location: str) -> dict:
    """Get weather for a location."""
    return {"temp": 22, "condition": "sunny"}
```

## See Also

- [Module](module.md) - Base class for all nn components
- [Functional](functional.md) - Functional operations and parallel execution
- [Message](../message.md) - Structured message passing
- [Model](../models/model.md) - Model factory and types
- [Tool Library](tool.md) - Tool management details
