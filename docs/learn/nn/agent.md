# nn.Agent


The `Agent` is a `Module` that uses language models to solve tasks. It can handle multimodal data, interact with environments through tool calls, and manage complex workflows with structured outputs.

## Overview

An `Agent` combines a language model with instructions and tools to accomplish tasks. The Agent module adopts a task decomposition strategy, allowing each part of a task to be treated in isolation.

### Key Features


- **Multimodal Support**: Handle text, images, audio, video, and files
- **Tool Calling**: Execute functions to interact with external systems
- **Generation Schemas**: Guides the model to generate typed responses, with support for reasoning strategies: Chain of Thought, ReAct, Self-Consistency
- **Flexible Configuration**: Customize behavior through message fields and config options
- **Template System**: Use Jinja templates for prompts and responses
- **Modular System Prompt**: Compose system prompts from independent components
- **Task Decomposition**: Break down complex tasks into manageable parts

---

## Quick Start

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

---

## AutoParams

The **preferred** way to define agents is using class attributes:

???+ example

    ```python
    # pip install msgflux[openai]
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    class SalesAgent(nn.Agent):
        """Persuasive sales agent focused on customer needs."""
        
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")        
        system_message = "You are an expert sales professional."
        instructions = "Identify customer needs and present value propositions."
        expected_output = "Respond with three bullet points."
        config = {"verbose": True}

    agent = SalesAgent()
    print(agent.name)         # "SalesAgent"
    print(agent.description)  # "Persuasive sales agent focused on customer needs."
    ```

AutoParams captures the class name as the agent *name*. The class docstring then becomes the *description* of that agent, which is useful for using that **agent-as-a-tool**.

---

## Async

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

## Streaming

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

---

## How to Debug an Agent

Understanding what's happening inside your agent is essential for building reliable AI applications. When an agent produces unexpected results, you need visibility into the prompts being sent, the model's reasoning, and how tools are being called.

msgFlux provides several inspection mechanisms to help you debug and understand agent behavior:

- **Verbose Mode**: Real-time console output of model calls and tool executions
- **Inspect Model Execution**: View the exact parameters that will be passed to the LM
- **Return Model State**: Access the complete conversation history for analysis or continuation
- **State Dict**: Inspect the agent's internal buffers and parameters

???+ example

    === "Verbose Mode"

        Verbose mode will print the model call steps, tool calls and their return values ​​to the console.    

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            config = {"verbose": True}       

        agent = Assistant()
        agent("Can I help me?")  
        ```

    === "Inspect Model Execution"

        This inspection allows you to view the exact values ​​that will be passed to the LM call.

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        agent = Assistant()
        params = agent.inspect_model_execution_params("Hello")
        print(params)
        ```

    === "Return Model State"

        Another inspection possibility is to analyze the internal agent state (messages). In msgFlux this is called `messages`. Returning the `messages` allows you to continue an interaction in future calls.

        When the configuration `config={"return_messages": True}` is passed, the agent returns a dict containing the keys `response` and `messages`.

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            config = {"return_messages": True}

        agent = Assistant()
        response = agent("Hello")
        print(response.response)
        print(response.messages)
        ```

    === "State Dict"

        To inspect the agent's buffers and parameters, simply call its *.state_dict()* method.

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message  = "You are a helpful assistant."
            instructions    = "Treat the user well."
            expected_output = "Correct responses."

        agent = Assistant()
        print(agent.state_dict())
        ```

---

## System Prompt Components

The system prompt is composed of 6 components:

| Component | Description | Example |
|-----------|-------------|---------|
| **system_message** | Agent behavior and role | "You are an agent specialist in..." |
| **instructions** | What the agent should do | "You MUST respond to the user..." |
| **expected_output** | Format of the response | "Your answer must be concise..." |
| **examples** | Input/output examples | Examples of reasoning and outputs |
| **system_extra_message** | Additional system context | Extra instructions or constraints |
| **include_date** | Include current date | Adds "Weekday, Month DD, YYYY" |

All components are assembled using a **system prompt template**, that can be customized via `templates={"system": "..."}`. By default, the template concatenates all defined components in a structured format using XML tags.

???+ example

    ```python
    # pip install msgflux[openai]
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    class BusinessAgent(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")        
        system_message = """
        You are a business development assistant,        
        focused on helping sales teams.
        """
        instructions = """
        When given a company description, identify potential needs,
        suggest an outreach strategy, and provide a value proposition.
        """
        expected_output = """
        Respond in three bullet points:
        - Identified Needs
        - Outreach Strategy  
        - Value Proposition
        """
        system_extra_message = """
        Ensure recommendations align with ethical sales practices.
        """
        config = {"include_date": True, "verbose": True}

    agent = BusinessAgent()
    print(agent.get_system_prompt())
    ```

### Examples

**[In-Context Learning (ICL)](https://arxiv.org/abs/2005.14165)** is a technique where language models learn to perform tasks by observing examples provided directly in the prompt, without any parameter updates. This allows models to generalize from just a few demonstrations.

**[Few-Shot Learning](https://arxiv.org/abs/2005.14165)** refers to providing a small number of input-output examples that guide the model's behavior. These examples act as implicit instructions, showing the model the expected format, reasoning style, and output structure.

Benefits of using examples:

- **Format consistency**: The model mimics the structure of your examples
- **Implicit instructions**: Complex behaviors can be demonstrated rather than explained
- **Reasoning patterns**: Show chain-of-thought reasoning for the model to follow
- **Domain adaptation**: Tailor responses to your specific use case

There are three ways to pass examples to an Agent:

???+ note "Few-shot Example Formats"

    === "String Examples"

        Simple text format:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        examples = """
        Input: "A startup offering AI tools for logistics."
        Output:
        - Needs: Supply chain optimization
        - Strategy: Highlight cost savings
        - Value: Reduce delays with predictive analytics

        Input: "An e-commerce platform for handmade crafts."
        Output:
        - Needs: Market visibility
        - Strategy: Cross-promotion with eco marketplaces
        - Value: Global audience access for artisans
        """

        class SalesAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You are a business development assistant."
            instructions = "Identify needs and suggest strategies.",
            expected_output = "Three bullet points: Needs, Strategy, Value"
            examples = examples

        agent = SalesAgent()
        print(agent.get_system_prompt())
        ```

    === "Example Class"

        Structured examples with metadata:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        examples = [
            mf.Example(
                inputs="A fintech offering digital wallets.",
                labels={
                    "Needs": "Payment integration and trust",
                    "Strategy": "Position as secure and easy-to-use",
                    "Value": "Simplify digital payments"
                },
                reasoning="Small retailers need trust and ease.",
                title="Fintech Lead",
                topic="Sales"
            ),
            mf.Example(
                inputs="An e-commerce for handmade crafts.",
                labels={
                    "Needs": "Visibility and market reach",
                    "Strategy": "Partner with eco marketplaces",
                    "Value": "Global audience for artisans"
                },
                reasoning="Handmade crafts need visibility to scale."
            )
        ]

        class SalesAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            examples = examples

        agent = SalesAgent()
        print(agent.get_system_prompt())
        ```

    === "Dict Examples"

        Dict-based examples are converted to `Example` objects:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        examples = [
            {
                "inputs": "A startup offering AI tools for logistics.",
                "labels": {
                    "Needs": "Supply chain optimization",
                    "Strategy": "Highlight cost savings",
                    "Value": "Reduce delays with predictive analytics"
                }
            },
            {
                "inputs": "An e-commerce for handmade crafts.",
                "labels": {
                    "Needs": "Market visibility",
                    "Strategy": "Cross-promotion with eco marketplaces",
                    "Value": "Global audience access"
                }
            }
        ]

        class SalesAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            examples = examples

        agent = SalesAgent()
        print(agent.get_system_prompt())
        ```

---

## Generation Schemas

`generation_schema` guide the model to respond in a pre-established structured format. By defining a schema using `msgspec.Struct`, the agent automatically constrains the model's output to match the expected structure, ensuring type-safe and predictable responses.

!!! tip "Performance"
    `msgspec` is the fastest validation and serialization library, which is why it was chosen to deliver maximum performance. See the [benchmarks](https://jcristharif.com/msgspec/benchmarks.html).

???+ example

    ```python
    # pip install msgflux[openai]
    from msgspec import Struct
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    class ContentCheck(Struct):
        reason: str        
        is_safe: bool

    class Moderator(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        generation_schema = ContentCheck
        config = {"verbose": True}

    agent = Moderator()
    result = agent(
        "Analyze this message: 'You are amazing and I appreciate your help!'"
    )
    print(result.is_safe)  # True/False
    ```

### Reasoning Schemas

msgFlux provides built-in generation schemas that implement common reasoning strategies. These schemas guide the model through structured thinking patterns before producing a response.

All reasoning schemas produce a `final_answer: str` field containing the model's concluded response.

???+ example "Built-in Reasoning Schemas"

    === "Chain of Thought"

        **Chain of Thought (CoT)** prompts the model to think step-by-step before providing a final answer. This improves reasoning accuracy, especially for math, logic, and multi-step problems.

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn        
        from msgflux.generation.reasoning import ChainOfThought

        # mf.set_envs(OPENAI_API_KEY="...")

        class Solver(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            generation_schema = ChainOfThought
            config = {"verbose": True}

        agent = Solver()

        result = agent("Solve: 8x + 7 = -23")
        print(result)
        # ChainOfThought(reasoning="First, subtract 7 from both sides: 8x = -30. Then divide by 8: x = -3.75", final_answer="x = -3.75")
        ```

    === "ReAct"

        **ReAct** (Reasoning + Acting) interleaves reasoning traces with tool actions. The agent thinks about what to do, executes a tool, observes the result, and repeats until the task is complete.
        
        Unlike standard tool calling, ReAct injects reasoning instructions and formats tools as text descriptions rather than passing function definitions directly to the model's `tools` parameter.

        ```python
        # pip install msgflux[openai]
        import httpx
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.generation.reasoning import ReAct

        # mf.set_envs(OPENAI_API_KEY="...")

        def fetch_webpage(url: str) -> str:
            """Fetch and return the content of a webpage."""
            response = httpx.get(url, follow_redirects=True)
            return response.text[:2000]  # First 2000 chars

        class WebResearcher(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            generation_schema = ReAct
            tools = [fetch_webpage]
            config = {"verbose": True}

        agent = WebResearcher()
        result = agent("What is the latest Python version from python.org?")
        # ReAct(thought="I need to check the official Python website", action="fetch_webpage", action_input={"url": "https://python.org"}, observation="...", final_answer="Python 3.12.x")
        ```

    === "Self Consistency"

        **Self-Consistency** generates multiple reasoning paths and selects the most consistent answer through majority voting. This reduces errors by cross-checking different chains of thought.

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn        
        from msgflux.generation.reasoning import SelfConsistency

        # mf.set_envs(OPENAI_API_KEY="...")

        class Solver(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            generation_schema = SelfConsistency
            config = {"verbose": True}

        agent = Solver()
        result = agent("If a train travels 120km in 2 hours, what is its speed?")
        # SelfConsistency(paths=["120/2 = 60 km/h", "Distance/Time = 120/2 = 60", "Speed = 60 km/h"], final_answer="60 km/h")
        ```

#### Extending Reasoning Schemas

You can extend any reasoning schema by inheriting from it and redefining the `final_answer` field with a custom type.


!!! example

    === "Python Type"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.generation.reasoning import ChainOfThought

        # mf.set_envs(OPENAI_API_KEY="...")

        class NumericAnswer(ChainOfThought):
            final_answer: int

        class Calculator(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            generation_schema = NumericAnswer
            config = {"verbose": True}

        agent = Calculator()
        result = agent("What is 25 + 17?")
        print(result.final_answer)  # 42 (int)
        ```

    === "Struct Type"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn        
        from msgspec import Struct
        from msgflux.generation.reasoning import ChainOfThought

        # mf.set_envs(OPENAI_API_KEY="...")

        class Decision(Struct):
            approved: bool
            confidence: float
            justification: str

        class ReasonedDecision(ChainOfThought):
            final_answer: Decision

        class Reviewer(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            generation_schema = ReasonedDecision
            config = {"verbose": True}

        agent = Reviewer()
        result = agent("Should we approve this budget request for $5000?")
        print(result.final_answer.approved)      # True/False
        print(result.final_answer.confidence)    # 0.85
        print(result.final_answer.justification) # "The request is within budget limits..."
        ```

---

## Task and Context

The agent receives input through **task** (what to do) and **context** (background information). When both are provided, they're combined using XML-like tags in the final prompt.

### Imperative vs Declarative

There are two ways to pass inputs to an agent:

| Mode | How it works |
|------|--------------|
| **Imperative** (kwargs) | Pass parameters directly via function arguments |
| **Declarative** (message_fields) | Agent extracts inputs from a `mf.Message` object |

The **declarative approach** with `message_fields` shines when designing complex systems: instead of manually wiring inputs and outputs between agents, you configure each agent once and let them consume/produce from a shared `Message`. This lets you **focus on system design** rather than plumbing variables between components. See [Declarative Mode with Message](#declarative-mode-with-message) for full details.

### Input Parameters

| Parameter | Description | Init | Runtime |
|-----------|-------------|:----:|:-------:|
| `task_inputs` | Main task input (string or dict for templates) | | ✅ |
| `context_inputs` | Dynamic context passed at call time | | ✅ |
| `context_cache` | Fixed context stored in the agent | ✅ | |
| `task_multimodal_inputs` | Multimodal inputs (image, audio, file) | | ✅ |
| `task_messages` | Conversation history (ChatML format) | | ✅ |
| `vars` | Variables for Agent, Templates and Tools | | ✅ |

### How Task and Context are Combined

When you pass `context_inputs`, the context is injected **inside the task** using XML-like tags:

```xml
<context>
Company: FinData Analytics
Industry: FinTech
Product: AI-powered risk analysis
</context>

<task>
Create a pitch for this client
</task>
```

This structure helps the model clearly distinguish between background information (context) and what it needs to do (task).

???+ example

    === "Context Inputs (str)"

        Pass task as first argument and context via `context_inputs`:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class SalesAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            config = {"verbose": True}

        agent = SalesAgent()

        context = """
        Company: FinData Analytics
        Industry: FinTech
        Product: AI-powered risk analysis
        """

        params = agent.inspect_model_execution_params(
            "Create a pitch for this client", context_inputs=context
        )
        print(params)
        ```

    === "Context Cache"

        Use `context_cache` for context that doesn't change between calls:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class CompanyAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            context_cache = """
            Company values:
            - Customer first
            - Innovation
            - Integrity
            """
            config = {"verbose": True}

        agent = CompanyAgent()
        params = agent.inspect_model_execution_params(
            "Write a response to a customer complaint", context_inputs=context
        )
        print(params)
        ```

    === "Same Agent, Two Modes"

        The same agent logic can be used imperatively or declaratively:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        # Define agent with message_fields for declarative mode
        class Scraper(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [scrape_website]
            templates = {"task": "Summarize the news on this site: {}"}
            message_fields = {"task_inputs": "content"} # Where to read
            response_mode = "summary"                   # Where to write

        scraper = Scraper()

        # Imperative: Pass directly, get directly
        response = scraper("https://example.com/news")
        print(response)  # Direct string response

        # Declarative: Read from Message, write to Message
        msg = mf.Message(content="https://example.com/news")
        msg = scraper(msg)
        print(msg.summary)  # Response stored in message
        ```

        **When to choose declarative:**

        - Building pipelines where agents pass data to each other
        - Production systems where you want agents to be self-documenting
        - When you want to **design the system once** and let data flow through it

### Templates

Templates use **Jinja2** syntax to format inputs and outputs. There are three template types:

| Template | Purpose | Data Source |
|----------|---------|-------------|
| `task` | Format the task/question sent to the model | `task_inputs` dict + [vars](#vars) |
| `context` | Format background context | `context_inputs` dict + [vars](#vars) |
| `response` | Format the model's output before returning | Model output fields + [vars](#vars) |

!!! tip "Response Template + Generation Schema"
    The `response` template is especially powerful when combined with `generation_schema`. The model outputs structured data, and you use Jinja to transform it into a human-readable format. This separates **what the model extracts** from **how you present it**.

???+ note "Template Examples"

    === "Task Template"

        Use `templates={"task": ...}` to format the task input:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            templates = {
                "task": """
                {% if user_name %}
                My name is {{ user_name }}.
                {% endif %}
                {{ user_input }}
                """
            }
            config = {"verbose": True}

        agent = Assistant()

        response = agent(
            task_inputs={"user_input": "Who was Nikola Tesla?"},
            vars={"user_name": "Bruce Wayne"}
        )
        ```

    === "Context Template"

        Use `templates={"context": ...}` to format structured context:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class SalesAgent(nn.Module):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            templates = {
                "context": """
                The client is **{{ client_name }}** in the **{{ industry }}** sector.

                Challenges:
                {%- for pain in pain_points %}
                - {{ pain }}
                {%- endfor %}
                """
            }
            config = {"verbose": True}

        agent = SalesAgent()

        response = agent(
            "Create a pitch",
            context_inputs={
                "client_name": "EcoSupply Ltd.",
                "industry": "Sustainable packaging",
                "pain_points": ["High costs", "Certification needs"]
            }
        )
        ```

    === "Response Template (String)"

        For plain text outputs, use `{}` as placeholder for the model response:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            templates = {
                "response": """
                {% if user_name %}
                Hi {{ user_name }},
                {% endif %}
                {}
                """
            }
            config = {"verbose": True}

        agent = Assistant()

        response = agent(
            "Who was Nikola Tesla?",
            vars={"user_name": "Bruce Wayne"}
        )
        # Output: "Hi Bruce Wayne,\n\nNikola Tesla was a Serbian-American inventor..."
        ```

    === "Response Template (Structured)"

        When using `generation_schema`, access output fields directly in the template:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn
        from msgspec import Struct
        from typing import Optional

        # mf.set_envs(OPENAI_API_KEY="...")

        class SafetyCheck(Struct):
            safe: bool
            answer: Optional[str]

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            instructions = "Only respond if the question is safe."
            generation_schema = SafetyCheck
            templates={
                "response": """
                {% if safe %}
                Hi! {{ answer }}
                {% else %}
                Sorry, I can't answer that question.
                {% endif %}
                """
            }
            message_fields = {"task_inputs": "content"}
            response_mode = "assistant.output"
            config = {"verbose": True}

        agent = Assistant()

        msg = mf.Message(content="Who was Nikola Tesla?")
        msg = agent(msg)
        print(msg.get("assistant.output"))
        # Model outputs: {"safe": true, "answer": "Nikola Tesla was..."}
        # Template formats: "Hi! Nikola Tesla was..."
        ```

    === "Response + Extraction"

        Extract structured data and format a personalized response:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn
        from msgspec import Struct

        # mf.set_envs(OPENAI_API_KEY="...")

        class ClientInfo(Struct):
            client_name: str
            company_name: str
            industry: str
            pain_points: list[str]

        class Extractor(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You are an information extractor."
            instructions = "Extract information from the customer's message."
            generation_schema = ClientInfo
            templates = {
                "response": """
                Dear {{ client_name }},

                I understand that {{ company_name }} operates in {{ industry }}.

                Your main challenges:
                {%- for pain in pain_points %}
                - {{ pain }}
                {%- endfor %}

                Our solution addresses these exact pain points.

                Best regards,
                {{ seller }}.
                """
        }


        task = """
        Hello, my name is John and I work at EcoSupply Ltd.,
        a sustainable packaging company. We face high logistics
        costs and need ecological certifications.
        """

        agent = Assistant()

        response = agent(task, vars={"seller": "Hal Jordan"})
        # Model extracts: {"client_name": "John", "company_name": "EcoSupply Ltd.", ...}
        # Template produces personalized letter
        ```

    === "Response + Reasoning"

        Extract only `final_answer` from reasoning schemas like ReAct:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.generation.reasoning import ReAct

        # mf.set_envs(OPENAI_API_KEY="...")

        def web_search(query: str) -> str:
            """Search the web for information."""
            # Simplified example - in production, use a real search API
            return f"Search results for '{query}': ..."

        class Researcher(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [web_search]
            generation_schema = ReAct
            templates = {
                "task": "Research this topic: {}",
                "response": "{{ final_answer }}"  # Only return the final answer
            }
            config = {"verbose": True}
        
        agent = Researcher()

        # ReAct outputs: {"current_step": {...}, "final_answer": "..."}
        # Template extracts just the final_answer
        response = agent("What is the population of Tokyo?")
        ```

    === "Signature + Response"

        Combine signature with response template for clean tool outputs:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn
        from typing import Literal

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        class Classify(mf.Signature):
            """Classify sentiment of a given sentence."""
            sentence: str = mf.InputField()
            sentiment: Literal["positive", "negative", "neutral"] = mf.OutputField()
            confidence: float = mf.OutputField()

        @mf.tool_config(return_direct=True)
        class SentimentClassifier(nn.Agent):
            """Specialist in sentiment analysis."""
            model = model
            signature = Classify
            templates = {
                "response":
                """The sentence was classified as {{ sentiment }},
                with a confidence of {{ (confidence * 100) | round(2) }}%."""
            }
            config = {"verbose": True}

        class Assistant(nn.Agent):
            model = model
            tools = [SentimentClassifier]
            config = {"verbose": True}

        assistant = Assistant()
        response = assistant("Classify: 'This book was amazing!'")
        # Output: "The sentence was classified as positive, with a confidence of 92.5%."
        ```

!!! info "Task Template Without task_inputs"
    When you configure a `task` template but don't pass `task_inputs`, the rendered template itself becomes the task. This is useful for scenarios where the prompt is fixed and only some component changes (like images or [vars](#vars)).

### Multimodal Inputs

Pass images, audio, or files via `task_multimodal_inputs`. Requires a multimodal model (e.g., `gpt-4.1`, `gpt-4.1-mini`).

| Media | Single | Multiple |
|-------|:------:|:--------:|
| Image | ✅ | ✅ |
| Audio | ✅ | ❌ |
| Video | ✅ | ❌ |
| File  | ✅ | ❌ |

!!! warning "Model Compatibility"
    Not all models support all multimodal inputs. Before using video, audio, or file inputs, verify that your chosen model supports that media type. Check the model provider's documentation for supported input types.


???+ note "Multimodal Examples"

    === "Image"

        Single image or multiple images for comparison:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class VisionAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        agent = VisionAgent()

        # Single image (URL)
        response = agent(
            "Describe this image",
            task_multimodal_inputs={
                "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            }
        )

        # Multiple images for comparison
        response = agent(
            "Compare these two images",
            task_multimodal_inputs={
                "image": [
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
                ]
            }
        )

        # With task template (fixed prompt, variable image)
        # When task_inputs is not passed, the rendered template becomes the task
        class DescribeAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            templates = {"task": "Describe this image in {{ language }}."}

        describe_agent = DescribeAgent()

        # Inspect what the model would receive
        params = describe_agent.inspect_model_execution_params(
            task_multimodal_inputs={
                "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            },
            vars={"language": "Portuguese"}
        )
        print(params["messages"])
        # The task will be: "Describe this image in Portuguese."

        # Execute the agent
        response = describe_agent(
            task_multimodal_inputs={
                "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            },
            vars={"language": "Portuguese"}
        )
        ```

    === "Image (Message Fields)"

        Use `message_fields` for declarative mapping from Message objects:

        !!! info "Graceful Handling of Missing Fields"
            If an image cannot be retrieved from the declarative mapping (e.g., the field doesn't exist or is empty), the task will still be assembled normally — the missing image simply won't be included. This allows flexible pipelines where multimodal inputs are optional.

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        # Single image mapping
        class VisionAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            message_fields = {
                "task_inputs": "user.query",
                "task_multimodal_inputs": {"image": "user.image_url"}
            }

        agent = VisionAgent()

        # Create message with structured data
        msg = mf.Message()
        msg.set("user.query", "What objects are in this image?")
        msg.set("user.image_url", "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg")

        response = agent(msg)

        # Multiple images mapping
        class ComparisonAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            message_fields = {
                "task_inputs": "user.query",
                "task_multimodal_inputs": {"image": "user.images"}
            }

        comparison_agent = ComparisonAgent()

        msg = mf.Message()
        msg.set("user.query", "Compare these two images and describe the differences.")
        msg.set("user.images", [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
        ])

        response = comparison_agent(msg)
        ```

    === "Audio"

        Transcribe or analyze audio files:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class AgentTranscriber(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-audio-preview")

        agent = AgentTranscriber()

        response = agent(
            "Transcribe this audio and identify the speaker's emotion",
            task_multimodal_inputs={"audio": "/path/to/recording.wav"}
        )
        ```

    === "File (PDF)"

        Analyze PDF documents:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class AnalyzerAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1")

        agent = AnalyzerAgent()

        # From URL
        response = agent(
            "Summarize the key findings of this paper",
            task_multimodal_inputs={"file": "https://arxiv.org/pdf/1706.03762.pdf"}
        )

        # From local file
        response = agent(
            "Extract the main conclusions",
            task_multimodal_inputs={"file": "./report.pdf"}
        )
        ```


### Task Messages (Chat History)

Pass a list of messages in ChatML format to provide conversation history. This is useful for chatbots and multi-turn conversations.

Use `config={"return_messages": True}` to get back both the agent's response and the internal message history, which you can feed back into the agent for the next turn.

???+ note "Chat History Examples"

    === "Basic Chat"

        Use `return_messages` to capture and reuse conversation history:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            config = {"return_messages": True}

        agent = Assistant()

        # First message - no history yet
        result = agent("Hi, my name is Peter Parker, and I'm a photographer.")
        print(result.response)

        # result.messages contains the ChatML messages (user + assistant)
        # Feed it back for the next turn
        result = agent(
            "Can you recommend some cameras for my freelance job?",
            messages=result.messages
        )
        print(result.response)
        # The agent remembers you're Peter Parker and a photographer
        ```

    === "Multi-turn Conversation"

        Chain multiple turns by passing `messages` each time:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Advisor(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You are a helpful camera advisor."
            config = {"return_messages": True}

        agent = Advisor()

        # Turn 1
        result = agent("I'm looking for a compact camera under $500.")
        print(f"Assistant: {result.response}")

        # Turn 2 - pass previous messages
        result = agent(
            "Which one has better low-light performance?",
            messages=result.messages
        )
        print(f"Assistant: {result.response}")

        # Turn 3 - continue the chain
        result = agent(
            "What about battery life?",
            messages=result.messages
        )
        print(f"Assistant: {result.response}")
        ```

    === "ChatBot Pattern"

        Complete chatbot loop with streaming:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class ChatBot(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You are a helpful assistant."
            config = {"stream": True, "return_messages": True}

        agent = ChatBot()
        messages = None  # No history initially

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break

            # Pass previous messages (None on first turn)
            result = agent(user_input, messages=messages)

            # Handle streaming response
            full_response = ""
            print("Assistant: ", end="", flush=True)
            for chunk in result.consume():
                print(chunk, end="", flush=True)
                full_response += chunk
            print()

            # Update messages for next turn
            messages = result.messages
        ```

---

## Vars

`vars` is the **unified execution variable space**. Is a Mapping (dict-like). The data is transferred between Agent, Tools, and templates. It can be read, updated, and overwritten.

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

        response = agent(
            "Help me with a purchase",
            vars={"customer_name": "Clark Kent"}
        )     
        ```

    === "Tools"

        Tools can access the `vars` using the `@mf.tool_config(inject_vars=True)` decorator. See [inject_vars](#inject_vars) for more details.

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
        response = agent(
            "What discount do I have?",
            vars={"customer_name": "Clark Kent"}
        )
        # Tool returns: "Clark Kent has a 15% loyalty discount."
        ```

---

## Tools

Tools are interfaces that allow models to perform actions or query information.

### What are Tools?
1.  **Function Calling** - A tool is exposed as a function with defined name, parameters, and types
    - Example: `web_search(query: str)`
    - The model decides whether to call it and provides arguments

2.  **Extending Capabilities** - Tools allow you to:
    - Search for real-time data (news, stocks, databases)
    - Perform precise calculations
    - Manipulate systems (send emails, schedule events)
    - Integrate with external APIs

3.  **Agent-based Orchestration** - The LLM acts as an agent that decides:
    - When to use a tool
    - Which tool to use
    - How to interpret the tool's output

In msgFlux, a **Tool can be any callable** (function, class with `__call__`/`acall` e.g. nn.Agent).

!!! info

    While more tools enable more actions, too many tools can confuse the model about which one to use.

!!! tip

    A good practice is to inform the model in the system prompt when it should use that tool.    

???+ example

    === "Web Scraper"

        Extract text content from web pages using `httpx` and `BeautifulSoup`:

        ```python
        # pip install msgflux[openai] beautifulsoup4
        import httpx
        import msgflux as mf
        import msgflux.nn as nn
        from bs4 import BeautifulSoup

        # mf.set_envs(OPENAI_API_KEY="...")

        def scrape_webpage(url: str) -> str:
            """Fetch and extract text content from a URL.

            Args:
                url: The webpage URL to scrape.

            Returns:
                Cleaned text content from the page.
            """
            try:
                response = httpx.get(url, timeout=10, follow_redirects=True)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove non-content elements
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.extract()

                text = soup.get_text(separator="\n")
                clean_text = "\n".join(
                    line.strip() for line in text.splitlines() if line.strip()
                )

                # Truncate if too long for context
                if len(clean_text) > 8000:
                    clean_text = clean_text[:8000] + "\n...[truncated]"

                return clean_text
            except httpx.HTTPError as e:
                return f"Error fetching {url}: {e}"

        class WebReader(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You help users understand web content."
            tools = [scrape_webpage]            
            config = {"verbose": True}

        agent = WebReader()

        response = agent("Summarize the main points from https://news.ycombinator.com")
        ```

    === "GitHub API"

        Query GitHub's public API for repository information:

        ```python
        # pip install msgflux[openai]
        import httpx
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        def get_github_repo(owner: str, repo: str) -> str:
            """Get information about a GitHub repository.

            Args:
                owner: Repository owner (username or organization).
                repo: Repository name.

            Returns:
                Repository details including stars, forks, and description.
            """
            url = f"https://api.github.com/repos/{owner}/{repo}"
            response = httpx.get(url, timeout=10)

            if response.status_code == 404:
                return f"Repository {owner}/{repo} not found."

            if response.status_code != 200:
                return f"Error fetching repository: {response.status_code}"

            data = response.json()
            return f"""
            Repository: {data['full_name']}
            Description: {data.get('description', 'No description')}
            Stars: {data['stargazers_count']:,}
            Forks: {data['forks_count']:,}
            Language: {data.get('language', 'Unknown')}
            Open Issues: {data['open_issues_count']}
            URL: {data['html_url']}
            """

        class GithubAssistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You help users explore GitHub repositories."
            tools = [get_github_repo]            
            config = {"verbose": True}

        response = agent("Tell me about the pytorch repository")
        ```

    === "File Operations"

        Real file system operations:

        ```python
        # pip install msgflux[openai]
        import os        
        from pathlib import Path        
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        def list_files(directory: str, pattern: str = "*") -> str:
            """List files in a directory matching a pattern.

            Args:
                directory: Path to the directory.
                pattern: Glob pattern to filter files (default: all files).

            Returns:
                List of matching files with sizes.
            """
            path = Path(directory).expanduser()

            if not path.exists():
                return f"Directory not found: {directory}"

            if not path.is_dir():
                return f"Not a directory: {directory}"

            files = list(path.glob(pattern))[:20]  # Limit results

            if not files:
                return f"No files matching '{pattern}' in {directory}"

            result = []
            for f in files:
                size = f.stat().st_size if f.is_file() else 0
                size_str = f"{size:,} bytes" if f.is_file() else "directory"
                result.append(f"  {f.name} ({size_str})")

            return f"Files in {directory}:\n" + "\n".join(result)

        def read_file(filepath: str, max_lines: int = 50) -> str:
            """Read content from a text file.

            Args:
                filepath: Path to the file.
                max_lines: Maximum lines to read (default: 50).

            Returns:
                File content or error message.
            """
            path = Path(filepath).expanduser()

            if not path.exists():
                return f"File not found: {filepath}"

            if not path.is_file():
                return f"Not a file: {filepath}"

            try:
                lines = path.read_text().splitlines()[:max_lines]
                content = "\n".join(lines)
                if len(lines) == max_lines:
                    content += "\n...[truncated]"
                return content
            except Exception as e:
                return f"Error reading file: {e}"

        class FileAssistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You help users explore files on their system."
            tools = [list_files, read_file]            
            config = {"verbose": True}

        response = agent("List Python files in the current directory")
        ```

    === "Wikipedia Search"

        Use msgflux's built-in Wikipedia retriever as a tool:

        ```python
        # pip install msgflux[openai] wikipedia
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        # Create Wikipedia search tool from built-in retriever
        wikipedia = mf.Retriever.web_search("wikipedia")

        class Researcher(nn.Module):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You are a research assistant with access to Wikipedia.",
            tools = [wikipedia]            
            config = {"verbose": True}

        response = agent("Tell me about the history of the Python programming language")
        ```

    === "Multiple Tools"

        Combine multiple tools for a research assistant:

        ```python
        # pip install msgflux[openai] wikipedia beautifulsoup4
        import httpx
        import msgflux as mf
        import msgflux.nn as nn
        from bs4 import BeautifulSoup

        # mf.set_envs(OPENAI_API_KEY="...")

        # Built-in Wikipedia search
        wikipedia = mf.Retriever.web_search("wikipedia")

        # Custom web scraper for any URL
        def scrape_url(url: str) -> str:
            """Fetch and extract text from any webpage.

            Args:
                url: The URL to scrape.
            """
            try:
                resp = httpx.get(url, timeout=10, follow_redirects=True)
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav"]):
                    tag.extract()
                text = soup.get_text(separator="\n", strip=True)
                return text[:5000] + "..." if len(text) > 5000 else text
            except httpx.HTTPError as e:
                return f"Error: {e}"

        class Researcher(nn.Module):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [wikipedia, scrape_url]
            system_message = "You research topics using Wikipedia and web scraping."
            config = {"verbose": True}

        response = agent(
            "Compare what Wikipedia says about Rust vs the official rust-lang.org site"
        )
        ```

### Writing Good Tools

#### Tool Names

A well-defined tool is fundamental for the model to understand **when** and **how** to use it. The model reads the tool's name, description (docstring), and parameter definitions to decide if it should call the tool and with what arguments.

Poor tool definitions lead to:

- The model not calling the tool when it should
- Incorrect parameter values being passed
- Confusion when multiple tools have similar names

???+ example "Tool Names and Description"

    === "Good Tool Name"

        A simple, descriptive name helps the model quickly understand the tool's purpose. Combined with a clear docstring and well-documented parameters, the model can make accurate decisions about when to use the tool.

        **Best practices:**

        - Use short, action-oriented names (`search`, `send_email`)
        - Document the purpose in the docstring
        - Describe each parameter with type hints and descriptions

        ```python
        def web_search(query: str) -> str:
            """Search for content similar to query.

            Args:
                query: Term to search on the web.

            Returns:
                Results similar to query.
            """
            pass
        ```

    === "Bad Tool Name"

        Long, complex names with unnecessary prefixes confuse the model. Missing or poor descriptions make it impossible for the model to understand when to use the tool.

        **Common problems:**

        - Overly long names with implementation details (`superfast_brave_web_search`)
        - Redundant parameter names (`query_to_search` instead of `query`)
        - Missing docstrings or parameter descriptions
        - No type hints

        ```python
        def superfast_brave_web_search(query_to_search: str) -> str:
            pass  # No docstring, no parameter description
        ```


#### Tool Returns

The way a tool returns information affects how well the model interprets and uses the result.

???+ example "Return Value Best Practices"

    === "Basic Return"

        Returns the value, but model must infer context:

        ```python
        def add(a: float, b: float) -> float:
            """Sum two numbers."""
            return a + b  # Returns: 8
        ```

    === "Descriptive Return"

        Provides context that helps the model respond naturally:

        ```python
        def add(a: float, b: float) -> str:
            """Sum two numbers."""
            c = a + b
            return f"The sum of {a} plus {b} is {c}"
            # Returns: "The sum of 5 plus 3 is 8"
        ```

    === "Instructive Return"

        Guides the model on how to use the result:

        ```python
        def add(a: float, b: float) -> str:
            """Sum two numbers."""
            c = a + b
            return f"The calculation is complete. Tell the user: {a} + {b} = {c}"
        ```


### Tool Choice

Control how the model selects tools.

**Options:**

| Value | Behavior |
|-------|----------|
| `"auto"` | Model decides whether to use tools (default) |
| `"required"` | Model must call at least one tool |
| `"none"` | Model cannot use tools |
| `"tool_name"` | Model must call the specific tool |

???+ example

    === "auto (default)"

        Model decides when to use tools:

        ```python
        # pip install msgflux[openai] wikipedia
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        wikipedia = mf.Retriever.web_search("wikipedia")

        class Researcher(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [wikipedia]
            config = {"tool_choice": "auto", "verbose": True}

        agent = Researcher()

        # Model may or may not use the tool
        response = agent("What is the capital of France?")  # Probably won't use tool
        response = agent("Tell me about quantum entanglement")  # Will likely use tool
        ```

    === "required"

        Force the model to always use a tool:

        ```python
        # pip install msgflux[openai] wikipedia
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        wikipedia = mf.Retriever.web_search("wikipedia")

        class Researcher(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [wikipedia]
            config = {"tool_choice": "required", "verbose": True}

        agent = Researcher()

        # Model MUST call a tool before responding
        response = agent("What is photosynthesis?")
        ```

    === "Specific Tool"

        Force a specific tool to be called:

        ```python
        # pip install msgflux[openai] wikipedia      
        import httpx
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        wikipedia = mf.Retriever.web_search("wikipedia")

        def search_github(query: str) -> str:
            """Search GitHub repositories."""
            resp = httpx.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "per_page": 5}
            )
            repos = resp.json().get("items", [])
            return "\n".join(f"- {r['full_name']}: {r['description']}" for r in repos)

        class SearchAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [search_github, wikipedia]
            # Always use GitHub
            config = {"tool_choice": "search_github", "verbose": True}

        agent = SearchAgent()
        response = agent("Find machine learning projects")
        ```

    === "none"

        Disable tool usage temporarily:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")
        
        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [wikipedia, search_github]
            # Tools disabled
            config = {"tool_choice": "none", "verbose": True}

        # Model will respond without using any tools
        response = agent("What do you know about Python?")
        ```

    === "Router Pattern"

        Use `required` for routing agents:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        @mf.tool_config(return_direct=True)
        class PythonExpert(nn.Agent):
            """Expert in Python programming."""

            model = model
            system_message = "You are a Python expert."

        @mf.tool_config(return_direct=True)
        class RustExpert(nn.Agent):
            """Expert in Rust programming."""

            model = model
            system_message = "You are a Rust expert."

        class Router(nn.Agent):
            """Routes programming questions to the right expert."""
            model = model
            system_message = "Route questions to the appropriate expert."
            tools = [PythonExpert, RustExpert]
            config = {"tool_choice": "required", "verbose": True}

        router = Router()

        # Router MUST pick an expert
        response = router("How do I handle errors in Rust?")
        ```

### Async Tools

When your agent runs asynchronously with `acall()`, prefer writing async tools as well. This ensures non-blocking execution and better performance when tools perform I/O operations.

???+ note "Sync vs Async Tools"

    === "Async Tool (Recommended)"

        ```python
        import httpx

        async def fetch_data(url: str) -> str:
            """Fetch data from a URL asynchronously."""
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                return response.text
        ```

    === "Sync Tool"

        ```python
        import httpx

        def fetch_data(url: str) -> str:
            """Fetch data from a URL."""
            response = httpx.get(url, follow_redirects=True)
            return response.text
        ```

You can also implement a class-based async tool using the `acall` method:

???+ example

    ```python
    import httpx

    class WebFetcher:
        """Fetch content from web pages."""

        def __init__(self, timeout: int = 30):
            self.timeout = timeout

        async def acall(self, url: str) -> str:
            """Fetch content from URL asynchronously."""
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                return response.text
    ```

### Class-based Tools

Tools can be implemented as classes with `__call__` or `acall` methods. This is useful when you need to maintain state or configure the tool at initialization.

???+ example "Class-based Tool"

    === "Basic Class Tool"

        ```python
        from typing import Optional        
        import httpx

        class GitHubSearch:
            """Search GitHub repositories."""

            def __init__(self, max_results: Optional[int] = 5):
                self.max_results = max_results

            def __call__(self, query: str) -> str:
                """Search for repositories matching query.

                Args:
                    query: Search term for repositories.
                """
                url = "https://api.github.com/search/repositories"
                params = {"q": query, "per_page": self.max_results}
                response = httpx.get(url, params=params, timeout=10)

                if response.status_code != 200:
                    return f"Error: {response.status_code}"

                data = response.json()
                results = []
                for repo in data.get("items", []):
                    results.append(f"- {repo['full_name']} ({repo['stargazers_count']}⭐)")

                return "\n".join(results) if results else "No repositories found."
        ```

    === "Override Tool Name"

        Use the `name` attribute to override the class name:

        ```python
        import httpx

        class GitHubRepoSearchV2:
            name = "search_repos"  # Exposed as "search_repos" instead of class name

            def __init__(self, max_results: int = 5):
                self.max_results = max_results

            def __call__(self, query: str) -> str:
                """Search GitHub for repositories."""
                url = "https://api.github.com/search/repositories"
                resp = httpx.get(url, params={"q": query, "per_page": self.max_results})
                repos = resp.json().get("items", [])
                return "\n".join(f"- {r['full_name']}" for r in repos) or "No results."
        ```

### Return Types

Tools can return any data type. Non-string returns are automatically serialized using `msgspec.json.encode` before being passed to the model.

???+ note "Tool Return Examples"

    === "String Return"

        ```python
        def add(a: float, b: float) -> str:
            """Sum two numbers."""
            return f"The sum of {a} plus {b} is {a + b}"
        ```

    === "Dict Return"

        ```python
        from typing import Dict

        def web_search(query: str) -> Dict[str, str]:
            """Search for content."""
            return {
                "title": "Result title",
                "snippet": "Result snippet",
                "url": "https://example.com"
            }
        ```

    === "List Return"

        ```python
        from typing import List

        def get_top_results(query: str) -> List[Dict]:
            """Get top search results."""
            return [
                {"title": "Result 1", "url": "..."},
                {"title": "Result 2", "url": "..."}
            ]
        ```


---

### Tool Config

The `@mf.tool_config` decorator adds special behaviors to tools.

#### return_direct

When `return_direct=True`, the tool result is returned directly as the final response instead of going back to the model.

Use cases:

- Reduce agent calls by designing tools that return user-ready outputs
- Agent as router - delegate to specialists and return their responses directly

???+ example

    === "Basic Usage"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(return_direct=True)
        def get_report() -> str:
            """Return the report."""
            return "This is your detailed report..."

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [get_report]

        agent = Assistant()
        response = agent("Give me the report")
        # Returns the tool result directly, no model formatting
        ```

    === "With Reasoning Models"

        Combine `return_direct` with reasoning models to optimize tool calls. The model reasons about which tool to use, but the result bypasses additional processing:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(GROQ_API_KEY="...")

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-20b", reasoning_effort="low"
        )

        @mf.tool_config(return_direct=True)
        def get_report() -> str:
            """Return the report from user."""
            return "This is your detailed report..."

        class ReporterAgent(nn.Agent):
            model = model
            tools = [get_report]
            config = {"tool_choice": "required", "verbose": True}

        agent = ReporterAgent()
        response = agent("Give me the report")
        ```

    === "Report Generator"

        Combine with `inject_vars` for external processing:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(GROQ_API_KEY="...")

        @mf.tool_config(return_direct=True, inject_vars=True)
        def generate_formatted_report(**kwargs) -> str:
            """Generate a formatted sales report."""
            vars = kwargs.get("vars", {})
            date_range = vars.get("date_range", "Unknown")

            # Mock data - in production, query your database
            report = f"""
            Sales Report: {date_range}
            ─────────────────────────────
            Total Revenue: $124,500
            Total Orders: 847
            Average Order: $147.04
            Top Product: Widget Pro (234 units)
            ─────────────────────────────
            Generated automatically.
            """
            return report

        class Reporter(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [generate_formatted_report]
            config = {"verbose": True}
        
        agent = Reporter()
        response = agent("Generate the Q3 report", vars={"date_range": "2024-Q3"})
        ```

#### inject_vars

With `inject_vars=True`, tools can access and modify the agent's variable dictionary.

Use cases:

- Pass external credentials (API keys, tokens)
- Share state between tools
- Extract information from tools without returning it to the model (e.g., store metadata, logs, or intermediate results in `vars` for later use)

???+ example

    === "External Credentials"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(inject_vars=True)
        def save_to_s3(**kwargs) -> str:
            """Save file to S3."""
            vars = kwargs.get("vars")
            token = vars["aws_token"]
            # Use token for S3 upload
            return "File saved successfully"

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [save_to_s3]
        
        agent = Assistant()
        response = agent("Save my file", vars={"aws_token": "secret-123"})
        ```

    === "Named Parameters"

        Inject specific vars as named parameters:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(inject_vars=["api_key", "user_id"])
        def upload_file(**kwargs) -> str:
            """Upload user file."""
            api_key = kwargs["api_key"]
            user_id = kwargs["user_id"]
            return f"Uploaded for user {user_id}"

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [upload_file]
        
        agent = Assistant()
        response = agent("Upload my file", vars={"api_key": "...", "user_id": "123"})
        ```

    === "Mutable State"

        Tools can modify vars for persistent state:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(inject_vars=True)
        def save_preference(name: str, value: str, **kwargs):
            """Save a user preference."""
            vars = kwargs.get("vars")
            vars[name] = value  # Modifies the vars dict
            return f"Saved {name} = {value}"

        @mf.tool_config(inject_vars=True)
        def get_preference(name: str, **kwargs):
            """Get a user preference."""
            vars = kwargs.get("vars")
            return vars.get(name, "Not found")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [save_preference, get_preference]

        agent = Assistant()

        user_vars = {}
        agent("Save my favorite color as blue", vars=user_vars)
        agent("What is my favorite color?", vars=user_vars)

        print(user_vars)  # {"favorite_color": "blue"}
        ```

#### inject_messages

With `inject_messages=True`, the tool receives the agent's internal state (conversation history) as `task_messages` in kwargs. This is particularly useful for **agent-as-a-tool** patterns where you want to pass the full conversation context to a specialist agent.

Use cases:

- Agent-as-a-tool: Pass conversation history to specialist agents
- Safety/moderation checks on conversation
- Access multimodal context (e.g. images in conversation)
- Context-aware tool execution

???+ example

    === "Agent-as-a-Tool (Primary Use)"

        When an agent is used as a tool, `inject_messages` passes the conversation history so the specialist has full context:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        # With inject_messages, the specialist receives
        # the coordinator's conversation as task_messages
        @mf.tool_config(inject_messages=True)
        class Specialist(nn.Agent):
            """Expert that needs conversation context."""

            model = model
            system_message = "You are a specialist."

        class Coordinator(nn.Agent):
            model = model
            system_message = "Route to specialists when needed."
            tools = [Specialist]
            config = {"verbose": True}

        coordinator = Coordinator()

        # When coordinator calls specialist, the full conversation
        # is passed via task_messages parameter
        response = coordinator("Help me with a complex problem")
        ```

    === "Safety Checker"

        Check conversation safety before responding:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(inject_messages=True)
        def check_safety(**kwargs) -> dict:
            """Check if the conversation is safe to continue."""
            messages = kwargs.get("task_messages", [])
            last_message = messages[-1]["content"] if messages else ""

            # Simple keyword-based safety check
            forbidden_keywords = ["hack", "exploit", "malware", "attack"]
            content_lower = last_message.lower()
            is_safe = not any(kw in content_lower for kw in forbidden_keywords)

            return {
                "safe": is_safe,
                "reason": None if is_safe else "Potentially harmful content detected"
            }
        
        class SafeAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            instructions = "Always check safety before responding."
            tools = [check_safety]
            config = {"verbose": True}

        agent = SafeAgent()
        response = agent("Can you help me write a Python script?")
        ```

    === "Context-Aware Processing"

        Access images or other multimodal content from conversation:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")
                
        @mf.tool_config(inject_messages=True)
        def analyze_shared_images(**kwargs) -> str:
            """Analyze all images shared in the conversation."""
            messages = kwargs.get("task_messages", [])

            images = []
            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "image_url":
                            images.append(block["image_url"]["url"])

            if not images:
                return "No images found in conversation."

            return f"Found {len(images)} images to analyze."
        ```

#### handoff

When `handoff=True`, the tool is configured for seamless agent-to-agent handoff:

- Sets `return_direct=True` and `inject_messages=True`
- Changes tool name to `transfer_to_{original_name}`
- Removes input parameters (conversation history is passed instead)

???+ example

    ```python
    # pip install msgflux[openai]
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    # Tool is now "transfer_to_TechnicalSupport" with no parameters
    @mf.tool_config(handoff=True)
    class TechnicalSupport(nn.Agent):
        """Specialist for technical issues, debugging, and troubleshooting."""
        model = model
        system_message = "You are a technical support specialist."
        instructions = "Help users solve technical problems step by step."
        config = {"verbose": True}

    class Coordinator(nn.Agent):
        """Routes user queries to the appropriate specialist."""
        model = model
        system_message = "You are a support coordinator."
        instructions = "Transfer users to technical support for technical issues."
        tools = [TechnicalSupport]
        config = {"verbose": True}

    coordinator = Coordinator()
    response = coordinator("My application crashes when I try to connect to the database")
    ```

#### call_as_response

Return tool call parameters **without executing** the tool. Useful for extracting structured data.

Use cases:

- BI report parameter extraction
- API call preparation
- Form data collection

???+ example

    ```python
    # pip install msgflux[openai]
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    @mf.tool_config(call_as_response=True)
    def generate_sales_report(
        start_date: str, end_date: str, metrics: list[str], group_by: str
    ) -> dict:
        """Generate a sales report within a given date range.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            metrics: List of metrics to include (e.g., ["revenue", "orders", "profit"]).
            group_by: Dimension to group data by (e.g., "region", "product", "sales_rep").

        Returns:
            A structured sales report as a dictionary.
        """
        return  # Never executed

    class BIAnalyst(nn.Agent):
        model = model
        system_message = """You're a BI analyst. When a user requests sales reports,
        you should simply complete the generate_sales_report tool call,
        extracting the requested metrics, dates, and groupings."""
        tools = [generate_sales_report]
        config = {"verbose": True}

    agent = BIAnalyst()
    response = agent(
        "I need a report of sales between July 1st and August 31st, 2025, "
        "showing revenue and profit, grouped by region."
    )
    # Returns the tool call parameters without executing the function
    ```

#### background

Run tool in background without waiting for completion. Requires async tool.

Use cases:

- Fire-and-forget operations (emails, notifications)
- Long-running tasks that don't need immediate results

???+ example

    ```python
    # pip install msgflux[openai]    
    import asyncio
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    @mf.tool_config(background=True)
    async def send_notification(user_id: str, message: str):
        """Send notification asynchronously. Runs in background."""
        # Simulate async operation (e.g., API call, email sending)
        await asyncio.sleep(2)
        print(f"Notification sent to {user_id}: {message}")

    class Notifier(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        tools = [send_notification]
        config = {"verbose": True}

    agent = Notifier()

    # Agent returns immediately, notification sends in background
    response = agent("Notify user123 that their order shipped")
    ```

#### name_override

Assign a custom name to a tool:

```python
import httpx

@mf.tool_config(name_override="search_repos")
def github_repository_search_v2_extended(query: str) -> str:
    """Search GitHub repositories."""
    url = "https://api.github.com/search/repositories"
    resp = httpx.get(url, params={"q": query, "per_page": 3})
    repos = resp.json().get("items", [])
    return "\n".join(f"- {r['full_name']}" for r in repos)

# Tool is exposed as "search_repos" instead of the long function name
```

---

### Agent-as-a-Tool

Agents can be used as tools for other agents, enabling hierarchical task delegation. Using AutoParams makes this pattern especially clean: the class name becomes the tool name, and the docstring becomes the tool description.

???+ note "Agent-as-a-Tool Examples"

    === "Health Team"

        A coordinator agent delegates to specialist agents:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        class Nutritionist(nn.Agent):
            """Specialist in nutrition, diet planning, and healthy eating habits.
            Consult for meal plans, dietary recommendations, and nutritional advice."""

            model = model
            system_message = "You are a certified nutritionist."
            instructions = """Create clear and practical meal plans tailored to the user's goals.
            Be objective, technical, and structured."""

        class FitnessTrainer(nn.Agent):
            """Specialist in fitness, exercise routines, and physical training.
            Consult for workout plans, training schedules, and exercise guidance."""

            model = model
            system_message = "You are a certified personal trainer."
            instructions = """Design workout routines based on the user's fitness level and goals.
            Focus on safety, progression, and sustainability."""

        class HealthCoordinator(nn.Agent):
            """Coordinates health specialists to provide comprehensive wellness advice."""

            model = model
            system_message = "You coordinate a team of health specialists."
            instructions = "Delegate user requests to the appropriate specialist."
            tools = [Nutritionist, FitnessTrainer]
            config = {"verbose": True}

        coordinator = HealthCoordinator()

        response = coordinator("I want to lose 10kg and build muscle")
        ```

    === "Research Team"

        Multiple research specialists with a coordinator:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        class AcademicResearcher(nn.Agent):
            """Expert in academic research with peer-reviewed sources.
            Use for scholarly inquiries and scientific topics."""

            model = model
            system_message = "You are an academic researcher."
            expected_output = "Provide academic-level analysis with citations."

        class MarketResearcher(nn.Agent):
            """Expert in market research and competitive analysis.
            Use for business intelligence and market sizing."""

            model = model
            system_message = "You are a market research analyst."
            expected_output = "Provide actionable business insights."

        class TechnicalResearcher(nn.Agent):
            """Expert in technical documentation and APIs.
            Use for programming questions and library comparisons."""

            model = model
            system_message = "You are a technical researcher."
            expected_output = "Provide technical details with code examples."

        class ResearchCoordinator(nn.Agent):
            model = model
            system_message = "You coordinate research specialists."
            instructions = "Delegate to the appropriate researcher based on the query type."
            tools = [
                AcademicResearcher,
                MarketResearcher,
                TechnicalResearcher
            ]
            config = {"verbose": True}

        coordinator = ResearchCoordinator()

        response = coordinator("Compare FastAPI vs Flask for building REST APIs")
        ```

    === "Agent Router"

        Route requests directly to specialists using `return_direct`:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        @mf.tool_config(return_direct=True)
        class PythonExpert(nn.Agent):
            """Expert in Python performance optimization."""

            model = model
            system_message = "You specialize in Python performance."

        @mf.tool_config(return_direct=True)
        class JavaScriptExpert(nn.Agent):
            """Expert in JavaScript and Node.js."""

            model = model
            system_message = "You specialize in JavaScript."

        class Router(nn.Agent):
            model = model
            system_message = "Route programming questions to the right expert."
            tools = [PythonExpert, JavaScriptExpert]
            config = {"verbose": True}

        router = Router()

        # Response comes directly from the specialist
        response = router("How do I optimize a Python loop?")
        ```

    === "Handoff Pattern"

        Seamless conversation handoff between agents:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        
        # Enable handoff - transfers conversation history
        @mf.tool_config(handoff=True)
        class StartupSpecialist(nn.Agent):
            """Specialist in scaling digital startups.
            Use for growth strategies, metrics, and funding."""

            model = model
            system_message = "You are a startup scaling expert."

        class BusinessConsultant(nn.Agent):
            model = model
            system_message = """You are a business consultant.
            If the context is a startup, transfer to the specialist."""
            tools = [StartupSpecialist]
            config = {"verbose": True}            

        consultant = BusinessConsultant()

        # Conversation is handed off to specialist
        response = consultant(
            "My SaaS has a CAC of $120 and LTV of $600. How do I scale?"
        )
        ```

---

### MCP

The **Model Context Protocol (MCP)** allows agents to connect to external tool servers. MCP servers expose tools that can be called by the agent, enabling integration with filesystems, databases, APIs, and other services.

Configure MCP servers using the `mcp_servers` attribute:

???+ example

    === "Stdio Transport"

        Connect to an MCP server via standard I/O:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class FileAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            mcp_servers = [{
                "name": "filesystem",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            }]
            config = {"verbose": True}

        agent = FileAgent()
        response = agent("List all files in the current directory")
        ```

    === "HTTP Transport"

        Connect to an MCP server via HTTP:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class APIAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            mcp_servers = [{
                "name": "api",
                "transport": "http",
                "base_url": "http://localhost:8000",
                "headers": {"Authorization": "Bearer token"}
            }]

        agent = APIAgent()
        ```

    === "With Tool Configuration"

        Apply `tool_config` options to MCP tools:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class ConfiguredAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            mcp_servers = [{
                "name": "filesystem",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "include_tools": ["read_file", "write_file"],
                "tool_config": {
                    "read_file": {"inject_vars": True}
                }
            }]

        agent = ConfiguredAgent()
        ```

**Server Configuration Options:**

| Option | Description |
|--------|-------------|
| `name` | Namespace for tools from this server |
| `transport` | `"stdio"` or `"http"` |
| `command` | Command to start the server (stdio only) |
| `args` | Command arguments (stdio only) |
| `cwd` | Working directory (stdio only) |
| `env` | Environment variables (stdio only) |
| `base_url` | Server URL (http only) |
| `headers` | HTTP headers (http only) |
| `include_tools` | Allowlist of tools to expose |
| `exclude_tools` | Blocklist of tools to hide |
| `tool_config` | Per-tool configuration options |

---

## Signatures

A **Signature** is a declarative specification of input/output behavior for an Agent. Instead of hand-crafting prompts, you define the semantic roles of inputs and outputs, and msgFlux handles the prompt engineering for you.

This [DSPy-inspired](https://dspy.ai/learn/programming/signatures/) feature automatically generates:

- **System prompt** with task description (from docstring)
- **Task template** with input placeholders
- **Generation schema** for structured output
- **Annotations** for agent-as-a-tool integration

### Why Use Signatures?

| Without Signature | With Signature |
|-------------------|----------------|
| Manual prompt engineering | Declarative task specification |
| String manipulation for inputs | Type-safe input/output fields |
| Ad-hoc output parsing | Automatic structured responses |
| Manual tool schema definition | Auto-generated annotations |

Signatures let you focus on **what** the task should accomplish, not **how** to prompt the model.

### Inline Signatures

For simple tasks, use the shorthand string notation with arrow syntax:

```python
"input_field -> output_field"
```

The default type is `str` when unspecified.

???+ example "Inline Signature Examples"

    ```python
    # pip install msgflux[openai]
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    # Simple: single input -> single output
    class Translator(nn.Agent):
        model = model
        signature = "english -> portuguese"

    agent = Translator()
    response = agent(english="hello world")
    print(response.portuguese)  # "olá mundo"

    # With types: specify field types explicitly
    class Extractor(nn.Agent):
        model = model
        signature = "text -> sentiment: Literal['positive', 'negative', 'neutral'], confidence: float"

    agent = Extractor()
    result = agent(text="I love this product!")
    print(result.sentiment)    # "positive"
    print(result.confidence)   # 0.95

    # Multiple inputs
    class Calculator(nn.Agent):
        model = model
        signature = "expression: str, precision: int -> result: float"

    agent = Calculator()
    result = agent(expression="sqrt(2)", precision=4)
    print(result.result)  # 1.4142
    ```

### Class-Based Signatures

For complex tasks, class-based signatures provide full control with typed fields, descriptions, and docstrings. The class docstring becomes the instruction for the model.

```python
class TaskName(mf.Signature):
    """Task description that guides the model."""

    input_field: type = mf.InputField(desc="Field description")
    output_field: type = mf.OutputField(desc="Field description")
```

???+ example "Class Signature Examples"

    === "Basic Classification"

        ```python
        # pip install msgflux[openai]
        from typing import Literal
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Classify(mf.Signature):
            """Classify the sentiment of a given sentence."""

            sentence: str = mf.InputField()
            sentiment: Literal["positive", "negative", "neutral"] = mf.OutputField()
            confidence: float = mf.OutputField(desc="Confidence score between 0 and 1")

        class Classifier(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = Classify

        agent = Classifier()
        response = agent(sentence="This book was super fun to read!")
        print(response.sentiment)    # "positive"
        print(response.confidence)   # 0.92
        ```

    === "Complex Extraction"

        ```python
        # pip install msgflux[openai]
        from typing import List, Optional
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class ExtractEntities(mf.Signature):
            """Extract named entities from text with their types and context."""

            text: str = mf.InputField(desc="Text to analyze")
            entities: List[dict] = mf.OutputField(
                desc="List of {name, type, context} objects"
            )
            summary: str = mf.OutputField(desc="Brief summary of the text")

        class EntityExtractor(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = ExtractEntities

        agent = EntityExtractor()
        result = agent(text="Apple CEO Tim Cook announced new products in Cupertino.")
        # result.entities = [
        #     {"name": "Apple", "type": "ORG", "context": "technology company"},
        #     {"name": "Tim Cook", "type": "PERSON", "context": "CEO of Apple"},
        #     {"name": "Cupertino", "type": "LOC", "context": "city in California"}
        # ]
        ```

    === "With Detailed Instructions"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Translate(mf.Signature):
            """Translate text accurately while preserving meaning, tone, and cultural nuances.

            Guidelines:
            - Maintain the original tone (formal/informal)
            - Preserve idiomatic expressions when possible
            - Adapt cultural references appropriately
            """

            text: str = mf.InputField(desc="Text to translate")
            source_language: str = mf.InputField(desc="Source language code (e.g., 'en', 'pt')")
            target_language: str = mf.InputField(desc="Target language code")
            translation: str = mf.OutputField(desc="Translated text")
            notes: str = mf.OutputField(desc="Translation notes about cultural adaptations")

        class Translator(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = Translate

        agent = Translator()
        result = agent(
            text="It's raining cats and dogs!",
            source_language="en",
            target_language="pt"
        )
        print(result.translation)  # "Está chovendo canivetes!"
        print(result.notes)        # "Adapted English idiom to Brazilian Portuguese equivalent"
        ```

### Field Types

Signatures support various field types for different use cases:

| Type | Description | Example |
|------|-------------|---------|
| `str` | Text (default) | `name: str` |
| `int`, `float` | Numbers | `count: int`, `score: float` |
| `bool` | Boolean | `is_valid: bool` |
| `Literal[...]` | Constrained choices | `sentiment: Literal["pos", "neg"]` |
| `List[T]` | Lists | `tags: List[str]` |
| `dict` | Dictionaries | `metadata: dict` |
| `Image` | Image input | `photo: Image` |
| `Audio` | Audio input | `recording: Audio` |
| `Video` | Video input | `clip: Video` |
| `File` | File input | `document: File` |

### InputField and OutputField

Both `InputField` and `OutputField` accept a `desc` (or `description`) parameter to provide additional context:

```python
import msgflux as mf

class Review(mf.Signature):
    """Analyze a product review."""

    review_text: str = mf.InputField(desc="The customer review text")
    product_name: str = mf.InputField(desc="Name of the product being reviewed")

    rating: int = mf.OutputField(desc="Rating from 1 to 5 stars")
    pros: List[str] = mf.OutputField(desc="List of positive aspects mentioned")
    cons: List[str] = mf.OutputField(desc="List of negative aspects mentioned")
```

!!! tip "Field Descriptions"
    Use `desc` to clarify ambiguous fields, specify constraints (e.g., "between 0 and 1"), or provide examples. This helps the model understand exactly what you expect.

### Multimodal Signatures

Use `Image`, `Audio`, `Video`, or `File` for multimodal inputs:

???+ example "Multimodal Signature"

    === "Class-based"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class ImageClassifier(mf.Signature):
            """Classify the content of an image and describe what you see."""

            photo: mf.Image = mf.InputField(desc="Image to analyze")
            label: str = mf.OutputField(desc="Main subject of the image")
            description: str = mf.OutputField(desc="Detailed description")
            confidence: float = mf.OutputField(desc="Confidence score 0-1")

        class Classifier(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1")
            signature = ImageClassifier

        agent = Classifier()

        # Task template automatically includes image placeholder
        print(agent.task_template)

        response = agent(task_multimodal_inputs={
            "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        })
        print(response.label)        # "Nature boardwalk"
        print(response.description)  # "A wooden boardwalk path..."
        ```

    === "Str-based"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Classifier(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1")
            instructions = "Classify the content of an image and describe what you see."
            signature = "photo: Image -> label, description, confidence: float"

        agent = Classifier()

        # Task template automatically includes image placeholder
        print(agent.task_template)

        response = agent(task_multimodal_inputs={
            "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        })
        print(response.label)        # "Nature boardwalk"
        print(response.description)  # "A wooden boardwalk path..."
        ```


### Passing Inputs

When using signatures, you can pass inputs in multiple ways:

???+ note "Input Methods"

    === "As Kwargs"

        Pass inputs as keyword arguments (recommended):

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Translator(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1")
            signature = "english -> portuguese"

        agent = Translator()
        response = agent(english="hello world")
        ```

    === "As Dict"

        Pass all inputs as a dictionary:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Translator(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1")
        
        agent = Translator()
        response = agent({"english": "hello world"})
        ```

    === "With Context"

        Combine with `context_inputs`:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Summarize(mf.Signature):
            """Summarize text in a specific style."""
            text: str = mf.InputField()
            style: str = mf.InputField(desc="e.g., 'formal', 'casual', 'technical'")
            summary: str = mf.OutputField()

        class Summarizer(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1")
            signature = Summarize

        agent = Summarizer()
        response = agent(
            text="Long article...",
            style="casual",
            context_inputs="Focus on the key takeaways"
        )
        ```

### Combining Signatures with Other Components

Signatures can be combined with other Agent components. Here's how they interact:

#### What You Can Combine

| Component | Behavior with Signature |
|-----------|------------------------|
| `system_message` | **Additive** - Included in the system prompt alongside signature-generated content |
| `instructions` | **Override** - If provided, takes precedence over the signature's docstring |
| `examples` | **Additive** - Combined with any examples defined in the signature |
| `system_extra_message` | **Additive** - Appended to the system prompt |
| `generation_schema` | **Fused** - Merged with signature outputs (e.g., ChainOfThought + Signature) |

#### What the Signature Controls

| Component | Behavior |
|-----------|----------|
| `task` template | **Generated** - Created from input fields, overwrites any existing task template |
| `expected_output` | **Generated** - Created from output fields |
| `annotations` | **Generated** - Created from input fields for tool integration |

???+ example "Combining Signature with System Components"

    === "With system_message"

        Add context that applies to all requests:

        ```python
        # pip install msgflux[openai]        
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Translate(mf.Signature):
            """Translate text accurately."""
            text: str = mf.InputField()
            target_lang: str = mf.InputField()
            translation: str = mf.OutputField()

        class Translator(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = Translate
            # system_message is ADDED to the prompt, not replaced
            system_message = "You are a professional translator specialized in technical documents."

        agent = Translator()
        # System prompt includes *both* the system_message
        # and signature instructions
        print(agent.get_system_prompt())
        ```

    === "With instructions (Override)"

        Override the signature's docstring:

        ```python
        # pip install msgflux[openai]        
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Summarize(mf.Signature):
            """Summarize the given text."""  # This will be *ignored*
            text: str = mf.InputField()
            summary: str = mf.OutputField()

        class Summarizer(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = Summarize
            # instructions OVERRIDE the signature's docstring
            instructions = "Create a bullet-point summary with exactly 3 key points."

        agent = Summarizer()
        # The docstring "Summarize the given text." is replaced by the instructions
        ```

    === "With examples"

        Combine examples from multiple sources:

        ```python
        # pip install msgflux[openai]        
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Classify(mf.Signature):
            """Classify sentiment."""
            text: str = mf.InputField()
            sentiment: str = mf.OutputField()

        class Classifier(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = Classify
            # These examples are *combined* with any examples in the signature
            examples = [
                mf.Example(
                    inputs={"text": "I love it!"},
                    outputs={"sentiment": "positive"}
                ),
                mf.Example(
                    inputs={"text": "Terrible product."},
                    outputs={"sentiment": "negative"}
                ),
            ]
        ```

    === "With generation_schema"

        Fuse reasoning strategies with typed outputs:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn        
        from msgflux.generation.reasoning import ChainOfThought

        # mf.set_envs(OPENAI_API_KEY="...")

        class Calculate(mf.Signature):
            """Solve the math problem."""
            problem: str = mf.InputField()
            answer: float = mf.OutputField()

        class Calculator(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = Calculate
            # ChainOfThought is FUSED with the signature output
            generation_schema = ChainOfThought

        agent = Calculator()
        response = agent(problem="What is 15% of 80?")
        print(response)
        ```

!!! warning "Task Template is Overwritten"
    If you define both a `signature` and a `task` template in `templates`, the signature will **overwrite** your task template. Use `context_inputs` or `system_message` for additional context instead.

    ```python
    # pip install msgflux[openai]
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    # Don't do this - task template will be overwritten by signature
    class Agent(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        signature = "text -> summary"
        templates = {"task": "Please summarize: {{ text }}"}  # Ignored!

    # Do this instead - use context_inputs for extra context
    response = agent(text="...", context_inputs="Focus on technical details")
    print(response)
    ```

### Combining with Reasoning Strategies

Signatures work seamlessly with reasoning strategies like Chain of Thought:

???+ example "Signature + Chain of Thought"

    ```python
    # pip install msgflux[openai]
    import msgflux as mf
    import msgflux.nn as nn
    from msgflux.generation.reasoning import ChainOfThought

    # mf.set_envs(OPENAI_API_KEY="...")

    class MathSolver(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        signature = "question -> answer: float"
        generation_schema = ChainOfThought

    agent = MathSolver()
    response = agent(question="Two dice are tossed. What is the probability the sum equals 2?")
    print(response)
    ```

### Signatures as Tools

When an agent has a signature, its annotations are automatically configured based on the input fields. This makes it **ready to be used as a tool** with properly typed parameters.

???+ example "Signature-Based Agent as Tool"

    ```python
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    # Without signature: default annotation is "message: str"
    class BasicAgent(nn.Agent):
        model = model

    print(BasicAgent().annotations)  # {"message": str}

    # With signature: annotations match the input fields
    class AnalyzeSentiment(mf.Signature):
        """Analyze the sentiment of text."""
        text: str = mf.InputField(desc="Text to analyze")
        sentiment: str = mf.OutputField()
        score: float = mf.OutputField(desc="Score from -1 to 1")

    class SentimentAnalyzer(nn.Agent):
        model = model
        signature = AnalyzeSentiment

    print(SentimentAnalyzer().annotations)  # {"text": str}

    # Use as a tool - the coordinator sees: analyze_sentiment(text: str)
    class Coordinator(nn.Agent):
        model = model
        tools = [SentimentAnalyzer]
        system_message = "You help analyze customer feedback."
        config = {"verbose": True}

    coordinator = Coordinator()
    response = coordinator("What's the sentiment of 'I absolutely love this product!'")
    print(response)
    ```

!!! tip "Best Practices"
    - **Start simple**: Begin with inline signatures and evolve to class-based as needed
    - **Be semantic**: Choose clear, meaningful field names (e.g., `question` not `q`)
    - **Use descriptions**: Add `desc` for ambiguous fields or specific constraints
    - **Docstrings matter**: The class docstring becomes the model's instruction
    - **Trust the system**: Avoid over-engineering prompts in descriptions

---

## Guardrails

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

---

## Model Gateway

When using a `ModelGateway` with multiple models, you can specify which model to use via `model_preference`:

```python
# pip install msgflux[openai]
import msgflux as mf
import msgflux.nn as nn

# mf.set_envs(OPENAI_API_KEY="...")

# Create models with different costs/capabilities
low_cost = mf.Model.chat_completion("openai/gpt-4.1-mini")
high_quality = mf.Model.chat_completion("openai/gpt-5.2")

# Create gateway
gateway = mf.ModelGateway([low_cost, high_quality])

agent = nn.Agent("agent", gateway)

# Use specific model for simple tasks
response = agent("Tell me a joke", model_preference="gpt-4.1-mini")

# Use better model for complex tasks
response = agent("Analyze this contract...", model_preference="gpt-4.1")
```

---

## Prefilling

Force an initial message that the model will continue from. Useful for guiding response format or triggering specific behavior.

```python
# pip install msgflux[openai]
import msgflux as mf
import msgflux.nn as nn

# mf.set_envs(OPENAI_API_KEY="...")

class Assistant(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

# Encourage step-by-step reasoning
agent = Assistant(prefilling="Let me solve this step by step.")
response = agent(
    "What is the derivative of x^(2/3)?",
)

# Force specific format
agent = Assistant(prefilling="Here are the planets:\n1.")
response = agent(
    "List the planets in our solar system",
)
```

---

## See Also

- [Module](module.md) - Base class for all nn components
- [Tool](tool.md) - Tool system details
- [Message](../message.md) - Structured message passing
- [Model](../models/model.md) - Model factory and types
