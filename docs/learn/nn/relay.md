# nn.Relay

## ✦₊⁺ Overview

`nn.Relay` is a declarative base module that provides `message_fields` and `response_mode` scaffolding **without requiring a model**. Subclass it and implement `forward()` to build custom processing modules that read from a `Message` and write back to it.

Think of it as a **Predictor without a model** — pure data transformation with the same declarative ergonomics as every other nn module.

---

## 1. **Quick Start**

!!! info "Initialization styles"

    === "Declarative (recommended)"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        class TextCleaner(nn.Relay):
            """Cleans and normalizes text input."""
            message_fields = {"task": "inputs.raw_text"}
            response_mode  = "inputs.clean_text"

            def forward(self, message, **kwargs):
                text = self._extract_message_values(self.task, message)
                return self._define_response_mode(text.strip().lower(), message)

        cleaner = TextCleaner()

        msg = mf.Message()
        msg.set("inputs.raw_text", "  HELLO WORLD  ")
        cleaner(msg)
        print(msg.get("inputs.clean_text"))  # "hello world"
        ```

    === "Direct"

        ```python
        import msgflux.nn as nn

        relay = nn.Relay(
            message_fields={"task": "inputs.data"},
            response_mode="outputs.result",
        )
        # Note: Direct instantiation requires overriding forward
        # before calling — use subclassing for real use cases.
        ```

---

## 2. **Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `modules` | `dict[str, Module] \| None` | Submodules registered as attributes on `self`. Access them via `self.<key>` in `forward()` |
| `message_fields` | `dict \| None` | Map `Message` field names to input paths. Valid keys: `task`, `task_multimodal`, `model_preference` |
| `response_mode` | `str \| None` | Field path on the `Message` where the result is written. `None` returns the result directly |
| `annotations` | `dict[str, type] \| None` | Type annotations for schema generation |
| `description` | `str \| None` | Module description. Auto-populated from docstring via `AutoParams` |
| `hooks` | `list[Hook] \| None` | Hook instances registered on the module |
| `name` | `str \| None` | Module name in snake_case. Auto-populated from class name via `AutoParams` |

---

## 3. **Core Pattern**

Every `Relay` subclass follows the same three-step pattern:

```python
class MyRelay(nn.Relay):
    message_fields = {"task": "path.to.input"}
    response_mode  = "path.to.output"

    def forward(self, message, **kwargs):
        # 1. Extract — read data from the message
        data = self._extract_message_values(self.task, message)

        # 2. Transform — apply your logic
        result = your_logic(data)

        # 3. Respond — write back or return directly
        return self._define_response_mode(result, message)
```

**What each step does:**

| Step | Method | Behavior |
|------|--------|----------|
| Extract | `_extract_message_values(self.task, message)` | Reads data from the path defined in `message_fields["task"]` |
| Transform | *(your code)* | Any Python logic — no restrictions |
| Respond | `_define_response_mode(result, message)` | If `response_mode` is set, writes to `message` and returns `None`. Otherwise returns `result` directly |

---

## 4. **Guides**

### 4.1 Registering Submodules

Use `modules` to register pre-built modules on `self` — no need to override `__init__`:

!!! info "Registration styles"

    === "Declarative (class attribute)"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        class SummarizePipeline(nn.Relay):
            """Summarizes text using an agent."""
            modules = {
                "agent": nn.Agent(
                    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
                    prompt="Summarize the following text in one sentence.",
                ),
            }
            message_fields = {"task": "inputs.text"}
            response_mode  = "outputs.summary"

            def forward(self, message, **kwargs):
                text = self._extract_message_values(self.task, message)
                result = self.agent(text)
                return self._define_response_mode(result, message)

        pipeline = SummarizePipeline()
        ```

    === "Direct (constructor)"

        ```python
        pipeline = SummarizePipeline(
            modules={
                "agent": nn.Agent(
                    model=mf.Model.chat_completion("openai/gpt-4.1-nano"),
                ),
            },
        )
        # self.agent is now the nano model instead
        ```

    === "Multiple modules"

        ```python
        class ReviewPipeline(nn.Relay):
            modules = {
                "summarizer": nn.Agent(
                    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
                    prompt="Summarize this review.",
                ),
                "sentiment": nn.Predictor(
                    model=mf.Model.text_classifier("vllm/sentiment-model"),
                ),
            }
            message_fields = {"task": "review.text"}

            def forward(self, message, **kwargs):
                text = self._extract_message_values(self.task, message)
                summary = self.summarizer(text)
                label = self.sentiment(text)
                result = {"summary": summary, "sentiment": label}
                return self._define_response_mode(result, message)
        ```

Registered modules are proper submodules — they appear in `named_modules()`, are included in `state_dict()`, and benefit from telemetry:

```python
pipeline = SummarizePipeline()
for name, module in pipeline.named_modules():
    print(name, type(module).__name__)
# ''     SummarizePipeline
# agent  Agent
```

---

### 4.2 Data Validation

Validate or filter data flowing through a pipeline:

```python
import msgflux as mf
import msgflux.nn as nn

class InputValidator(nn.Relay):
    """Validates that required fields are present and non-empty."""
    message_fields = {"task": "form.data"}
    response_mode  = "form.validated"

    def forward(self, message, **kwargs):
        data = self._extract_message_values(self.task, message)

        errors = []
        for field in ("name", "email"):
            if not data.get(field):
                errors.append(f"Missing required field: {field}")

        result = {"valid": len(errors) == 0, "errors": errors, "data": data}
        return self._define_response_mode(result, message)

validator = InputValidator()

msg = mf.Message()
msg.set("form.data", {"name": "Alice", "email": ""})
validator(msg)
print(msg.get("form.validated"))
# {"valid": False, "errors": ["Missing required field: email"], "data": {...}}
```

---

### 4.2 Data Enrichment

Enrich data with external lookups or computed fields:

```python
import msgflux as mf
import msgflux.nn as nn

PRICING = {"basic": 9.99, "pro": 29.99, "enterprise": 99.99}

class PriceEnricher(nn.Relay):
    """Adds pricing info to a subscription record."""
    message_fields = {"task": "subscription.plan"}
    response_mode  = "subscription.pricing"

    def forward(self, message, **kwargs):
        plan = self._extract_message_values(self.task, message)
        pricing = {
            "plan": plan,
            "price": PRICING.get(plan, 0),
            "currency": "USD",
        }
        return self._define_response_mode(pricing, message)

enricher = PriceEnricher()

msg = mf.Message()
msg.set("subscription.plan", "pro")
enricher(msg)
print(msg.get("subscription.pricing"))
# {"plan": "pro", "price": 29.99, "currency": "USD"}
```

---

### 4.3 Multiple Inputs (dict paths)

Read multiple fields from the message at once using a dict in `message_fields`:

```python
import msgflux as mf
import msgflux.nn as nn

class FullNameBuilder(nn.Relay):
    """Combines first and last name into a full name."""
    message_fields = {
        "task": {
            "first": "user.first_name",
            "last": "user.last_name",
        }
    }
    response_mode = "user.full_name"

    def forward(self, message, **kwargs):
        data = self._extract_message_values(self.task, message)
        full_name = f"{data.first} {data.last}"
        return self._define_response_mode(full_name, message)

builder = FullNameBuilder()

msg = mf.Message()
msg.set("user.first_name", "Alice")
msg.set("user.last_name", "Smith")
builder(msg)
print(msg.get("user.full_name"))  # "Alice Smith"
```

---

### 4.4 OR Inputs (fallback paths)

Use a tuple to try multiple paths — the first non-`None` value wins:

```python
import msgflux as mf
import msgflux.nn as nn

class TextExtractor(nn.Relay):
    """Extracts text from whichever field is available."""
    message_fields = {
        "task": ("content.body", "content.summary", "content.title")
    }
    response_mode = "extracted_text"

    def forward(self, message, **kwargs):
        text = self._extract_message_values(self.task, message)
        return self._define_response_mode(text, message)

extractor = TextExtractor()

msg = mf.Message()
msg.set("content.title", "Breaking News")
extractor(msg)
print(msg.get("extracted_text"))  # "Breaking News"
```

---

### 4.5 Direct Return (no response_mode)

When `response_mode` is `None`, the result is returned directly instead of being written to the message:

```python
import msgflux.nn as nn

class WordCounter(nn.Relay):
    """Counts words in the input text."""
    message_fields = {"task": "text"}

    def forward(self, message, **kwargs):
        text = self._extract_message_values(self.task, message)
        count = len(text.split())
        return self._define_response_mode(count, message)

counter = WordCounter()

msg = mf.dotdict(text="hello world foo bar")
result = counter(msg)
print(result)  # 4
```

---

### 4.6 Async Support

Implement `aforward` for async operations:

```python
import asyncio
import httpx
import msgflux as mf
import msgflux.nn as nn

class GeoLookup(nn.Relay):
    """Looks up geolocation from an IP address."""
    message_fields = {"task": "request.ip"}
    response_mode  = "request.geo"

    async def aforward(self, message, **kwargs):
        ip = self._extract_message_values(self.task, message)
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"https://ipapi.co/{ip}/json/")
            geo = resp.json()
        return self._define_response_mode(geo, message)

lookup = GeoLookup()

msg = mf.Message()
msg.set("request.ip", "8.8.8.8")
asyncio.run(lookup.acall(msg))
print(msg.get("request.geo.country_name"))
```

---

### 4.7 Guardrails with Hooks

Use `Guard` hooks for input validation — no need to add validation logic inside `forward()`:

```python
import msgflux.nn as nn
from msgflux.nn.hooks import Guard

def not_empty(data):
    return {"safe": data is not None and str(data).strip() != ""}

class SafeRelay(nn.Relay):
    message_fields = {"task": "inputs.text"}
    response_mode  = "outputs.result"
    hooks = [
        Guard(
            validator=not_empty,
            on="pre",
            message="Empty input, skipped.",
        )
    ]

    def forward(self, message, **kwargs):
        text = self._extract_message_values(self.task, message)
        return self._define_response_mode(text.upper(), message)
```

---

## 5. **Pipelines**

### 5.1 Sequential Composition

Chain Relays with other modules in a `Sequential` pipeline:

```python
import msgflux as mf
import msgflux.nn as nn

class Normalizer(nn.Relay):
    message_fields = {"task": "inputs.raw"}
    response_mode  = "inputs.normalized"

    def forward(self, message, **kwargs):
        text = self._extract_message_values(self.task, message)
        result = text.strip().lower()
        return self._define_response_mode(result, message)

class Tokenizer(nn.Relay):
    message_fields = {"task": "inputs.normalized"}
    response_mode  = "inputs.tokens"

    def forward(self, message, **kwargs):
        text = self._extract_message_values(self.task, message)
        tokens = text.split()
        return self._define_response_mode(tokens, message)

pipeline = nn.Sequential(Normalizer(), Tokenizer())

msg = mf.Message()
msg.set("inputs.raw", "  Hello World  ")
pipeline(msg)
print(msg.get("inputs.tokens"))  # ["hello", "world"]
```

### 5.2 Pre/Post Processing with Agents

Use Relays as preprocessing and postprocessing steps around an Agent:

```python
import msgflux as mf
import msgflux.nn as nn

class InputFormatter(nn.Relay):
    """Formats raw user input for the agent."""
    message_fields = {"task": "user_input"}
    response_mode  = "agent_input"

    def forward(self, message, **kwargs):
        raw = self._extract_message_values(self.task, message)
        formatted = f"User request: {raw}\nPlease respond concisely."
        return self._define_response_mode(formatted, message)

class ResponseParser(nn.Relay):
    """Extracts structured data from agent response."""
    message_fields = {"task": "agent_output"}
    response_mode  = "parsed_output"

    def forward(self, message, **kwargs):
        text = self._extract_message_values(self.task, message)
        return self._define_response_mode(
            {"text": text, "length": len(text)}, message
        )

class Assistant(nn.Agent):
    model          = mf.Model.chat_completion("openai/gpt-4.1-mini")
    message_fields = {"task": "agent_input"}
    response_mode  = "agent_output"

class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.formatter = InputFormatter()
        self.agent = Assistant()
        self.parser = ResponseParser()

    def forward(self, message, **kwargs):
        self.formatter(message)
        self.agent(message)
        self.parser(message)
        return message
```

### 5.3 Conditional Routing

Use Relays to implement routing logic:

```python
import msgflux as mf
import msgflux.nn as nn

class LanguageDetector(nn.Relay):
    """Detects the language and routes accordingly."""
    message_fields = {"task": "inputs.text"}
    response_mode  = "inputs.language"

    PATTERNS = {"hola": "es", "bonjour": "fr", "hello": "en"}

    def forward(self, message, **kwargs):
        text = self._extract_message_values(self.task, message)
        first_word = text.strip().split()[0].lower()
        lang = self.PATTERNS.get(first_word, "en")
        return self._define_response_mode(lang, message)

class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = LanguageDetector()
        self.agents = nn.ModuleDict({
            "en": EnglishAgent(),
            "es": SpanishAgent(),
            "fr": FrenchAgent(),
        })

    def forward(self, message, **kwargs):
        self.detector(message)
        lang = message.get("inputs.language")
        agent = self.agents.get(lang, self.agents["en"])
        return agent(message)
```

---

## 6. **Hierarchies**

Share configuration across related relays via inheritance:

```python
import msgflux.nn as nn

class BaseFormatter(nn.Relay):
    """Base class for all text formatters."""
    response_mode = "formatted_output"

class MarkdownFormatter(BaseFormatter):
    message_fields = {"task": "inputs.markdown"}

    def forward(self, message, **kwargs):
        text = self._extract_message_values(self.task, message)
        result = text.replace("**", "<b>").replace("**", "</b>")
        return self._define_response_mode(result, message)

class CSVFormatter(BaseFormatter):
    message_fields = {"task": "inputs.rows"}

    def forward(self, message, **kwargs):
        rows = self._extract_message_values(self.task, message)
        csv = "\n".join(",".join(str(c) for c in row) for row in rows)
        return self._define_response_mode(csv, message)
```

---

## 7. **AutoParams**

`Relay` uses `AutoParams` to automatically populate `name` and `description` from the class:

```python
import msgflux.nn as nn

class EmailSanitizer(nn.Relay):
    """Removes PII from email content before processing."""

    def forward(self, message, **kwargs):
        ...

sanitizer = EmailSanitizer()
print(sanitizer.name)         # "EmailSanitizer"
print(sanitizer.description)  # "Removes PII from email content before processing."
```

This is useful when relays are introspected, logged, or composed into larger systems.

---

## 8. **Annotations**

Define type annotations for schema generation or documentation:

```python
import msgflux.nn as nn

class TypedRelay(nn.Relay):
    annotations = {"input": str, "output": dict}

    def forward(self, message, **kwargs):
        ...

relay = TypedRelay()
print(relay.annotations)  # {"input": <class 'str'>, "output": <class 'dict'>}
```

---

## 9. **When to Use Relay**

| Scenario | Module |
|----------|--------|
| Data transformation without a model | **Relay** |
| Model inference (classification, moderation) | Predictor |
| Conversational AI with tools | Agent |
| Text-to-speech | Speaker |
| Embeddings | Embedder |

Use `Relay` when you need the declarative `message_fields`/`response_mode` pattern but your logic is pure Python — no model calls involved.
