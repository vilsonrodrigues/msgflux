# AutoParams

AutoParams is a metaclass that enables elegant, dataclass-style module definitions in msgflux. It automatically captures class attributes as default parameters, eliminating boilerplate code while maintaining full OOP capabilities.

## Overview

AutoParams transforms verbose module definitions into clean, declarative configurations:

**Traditional Approach** (verbose):
```python
class DataProcessor:
    def __init__(self, batch_size, timeout, max_retries):
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries

    def process(self, data):
        # Processing logic here
        pass

# Every instance needs all parameters
fast_processor = DataProcessor(batch_size=100, timeout=10, max_retries=3)
slow_processor = DataProcessor(batch_size=10, timeout=60, max_retries=5)
```

**With AutoParams** (clean):
```python
import msgflux as mf

# Define base class with logic once
class DataProcessor(metaclass=mf.AutoParams):
    def __init__(self, batch_size, timeout, max_retries):
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries

    def process(self, data):
        # Processing logic here
        pass

# Create variants by just setting defaults
class FastProcessor(DataProcessor):
    batch_size = 100
    timeout = 10
    max_retries = 3

class SlowProcessor(DataProcessor):
    batch_size = 10
    timeout = 60
    max_retries = 5

# Clean instantiation
fast = FastProcessor()  # Uses defaults
slow = SlowProcessor()  # Uses defaults
custom = FastProcessor(batch_size=200)  # Override specific params
```

## Why AutoParams?

### Benefits

1. **Eliminates Boilerplate** - No repetitive `__init__` assignments
2. **Declarative Configuration** - Parameters as class attributes
3. **Flexible Defaults** - Easy to override during instantiation
4. **Inheritance Support** - Build module hierarchies naturally
5. **Documentation Integration** - Docstrings and class names as parameters
6. **PyTorch-Like** - Familiar pattern for PyTorch users

### When to Use

- ✅ Defining `nn.Module` subclasses (Agent, Retriever, etc.)
- ✅ Creating configurable modules with many parameters
- ✅ Building module families with shared defaults
- ✅ When you want clean, readable code

## Quick Start

### Basic Usage

```python
import msgflux as mf

class Configuration(metaclass=mf.AutoParams):
    """Simple configuration class."""

    def __init__(self, host, port, debug):
        self.host = host
        self.port = port
        self.debug = debug

class DevelopmentConfig(Configuration):
    """Development configuration."""
    host = "localhost"
    port = 8000
    debug = True

class ProductionConfig(Configuration):
    """Production configuration."""
    host = "0.0.0.0"
    port = 443
    debug = False

# Use defaults
dev = DevelopmentConfig()
print(dev.host)  # "localhost"
print(dev.debug)  # True

# Override specific parameters
custom_dev = DevelopmentConfig(port=3000)
print(custom_dev.port)  # 3000
print(custom_dev.host)  # "localhost" (still default)

# Production instance
prod = ProductionConfig()
print(prod.debug)  # False
```

## Core Features

### 1. Docstring as Parameter

Use the class docstring as a parameter value:

```python
import msgflux as mf
import msgflux.nn as nn

class Agent(nn.Module, metaclass=mf.AutoParams):
    """Configurable agent base class."""

    _autoparams_use_docstring_for = "description"

    def __init__(self, name, description):
        super().__init__()
        self.name = name
        self.description = description

class ResearchAgent(Agent):
    """An AI agent specialized in research tasks with web search capabilities."""
    name = "research_assistant"

agent = ResearchAgent()
print(agent.name)  # "research_assistant"
print(agent.description)  # "An AI agent specialized in research tasks..."
```

### 2. Class Name as Parameter

Automatically use the class name as a parameter:

```python
import msgflux as mf
import msgflux.nn as nn

class Agent(nn.Module, metaclass=mf.AutoParams):
    """Base agent with automatic naming."""

    _autoparams_use_classname_for = "name"

    def __init__(self, name):
        super().__init__()
        self.name = name

class CustomerSupportAgent(Agent):
    pass

class SalesAgent(Agent):
    pass

support = CustomerSupportAgent()
print(support.name)  # "CustomerSupportAgent"

sales = SalesAgent()
print(sales.name)  # "SalesAgent"
```

### 3. Combined: Docstring + Class Name

Use both features together:

```python
import msgflux as mf
import msgflux.nn as nn

class Agent(nn.Module, metaclass=mf.AutoParams):
    """Base agent class."""

    _autoparams_use_docstring_for = "description"
    _autoparams_use_classname_for = "name"

    temperature = 0.7
    max_tokens = 2000

    def __init__(self, name, description, temperature, max_tokens):
        super().__init__()
        self.name = name
        self.description = description
        self.temperature = temperature
        self.max_tokens = max_tokens

class DataAnalyst(Agent):
    """An agent that analyzes data and generates insights."""
    temperature = 0.2  # Override: more deterministic for analysis

analyst = DataAnalyst()
print(analyst.name)  # "DataAnalyst"
print(analyst.description)  # "An agent that analyzes data..."
print(analyst.temperature)  # 0.2
print(analyst.max_tokens)  # 2000 (inherited)
```

## Integration with nn.Module

AutoParams is designed to work seamlessly with msgflux neural network modules:

### Basic Module Example

```python
import msgflux as mf
import msgflux.nn as nn

# Define base module with initialization logic
class CustomModule(nn.Module, metaclass=mf.AutoParams):
    """Base custom module."""

    def __init__(self, temperature, max_tokens, enable_cache):
        super().__init__()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_cache = enable_cache

    def forward(self, msg):
        # Module logic here
        return msg

# Create variants with different defaults
class ConservativeModule(CustomModule):
    """Conservative settings for production."""
    temperature = 0.2
    max_tokens = 1000
    enable_cache = True

class CreativeModule(CustomModule):
    """Creative settings for exploration."""
    temperature = 0.9
    max_tokens = 2000
    enable_cache = False

# Use with defaults
conservative = ConservativeModule()
print(conservative.temperature)  # 0.2
print(conservative.enable_cache)  # True

# Override specific parameters
custom = CreativeModule(temperature=0.5)
print(custom.temperature)  # 0.5
print(custom.enable_cache)  # False (still default)
```

### Retriever Example

```python
import msgflux as mf
import msgflux.nn as nn

class DocumentRetriever(nn.Retriever):
    """Semantic retriever for technical documentation."""

    response_mode = "plain_response"
    top_k = 5
    threshold = 0.7

# Create vector database and embedder
vector_db = mf.data.VectorDB.qdrant(
    collection_name="docs",
    url="http://localhost:6333"
)
embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

# Use with defaults
retriever = DocumentRetriever(
    retriever=vector_db,
    model=embedder
)

# Or customize
strict_retriever = DocumentRetriever(
    retriever=vector_db,
    model=embedder,
    top_k=10,      # More results
    threshold=0.8  # Higher similarity required
)
```

### Speaker Example

```python
import msgflux as mf
import msgflux.nn as nn

class NaturalSpeaker(nn.Speaker):
    """Natural-sounding voice for user-facing applications."""

    response_format = "mp3"
    response_mode = "plain_response"
    voice = "alloy"
    speed = 1.0

class FastSpeaker(NaturalSpeaker):
    """Faster speech for time-constrained scenarios."""
    speed = 1.25

tts = mf.Model.text_to_speech("openai/tts-1")

# Normal speed
normal = NaturalSpeaker(model=tts)
audio1 = normal("Hello, how can I help you?")

# Fast speed
fast = FastSpeaker(model=tts)
audio2 = fast("Hello, how can I help you?")
```

## Inheritance Patterns

### Building Module Hierarchies

Create families of related modules:

```python
import msgflux as mf
import msgflux.nn as nn

# Base module with common configuration
class BaseModule(nn.Module, metaclass=mf.AutoParams):
    """Base module with standard configuration."""

    def __init__(self, temperature, max_tokens, enable_cache):
        super().__init__()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_cache = enable_cache

    def forward(self, msg):
        # Processing logic
        return msg

# Specialized for different use cases
class CreativeModule(BaseModule):
    """Module optimized for creative tasks."""
    temperature = 0.9  # More random/creative
    max_tokens = 3000  # Longer outputs
    enable_cache = False

class AnalyticalModule(BaseModule):
    """Module optimized for analytical tasks."""
    temperature = 0.2  # More deterministic
    max_tokens = 1500  # Focused responses
    enable_cache = True

class CodingModule(AnalyticalModule):
    """Module specialized in coding tasks."""
    max_tokens = 4000  # Longer code blocks

# Create instances
creative = CreativeModule()
print(creative.temperature)  # 0.9

analytical = AnalyticalModule()
print(analytical.temperature)  # 0.2

coder = CodingModule()
print(coder.temperature)  # 0.2 (inherited from AnalyticalModule)
print(coder.max_tokens)  # 4000 (overridden)
```

### Multi-Level Inheritance

```python
import msgflux as mf

class BaseConfig(metaclass=mf.AutoParams):
    """Base configuration."""

    def __init__(self, timeout, retry_attempts):
        self.timeout = timeout
        self.retry_attempts = retry_attempts

class NetworkConfig(BaseConfig):
    """Network configuration."""
    timeout = 30
    retry_attempts = 3

class DatabaseConfig(NetworkConfig):
    """Database configuration inherits network settings."""
    connection_pool_size = 10

class ProductionDatabaseConfig(DatabaseConfig):
    """Production database with stricter settings."""
    retry_attempts = 5  # Override
    connection_pool_size = 50  # Override

# All parameters available
prod_db = ProductionDatabaseConfig()
print(prod_db.timeout)  # 30 (from NetworkConfig)
print(prod_db.retry_attempts)  # 5 (overridden)
print(prod_db.connection_pool_size)  # 50 (overridden)
```

## Advanced Patterns

### Dynamic Defaults

Use class methods or properties for computed defaults:

```python
import msgflux as mf
import msgflux.nn as nn
import os

class APIModule(nn.Module, metaclass=mf.AutoParams):
    """Module that calls external APIs."""

    def __init__(self, api_key, timeout, max_retries):
        super().__init__()
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def forward(self, msg):
        # API call logic
        return msg

class OpenAIModule(APIModule):
    """OpenAI-specific API module."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    timeout = 30
    max_retries = 3

# api_key is automatically loaded from environment
module = OpenAIModule()
print(module.api_key)  # Value from OPENAI_API_KEY env var
```

### Conditional Configuration

```python
import msgflux as mf
import msgflux.nn as nn

class Environment:
    DEVELOPMENT = "development"
    PRODUCTION = "production"

class ConfigurableModule(nn.Module, metaclass=mf.AutoParams):
    """Module with environment-specific defaults."""

    def __init__(self, environment, debug_mode, error_reporting):
        super().__init__()
        self.environment = environment
        self.debug_mode = debug_mode
        self.error_reporting = error_reporting

    def forward(self, msg):
        if self.debug_mode:
            print(f"Debug: Processing in {self.environment} mode")
        return msg

class DevelopmentModule(ConfigurableModule):
    """Development module."""
    environment = Environment.DEVELOPMENT
    debug_mode = True
    error_reporting = False

class ProductionModule(ConfigurableModule):
    """Production-ready module."""
    environment = Environment.PRODUCTION
    debug_mode = False
    error_reporting = True

dev = DevelopmentModule()
print(dev.debug_mode)  # True

prod = ProductionModule()
print(prod.debug_mode)  # False
print(prod.error_reporting)  # True
```

## Real-World Examples

### Multi-Language Support System

```python
import msgflux as mf
import msgflux.nn as nn

class TranscriptionModule(nn.Module, metaclass=mf.AutoParams):
    """Base transcription module."""

    def __init__(self, language, prompt, temperature):
        super().__init__()
        self.language = language
        self.prompt = prompt
        self.temperature = temperature

    def forward(self, msg):
        # Transcription logic
        return msg

class EnglishTranscriber(TranscriptionModule):
    """Optimized for English transcription."""
    language = "en"
    prompt = "Transcribe with proper punctuation and capitalization."
    temperature = 0.0

class SpanishTranscriber(TranscriptionModule):
    """Optimized for Spanish transcription."""
    language = "es"
    prompt = "Transcribir con puntuación y capitalización apropiadas."
    temperature = 0.0

class MultilingualTranscriber(TranscriptionModule):
    """Auto-detects language."""
    language = None
    prompt = "Transcribe in the detected language."
    temperature = 0.0

# Create language-specific instances
en_transcriber = EnglishTranscriber()
es_transcriber = SpanishTranscriber()
multi_transcriber = MultilingualTranscriber()
```

### Content Moderation Pipeline

```python
import msgflux as mf
import msgflux.nn as nn

class ModerationModule(nn.Module, metaclass=mf.AutoParams):
    """Base moderation module."""

    def __init__(self, system_message, threshold, temperature):
        super().__init__()
        self.system_message = system_message
        self.threshold = threshold
        self.temperature = temperature

    def forward(self, msg):
        # Moderation logic
        return msg

class StrictModerator(ModerationModule):
    """Strict moderation for public content."""
    system_message = "You are a strict content moderator. Flag any potentially harmful content."
    threshold = 0.3  # Low threshold = more strict
    temperature = 0.1

class LenientModerator(ModerationModule):
    """Lenient moderation for private communities."""
    system_message = "You are a lenient content moderator. Only flag clearly harmful content."
    threshold = 0.7  # High threshold = less strict
    temperature = 0.1

class ChildSafeModerator(StrictModerator):
    """Ultra-strict moderation for child-safe content."""
    threshold = 0.1  # Very low threshold

# Different moderation levels
strict = StrictModerator()
lenient = LenientModerator()
child_safe = ChildSafeModerator()
```

### RAG System with Different Retrieval Strategies

```python
import msgflux as mf
import msgflux.nn as nn

class SemanticRetriever(nn.Retriever):
    """Base semantic retriever."""
    response_mode = "plain_response"
    top_k = 5

class PreciseRetriever(SemanticRetriever):
    """High-precision retrieval."""
    top_k = 3
    threshold = 0.85  # Only very similar results

class BroadRetriever(SemanticRetriever):
    """Broad retrieval for exploration."""
    top_k = 10
    threshold = 0.6  # More permissive

class HybridRetriever(SemanticRetriever):
    """Combines semantic and keyword search."""
    top_k = 7
    threshold = 0.7
    use_keyword_boost = True

vector_db = mf.data.VectorDB.qdrant(
    collection_name="knowledge_base",
    url="http://localhost:6333"
)
embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

# Different retrieval strategies
precise = PreciseRetriever(retriever=vector_db, model=embedder)
broad = BroadRetriever(retriever=vector_db, model=embedder)
hybrid = HybridRetriever(retriever=vector_db, model=embedder)
```

## Best Practices

### 1. Use Descriptive Class Names

```python
import msgflux.nn as nn

# Good - Clear purpose
class CustomerSupportModule(nn.Module):
    """Module for customer support inquiries."""
    temperature = 0.7

# Avoid - Vague names
class Module1(nn.Module):
    temperature = 0.7
```

### 2. Document with Docstrings

```python
import msgflux as mf
import msgflux.nn as nn

# Good - Clear documentation
class AnalyticalModule(nn.Module, metaclass=mf.AutoParams):
    """
    Module optimized for analytical and data-driven tasks.

    Uses low temperature for consistent, factual responses.
    Suitable for data analysis, reporting, and calculations.
    """

    def __init__(self, temperature, max_tokens):
        super().__init__()
        self.temperature = temperature
        self.max_tokens = max_tokens

class DataAnalyzer(AnalyticalModule):
    temperature = 0.2
    max_tokens = 2000

# Avoid - No documentation
class MyModule(nn.Module, metaclass=mf.AutoParams):
    def __init__(self, temperature, max_tokens):
        super().__init__()
        self.temperature = temperature
        self.max_tokens = max_tokens

class Analyzer(MyModule):
    temperature = 0.2
    max_tokens = 2000
```

### 3. Group Related Configurations

```python
import msgflux as mf
import msgflux.nn as nn

# Good - Logical groupings
class BaseModule(nn.Module, metaclass=mf.AutoParams):
    def __init__(self, temperature, max_tokens, enable_cache, verbose):
        super().__init__()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_cache = enable_cache
        self.verbose = verbose

class ProductionModule(BaseModule):
    # Model parameters
    temperature = 0.7
    max_tokens = 2000

    # Behavior parameters
    enable_cache = True
    verbose = False
```

### 4. Use Inheritance for Variants

```python
# Good - Base + variants
class BaseRetriever(nn.Retriever):
    response_mode = "plain_response"
    top_k = 5

class FastRetriever(BaseRetriever):
    top_k = 3  # Fewer results for speed

class ThoroughRetriever(BaseRetriever):
    top_k = 10  # More results for coverage

# Avoid - Duplicating everything
class FastRetriever(nn.Retriever):
    response_mode = "plain_response"  # Duplicated
    top_k = 3

class ThoroughRetriever(nn.Retriever):
    response_mode = "plain_response"  # Duplicated
    top_k = 10
```

### 5. Provide Sensible Defaults

```python
import msgflux as mf
import msgflux.nn as nn

class WebSearchModule(nn.Module, metaclass=mf.AutoParams):
    """Base web search module."""

    def __init__(self, max_results, timeout, retry_attempts):
        super().__init__()
        self.max_results = max_results
        self.timeout = timeout
        self.retry_attempts = retry_attempts

# Good - Reasonable defaults
class StandardSearch(WebSearchModule):
    max_results = 5  # Reasonable number
    timeout = 30     # Reasonable timeout
    retry_attempts = 3  # Reasonable retries

# Avoid - Extreme or unclear defaults
class BadSearch(WebSearchModule):
    max_results = 1000  # Too many
    timeout = 1      # Too short
    retry_attempts = 100  # Too many
```

## API Reference

### Special Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `_autoparams_use_docstring_for` | str | Use class docstring as value for this parameter |
| `_autoparams_use_classname_for` | str | Use class name as value for this parameter |

### How It Works

1. **Class Definition**: AutoParams captures class attributes (except methods and special attributes)
2. **Initialization**: When instantiated, defaults from class attributes are used
3. **Override**: Parameters passed to `__init__` override defaults
4. **Inheritance**: Child classes inherit parent defaults and can override them

## Comparison: Traditional vs AutoParams

### Traditional Approach

```python
class TraditionalModule:
    def __init__(self, temperature=0.7, max_tokens=2000, enable_cache=True, verbose=False):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_cache = enable_cache
        self.verbose = verbose

    def process(self, data):
        # Processing logic
        pass

# Every variant needs all parameters repeated
fast = TraditionalModule(temperature=0.8, max_tokens=1500, enable_cache=True, verbose=False)
slow = TraditionalModule(temperature=0.2, max_tokens=3000, enable_cache=True, verbose=True)
```

**Issues:**
- Repetitive parameter assignments
- Constructor becomes long with many parameters
- Defaults buried in function signature
- Hard to create variants without repeating all parameters
- Configuration not visible at class level

### With AutoParams

```python
import msgflux as mf
import msgflux.nn as nn

class BaseModule(nn.Module, metaclass=mf.AutoParams):
    """Base module with AutoParams."""

    def __init__(self, temperature, max_tokens, enable_cache, verbose):
        super().__init__()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_cache = enable_cache
        self.verbose = verbose

    def process(self, data):
        # Processing logic
        pass

# Create variants by just setting defaults
class FastModule(BaseModule):
    temperature = 0.8
    max_tokens = 1500
    enable_cache = True
    verbose = False

class SlowModule(BaseModule):
    temperature = 0.2
    max_tokens = 3000
    enable_cache = True
    verbose = True

# Clean instantiation
fast = FastModule()
slow = SlowModule()
```
