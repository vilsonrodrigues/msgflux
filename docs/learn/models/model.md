# Model

The `Model` class provides a unified interface for working with AI models across different providers and modalities. It acts as a factory that creates provider-specific model instances with a consistent API.

**All code examples use the recommended import pattern:**

```python
import msgflux as mf
```

## Overview

Instead of learning different APIs for each provider, you use a single factory method:

```python
# OpenAI
model = mf.Model.chat_completion("openai/gpt-4o", temperature=0.7)

# Google
model = mf.Model.chat_completion("google/gemini-2.0-flash-exp", temperature=0.7)

# Anthropic
model = mf.Model.chat_completion("anthropic/claude-3-5-sonnet-20241022", temperature=0.7)
```

All models follow the pattern: `provider/model-id`

## Quick Start

### Installation

```bash
# Install with specific provider support
pip install msgflux[openai]
pip install msgflux[google]
pip install msgflux[anthropic]
```

### Basic Usage

```python
import msgflux as mf

# Set API key
mf.set_envs(OPENAI_API_KEY="sk-...")

# Create model
model = mf.Model.chat_completion("openai/gpt-4o")

# Use model (see chat_completion.md for details)
response = model(messages=[{"role": "user", "content": "Hello!"}])
print(response.consume())  # "Hello! How can I help you today?"
```

## Model Types

The `Model` class supports multiple model types:

### Available Model Types

```python
# Get all supported model types
types = mf.Model.model_types()
print(types)
# [
#     'chat_completion',
#     'text_embedder',
#     'text_to_image',
#     'image_text_to_image',
#     'text_to_speech',
#     'speech_to_text',
#     'moderation',
#     'text_classifier',
#     'image_classifier',
#     'image_embedder',
#     'text_reranker'
# ]
```

### Factory Methods

Each model type has a dedicated factory method:

| Model Type | Factory Method | Use Case |
|------------|---------------|----------|
| **chat_completion** | `Model.chat_completion()` | Chat and text generation |
| **text_embedder** | `Model.text_embedder()` | Convert text to vectors |
| **text_to_image** | `Model.text_to_image()` | Generate images from text |
| **image_text_to_image** | `Model.image_text_to_image()` | Edit images with text |
| **text_to_speech** | `Model.text_to_speech()` | Convert text to audio |
| **speech_to_text** | `Model.speech_to_text()` | Transcribe audio to text |
| **moderation** | `Model.moderation()` | Content moderation |
| **text_classifier** | `Model.text_classifier()` | Classify text |
| **image_classifier** | `Model.image_classifier()` | Classify images |
| **image_embedder** | `Model.image_embedder()` | Convert images to vectors |
| **text_reranker** | `Model.text_reranker()` | Rerank text results |

## Providers

### Available Providers

```python
# Get all providers by model type
providers = mf.Model.providers()
print(providers)
# {
#     'chat_completion': ['openai', 'google', 'anthropic', 'groq', 'together', ...],
#     'text_embedder': ['openai', 'google', 'jinaai', ...],
#     'text_to_image': ['openai', 'replicate', ...],
#     ...
# }
```

### Supported Providers

- **openai** - OpenAI models (GPT-4, DALL-E, Whisper, etc.)
- **google** - Google models (Gemini)
- **anthropic** - Anthropic models (Claude)
- **groq** - Groq models (fast inference)
- **together** - Together AI models
- **replicate** - Replicate models
- **ollama** - Local Ollama models
- **cerebras** - Cerebras models
- **sambanova** - SambaNova models
- **jinaai** - Jina AI embeddings
- **openrouter** - OpenRouter gateway
- **imagerouter** - Image router gateway
- **vllm** - vLLM local deployment

## Usage Examples

### Chat Completion

```python
import msgflux as mf

# Create model
model = mf.Model.chat_completion(
    "openai/gpt-4o",
    temperature=0.7,
    max_tokens=1000
)

# Single completion
response = model(messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
])

print(response.consume())  # "The capital of France is Paris."
```

### Text Embeddings

```python
import msgflux as mf

# Create embedder
embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

# Generate embedding
response = embedder("Hello, world!")

embedding = response.consume()
print(len(embedding))  # 1536
print(embedding[:5])  # [0.123, -0.456, 0.789, ...]
```

### Speech to Text

```python
import msgflux as mf

# Create transcription model
model = mf.Model.speech_to_text("openai/whisper-1")

# Transcribe audio file
response = model("path/to/audio.mp3")

transcription = response.consume()
print(transcription)  # "Hello, this is a test."
```

### Text to Image

```python
import msgflux as mf

# Create image generation model
model = mf.Model.text_to_image("openai/dall-e-3")

# Generate image
response = model(
    prompt="A cat wearing a space suit",
    size="1024x1024",
    quality="standard"
)

image_url = response.consume()
print(image_url)  # "https://..."
```

## Model Information

### Getting Model Metadata

```python
import msgflux as mf

model = mf.Model.chat_completion("openai/gpt-4o")

# Get model type
print(model.model_type)  # "chat_completion"

# Get instance type information
print(model.instance_type())
# {'model_type': 'chat_completion'}

# Get model info
print(model.get_model_info())
# {'model_id': 'gpt-4o', 'provider': 'openai'}
```

## Serialization

Models can be serialized and deserialized for storage or transfer:

### Serializing a Model

```python
import msgflux as mf

# Create and configure model
model = mf.Model.chat_completion(
    "openai/gpt-4o",
    temperature=0.7,
    max_tokens=500
)

# Serialize
state = model.serialize()
print(state)
# {
#     'msgflux_type': 'model',
#     'provider': 'openai',
#     'model_type': 'chat_completion',
#     'state': {
#         'model_id': 'gpt-4o',
#         'sampling_params': {...},
#         'sampling_run_params': {
#             'temperature': 0.7,
#             'max_tokens': 500,
#             ...
#         }
#     }
# }

# Save to file
mf.save(state, "model_config.json")
```

### Deserializing a Model

```python
import msgflux as mf

# Load from file
state = mf.load("model_config.json")

# Recreate model
model = mf.Model.from_serialized(
    provider=state['provider'],
    model_type=state['model_type'],
    state=state['state']
)

# Model is ready to use
response = model(messages=[{"role": "user", "content": "Hello"}])
```

## Response Types

All models return one of two response types:

### ModelResponse

For non-streaming responses (embeddings, transcription, etc.):

```python
import msgflux as mf

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")
response = embedder("Hello")

# Response is ModelResponse
print(type(response))  # <class 'msgflux.models.response.ModelResponse'>

# Get response type
print(response.response_type)  # "text_embedding"

# Consume the response
result = response.consume()
print(result)  # [0.123, -0.456, ...]
```

### ModelStreamResponse

For streaming responses (chat, text-to-speech, etc.):

```python
import msgflux as mf

model = mf.Model.chat_completion("openai/gpt-4o")

# Stream enabled
response = model(
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True
)

# Response is ModelStreamResponse
print(type(response))  # <class 'msgflux.models.response.ModelStreamResponse'>

# Consume stream
async for chunk in response.consume():
    print(chunk, end="", flush=True)
# Output: "1, 2, 3, 4, 5"
```

## Error Handling

### Retry Mechanism

All API-based models have automatic retry logic for transient failures:

```python
import msgflux as mf

# Model automatically retries on API failures
model = mf.Model.chat_completion("openai/gpt-4o")

try:
    response = model(messages=[{"role": "user", "content": "Hello"}])
except Exception as e:
    print(f"Failed after retries: {e}")
```

### Provider Not Available

```python
import msgflux as mf

try:
    # If provider isn't installed
    model = mf.Model.chat_completion("openai/gpt-4o")
except ImportError as e:
    print(e)
    # "`openai` client is not available. Install with `pip install msgflux[openai]`."
```

### Invalid Model Path

```python
import msgflux as mf

try:
    model = mf.Model.chat_completion("invalid-provider/model")
except ValueError as e:
    print(e)
    # "Provider `invalid-provider` not registered for type `chat_completion`"
```

## Best Practices

### 1. Use Environment Variables for API Keys

```python
# Good - Use environment variables
mf.set_envs(OPENAI_API_KEY="sk-...")

# Or use .env file
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=...
```

### 2. Reuse Model Instances

```python
# Good - Create once, use many times
model = mf.Model.chat_completion("openai/gpt-4o")

for query in queries:
    response = model(messages=[{"role": "user", "content": query}])
    print(response.consume())

# Bad - Creating new instance each time (slower)
for query in queries:
    model = mf.Model.chat_completion("openai/gpt-4o")
    response = model(messages=[{"role": "user", "content": query}])
```

### 3. Specify Model Parameters

```python
# Good - Explicit parameters
model = mf.Model.chat_completion(
    "openai/gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9
)

# Also good - Use defaults when appropriate
model = mf.Model.chat_completion("openai/gpt-4o")
```

### 4. Handle Errors Gracefully

```python
import msgflux as mf

def safe_completion(prompt):
    try:
        model = mf.Model.chat_completion("openai/gpt-4o")
        response = model(messages=[{"role": "user", "content": prompt}])
        return response.consume()
    except ImportError:
        return "Provider not installed"
    except ValueError as e:
        return f"Configuration error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

result = safe_completion("Hello!")
```

### 5. Save Configurations

```python
import msgflux as mf

# Define configurations
configs = {
    "creative": {
        "temperature": 0.9,
        "max_tokens": 2000
    },
    "precise": {
        "temperature": 0.3,
        "max_tokens": 500
    }
}

# Create and save models
for name, params in configs.items():
    model = mf.Model.chat_completion("openai/gpt-4o", **params)
    state = model.serialize()
    mf.save(state, f"{name}_model.json")

# Load later
state = mf.load("creative_model.json")
model = mf.Model.from_serialized(**state)
```

## Common Patterns

### Multi-Provider Fallback

```python
import msgflux as mf

def get_completion(prompt, providers=None):
    """Try multiple providers in order."""
    if providers is None:
        providers = ["openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022", "google/gemini-2.0-flash-exp"]

    for provider_path in providers:
        try:
            model = mf.Model.chat_completion(provider_path)
            response = model(messages=[{"role": "user", "content": prompt}])
            return response.consume()
        except Exception as e:
            print(f"Failed with {provider_path}: {e}")
            continue

    raise RuntimeError("All providers failed")

result = get_completion("What is AI?")
```

### Batch Processing

```python
import msgflux as mf
import msgflux.nn.functional as F

# Create embedder
embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

# Process in parallel
texts = ["Hello", "World", "AI", "Embedding"]

results = F.map_gather(
    embedder,
    args_list=[(text,) for text in texts]
)

# Each result is a ModelResponse
embeddings = [r.consume() for r in results]
print(f"Generated {len(embeddings)} embeddings")
```

### Model Comparison

```python
import msgflux as mf

def compare_models(prompt, model_paths):
    """Compare responses from different models."""
    results = {}

    for path in model_paths:
        model = mf.Model.chat_completion(path, temperature=0.7)
        response = model(messages=[{"role": "user", "content": prompt}])
        results[path] = response.consume()

    return results

# Compare
models = [
    "openai/gpt-4o",
    "anthropic/claude-3-5-sonnet-20241022",
    "google/gemini-2.0-flash-exp"
]

responses = compare_models("Explain quantum computing", models)
for model, response in responses.items():
    print(f"\n{model}:\n{response}\n")
```

## See Also

- [Chat Completion](chat_completion.md) - Detailed chat completion usage
- [Embeddings](embeddings.md) - Text embedding models
- [Image Generation](image_generation.md) - Text-to-image models
- [Speech](speech.md) - Speech-to-text and text-to-speech
- [Moderation](moderation.md) - Content moderation
