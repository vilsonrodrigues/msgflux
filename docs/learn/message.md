# Message

The `Message` class is a structured data container designed to facilitate information flow in computational graphs created with `nn` modules. Inspired by `torch.Tensor` and built on top of `dotdict`, it provides an organized way to pass data between different components of a system.

## Overview

`Message` implements a permission-based system where each Module can have specific read and write access to predefined fields. This creates a clean separation of concerns and makes data flow explicit and traceable.

### Key Features

- **Structured Fields**: Predefined fields for different data types (text, audio, images, videos)
- **Flexible Access**: Set and get data using dot notation or string paths
- **Automatic Metadata**: Auto-generated execution IDs and user tracking
- **Module Integration**: Built-in support for `nn.Module` components
- **Type Safety**: Organized structure for multimodal data

## Quick Start

### Basic Usage

```python
import msgflux as mf

# Create empty message
msg = mf.Message()

# Create with content
msg = mf.Message(content="Hello, world!")

# Create with metadata
msg = mf.Message(
    content="Hello!",
    user_id="user_123",
    user_name="Alice",
    chat_id="chat_456"
)

print(msg)
```

### Setting Data

```python
import msgflux as mf

msg = mf.Message()

# Using dot notation
msg.content = "Hello, world!"
msg.texts.input = "User question"
msg.texts.output = "AI response"

# Using set method
msg.set("context.history", ["previous message"])
msg.set("extra.custom_field", {"key": "value"})

print(msg.content)  # "Hello, world!"
print(msg.texts.input)  # "User question"
```

### Getting Data

```python
import msgflux as mf

msg = mf.Message(content="Hello!")
msg.texts.input = "Question"

# Using dot notation
content = msg.content
text_input = msg.texts.input

# Using get method
content = msg.get("content")
text_input = msg.get("texts.input")

print(content)  # "Hello!"
print(text_input)  # "Question"
```

## Message Structure

### Default Fields

Every `Message` instance has these predefined fields:

```python
import msgflux as mf

msg = mf.Message()

# Available fields
msg.metadata    # Execution ID, user info
msg.content     # Main content (text, dict, etc.)
msg.texts       # Text-based data
msg.context     # Contextual information
msg.audios      # Audio data
msg.images      # Image data
msg.videos      # Video data
msg.extra       # Custom/extra data
msg.outputs     # Module outputs
msg.response    # Final response data
```

### Metadata

Metadata is automatically populated:

```python
import msgflux as mf

msg = mf.Message(
    user_id="user_123",
    user_name="Alice",
    chat_id="chat_456"
)

print(msg.metadata)
# {
#     'execution_id': 'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
#     'user_id': 'user_123',
#     'user_name': 'Alice',
#     'chat_id': 'chat_456'
# }

# Execution ID is auto-generated
print(msg.metadata.execution_id)
```

## Working with Fields

### Text Fields

Store text-based data:

```python
import msgflux as mf

msg = mf.Message()

# Store different text types
msg.texts.input = "User input"
msg.texts.output = "Model output"
msg.texts.system = "System message"
msg.texts.history = ["msg1", "msg2"]

print(msg.texts.input)  # "User input"
```

### Context Fields

Store contextual information:

```python
import msgflux as mf

msg = mf.Message()

# Store context
msg.context.history = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"}
]
msg.context.settings = {"temperature": 0.7}
msg.context.metadata = {"source": "web"}

print(msg.context.history)
```

### Media Fields

Store multimedia data:

```python
import msgflux as mf

msg = mf.Message()

# Images
msg.images.input = "path/to/image.jpg"
msg.images.processed = "path/to/processed.jpg"

# Audios
msg.audios.recording = "path/to/audio.mp3"
msg.audios.transcription = "Transcribed text"

# Videos
msg.videos.input = "path/to/video.mp4"
msg.videos.frames = ["frame1.jpg", "frame2.jpg"]

print(msg.images.input)
```

### Extra Fields

Store custom data:

```python
import msgflux as mf

msg = mf.Message()

# Store any custom data
msg.extra.custom_field = "custom value"
msg.extra.metrics = {"accuracy": 0.95}
msg.extra.flags = {"is_verified": True}

print(msg.extra.custom_field)
```

## Nested Access

Access nested data using dot notation or paths:

```python
import msgflux as mf

msg = mf.Message()

# Set nested data
msg.set("context.user.preferences.language", "en")
msg.set("extra.settings.model.temperature", 0.7)

# Get nested data
language = msg.get("context.user.preferences.language")
temp = msg.get("extra.settings.model.temperature")

print(language)  # "en"
print(temp)  # 0.7

# Using dot notation
language = msg.context.user.preferences.language
temp = msg.extra.settings.model.temperature
```

## Responses

### Setting Responses

```python
import msgflux as mf

msg = mf.Message()

# Set response
msg.response.agent_output = "AI generated response"
msg.response.confidence = 0.95
msg.response.metadata = {"tokens": 150}

print(msg.response.agent_output)
```

### Getting Response

```python
import msgflux as mf

msg = mf.Message()
msg.response.result = "Final answer"

# Get first response value
response = msg.get_response()
print(response)  # "Final answer"

# Or access directly
response = msg.response.result
print(response)  # "Final answer"
```

## Integration with Modules

`Message` is designed to work seamlessly with `nn.Module`:

```python
import msgflux as mf

class MyAgent(mf.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mf.Model.chat_completion("openai/gpt-4o")

    def forward(self, msg: mf.Message):
        # Read from message
        user_input = msg.texts.input

        # Process
        response = self.model(messages=[
            {"role": "user", "content": user_input}
        ])

        # Write to message
        msg.texts.output = response.consume()
        msg.response.agent = msg.texts.output

        return msg

# Usage
agent = MyAgent()
msg = mf.Message()
msg.texts.input = "Hello, AI!"

# Process message
result = agent(msg)
print(result.texts.output)
```

## Common Patterns

### Conversation Pipeline

```python
import msgflux as mf

class ConversationPipeline(mf.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mf.Model.chat_completion("openai/gpt-4o")

    def forward(self, msg: mf.Message):
        # Get conversation history
        history = msg.context.history or []

        # Add current input
        history.append({
            "role": "user",
            "content": msg.texts.input
        })

        # Generate response
        response = self.model(messages=history)
        ai_response = response.consume()

        # Update message
        msg.texts.output = ai_response
        history.append({
            "role": "assistant",
            "content": ai_response
        })

        # Store updated history
        msg.context.history = history
        msg.response.conversation = ai_response

        return msg

# Usage
pipeline = ConversationPipeline()
msg = mf.Message(
    texts={"input": "What is AI?"},
    context={"history": []}
)

result = pipeline(msg)
print(result.texts.output)
print(result.context.history)
```

### Multi-Stage Processing

```python
import msgflux as mf

class Preprocessor(mf.nn.Module):
    def forward(self, msg: mf.Message):
        # Clean input
        text = msg.texts.input.strip().lower()
        msg.texts.processed = text
        return msg

class Analyzer(mf.nn.Module):
    def forward(self, msg: mf.Message):
        # Analyze processed text
        text = msg.texts.processed
        msg.outputs.analysis = {
            "length": len(text),
            "words": len(text.split())
        }
        return msg

class Responder(mf.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mf.Model.chat_completion("openai/gpt-4o")

    def forward(self, msg: mf.Message):
        # Generate response
        text = msg.texts.processed
        response = self.model(messages=[
            {"role": "user", "content": text}
        ])
        msg.response.final = response.consume()
        return msg

# Chain modules
preprocessor = Preprocessor()
analyzer = Analyzer()
responder = Responder()

msg = mf.Message()
msg.texts.input = "  HELLO WORLD  "

# Process through pipeline
msg = preprocessor(msg)
msg = analyzer(msg)
msg = responder(msg)

print(msg.texts.processed)  # "hello world"
print(msg.outputs.analysis)  # {'length': 11, 'words': 2}
print(msg.response.final)
```

### RAG System with Message

```python
import msgflux as mf

class RAGSystem(mf.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = mf.Model.text_embedder("openai/text-embedding-3-small")
        self.model = mf.Model.chat_completion("openai/gpt-4o")

    def forward(self, msg: mf.Message):
        query = msg.texts.input
        documents = msg.context.documents or []

        # Embed query
        query_emb = self.embedder(query).consume()
        msg.extra.query_embedding = query_emb

        # Simple retrieval (in practice, use vector DB)
        relevant_docs = documents[:3]  # Top 3
        msg.context.retrieved = relevant_docs

        # Build context
        context = "\n\n".join(relevant_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Generate response
        response = self.model(messages=[
            {"role": "user", "content": prompt}
        ])

        msg.texts.output = response.consume()
        msg.response.rag = msg.texts.output

        return msg

# Usage
rag = RAGSystem()
msg = mf.Message(
    texts={"input": "What is machine learning?"},
    context={"documents": [
        "ML is a subset of AI...",
        "It involves training models...",
        "Common techniques include..."
    ]}
)

result = rag(msg)
print(result.texts.output)
print(result.context.retrieved)
```

### Multimodal Processing

```python
import msgflux as mf

class MultimodalAgent(mf.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = mf.Model.chat_completion("openai/gpt-4o")
        self.tts_model = mf.Model.text_to_speech("openai/tts-1")

    def forward(self, msg: mf.Message):
        # Process image + text
        image_path = msg.images.input
        question = msg.texts.input

        # Vision analysis
        response = self.vision_model(messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_path}}
                ]
            }
        ])

        text_response = response.consume()
        msg.texts.output = text_response

        # Generate audio
        audio_response = self.tts_model(text_response)
        msg.audios.output = audio_response.consume()

        msg.response.multimodal = {
            "text": text_response,
            "audio": msg.audios.output
        }

        return msg

# Usage
agent = MultimodalAgent()
msg = mf.Message(
    texts={"input": "What's in this image?"},
    images={"input": "path/to/image.jpg"}
)

result = agent(msg)
print(result.texts.output)
print(result.audios.output)
```

## Best Practices

### 1. Use Appropriate Fields

```python
import msgflux as mf

# Good - Use semantic field names
msg = mf.Message()
msg.texts.user_question = "What is AI?"
msg.texts.ai_answer = "AI is..."
msg.context.conversation_id = "conv_123"

# Less clear - Everything in extra
msg.extra.data1 = "What is AI?"
msg.extra.data2 = "AI is..."
msg.extra.data3 = "conv_123"
```

### 2. Preserve Message History

```python
import msgflux as mf

class Agent(mf.nn.Module):
    def forward(self, msg: mf.Message):
        # Good - Preserve original input
        original_input = msg.texts.input
        msg.texts.original = original_input

        # Process
        processed = original_input.lower()
        msg.texts.processed = processed

        return msg
```

### 3. Use Metadata for Tracking

```python
import msgflux as mf

msg = mf.Message(
    user_id="user_123",
    user_name="Alice",
    chat_id="chat_456"
)

# Later in pipeline
def track_message(msg: mf.Message):
    print(f"Processing message {msg.metadata.execution_id}")
    print(f"User: {msg.metadata.user_name}")
    print(f"Chat: {msg.metadata.chat_id}")
```

### 4. Document Field Usage

```python
import msgflux as mf

class DocumentedAgent(mf.nn.Module):
    """
    Agent that processes text input.

    Message fields used:
    - Input: msg.texts.input (str) - User question
    - Output: msg.texts.output (str) - Agent response
    - Context: msg.context.history (list) - Conversation history
    """

    def forward(self, msg: mf.Message):
        # Implementation
        pass
```

## Advanced Usage

### Cloning Messages

```python
import msgflux as mf

msg = mf.Message(content="Original")
msg.texts.data = "Important data"

# Create a copy (dotdict supports this)
msg_copy = mf.Message(**dict(msg))
msg_copy.texts.data = "Modified data"

print(msg.texts.data)  # "Important data"
print(msg_copy.texts.data)  # "Modified data"
```

### Conditional Processing

```python
import msgflux as mf

class ConditionalAgent(mf.nn.Module):
    def forward(self, msg: mf.Message):
        # Check if field exists
        if msg.get("context.history"):
            # Process with history
            history = msg.context.history
            msg.texts.mode = "conversation"
        else:
            # Process without history
            msg.texts.mode = "standalone"

        return msg
```

### Merging Messages

```python
import msgflux as mf

def merge_messages(msg1: mf.Message, msg2: mf.Message) -> mf.Message:
    """Merge two messages."""
    merged = mf.Message()

    # Combine texts
    merged.texts.input1 = msg1.texts.input
    merged.texts.input2 = msg2.texts.input

    # Combine context
    merged.context.from_msg1 = msg1.context
    merged.context.from_msg2 = msg2.context

    return merged

msg1 = mf.Message(texts={"input": "First"})
msg2 = mf.Message(texts={"input": "Second"})
merged = merge_messages(msg1, msg2)

print(merged.texts.input1)  # "First"
print(merged.texts.input2)  # "Second"
```

## See Also

- [Module](nn/module.md) - Build custom modules with Message
- [Agent](nn/agent.md) - Agent implementation using Message
- [dotdict](dotdict.md) - Underlying data structure
- [Functional](nn/functional.md) - Functional operations with Message
