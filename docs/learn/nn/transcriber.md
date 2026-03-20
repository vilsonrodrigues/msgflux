# nn.Transcriber

The `nn.Transcriber` module wraps speech-to-text models to **transcribe audio** into text or structured data.

All code examples use the recommended import pattern:

```python
import msgflux as mf
import msgflux.nn as nn
```

## Quick Start

### AutoParams Initialization (Recommended)

Promotes reusability and declarative configuration.

```python
import msgflux as mf
import msgflux.nn as nn

class Speech2Text(nn.Transcriber):
    """Transcribes user voice notes."""
    model = mf.Model.speech_to_text("openai/whisper-1")
    response_mode = "content"
    # Map 'audio' input to the 'user_audio' field in the message
    message_fields = {"task_multimodal_inputs": {"audio": "user_audio"}}

# Instantiate
transcriber = Speech2Text()

# Use
# result = transcriber("/path/to/audio.mp3") 
```

### Traditional Initialization

```python
model = mf.Model.speech_to_text("openai/whisper-1")
transcriber = nn.Transcriber(model=model)

# From file path
result = transcriber("/path/to/audio.mp3")
```

---

## Input Types

### File Path

```python
result = transcriber("/path/to/audio.mp3")
```

### URL

```python
result = transcriber("https://example.com/audio.wav")
```

### Bytes

```python
with open("audio.mp3", "rb") as f:
    audio_bytes = f.read()

result = transcriber(audio_bytes)
```

### Message Object

Use `message_fields` to automatically extract audio from a structured message.

```python
from msgflux import Message

msg = Message()
msg.user_audio = "/path/to/audio.mp3"

# Configure to read from message
transcriber = nn.Transcriber(
    model=model,
    message_fields={"task_multimodal_inputs": {"audio": "user_audio"}},
    response_mode="transcription"
)

transcriber(msg)
print(msg.transcription)  # Output available here
```

---

## Configuration

### Parameters

| Parameter | Description |
|-----------|-------------|
| `model` | Speech-to-text model client instance |
| `message_fields` | Map inputs (audio) from Message fields |
| `response_mode` | Where to write the output in the Message |
| `response_template` | Jinja template to format the output string |
| `response_format` | "text" (default), "json", "verbose_json", "srt", "vtt" |
| `prompt` | Optional text prompt to guide style or vocabulary |
| `config` | Driver-specific config (e.g., temperature, language) |

### Response Formats

You can request structured output like JSON or subtitles.

```python
class SubtitleGenerator(nn.Transcriber):
    model = mf.Model.speech_to_text("openai/whisper-1")
    response_format = "srt"

gen = SubtitleGenerator()
srt_content = gen("video_audio.mp3")
```

---

## With Workflow & Agents

Transcribers are often the first step in a voice processing pipeline.

```python
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F

class Speech2Text(nn.Transcriber):
    model = mf.Model.speech_to_text("openai/whisper-1")
    message_fields = {"task_multimodal_inputs": {"audio": "user_audio"}}
    response_mode = "content"

class Analyzer(nn.Agent):
    """Analyzes the transcribed text."""
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    message_fields = {"task_inputs": "content"}
    response_mode = "analysis"

class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = Speech2Text()
        self.analyzer = Analyzer()
        self.components = nn.ModuleDict({
            "transcriber": self.transcriber,
            "analyzer": self.analyzer
        })
        # Execute transcriber only if audio is present, then analyzer
        self.register_buffer("flux", "{user_audio is not None? transcriber} -> analyzer")

    def forward(self, msg):
        return F.inline(self.flux, self.components, msg)

pipeline = Pipeline()

msg = mf.Message()
msg.user_audio = "/path/to/voice_note.mp3"

pipeline(msg)
print(f"Transcript: {msg.content}")
print(f"Analysis: {msg.analysis}")
```

---

## Async Support

All transcribers support `aforward` for non-blocking I/O.

```python
result = await transcriber.acall("/path/to/audio.mp3")
```

---

## Debugging

Inspect exactly what is sent to the model driver.

```python
# Enable verbose logging
transcriber = nn.Transcriber(model=model, config={"verbose": True})

# Inspect parameters without running
params = transcriber.inspect_model_execution_params("/path/to/audio.mp3")
print(params)
```
