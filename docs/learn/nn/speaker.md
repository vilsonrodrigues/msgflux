# nn.Speaker

The `nn.Speaker` module converts text into natural-sounding speech using text-to-speech models.

All code examples use the recommended import pattern:

```python
import msgflux as mf
import msgflux.nn as nn
```

## Quick Start

### AutoParams Initialization (Recommended)

Define reusable voice personas.

```python
import msgflux as mf
import msgflux.nn as nn

class NaturalVoiceSpeaker(nn.Speaker):
    """Natural-sounding speaker for user-facing applications."""
    response_mode = "plain_response"
    response_format = "mp3"

# Create TTS model
tts_model = mf.Model.text_to_speech("openai/tts-1")

# Create speaker
speaker = NaturalVoiceSpeaker(
    model=tts_model,
    config={"voice": "alloy", "speed": 1.0}
)

# Generate speech
audio = speaker("Hello, welcome to msgFlux!")

# Save audio
with open("welcome.mp3", "wb") as f:
    f.write(audio)
```

### Traditional Initialization

```python
model = mf.Model.text_to_speech("openai/tts-1")

speaker = nn.Speaker(
    model=model,
    response_format="mp3",
    config={"voice": "alloy"}
)

audio = speaker("Hello world")
```

---

## Audio Formats

Choose the right format for your use case.

| Format | Description | Use Case |
|--------|-------------|----------|
| `"mp3"` | Universal, compressed | Podcasts, UI sounds |
| `"opus"` | Low latency, high efficiency | Streaming, RTC |
| `"flac"` | Lossless compressed | Archival, high-end audio |
| `"wav"` | Uncompressed | Editing, post-processing |
| `"aac"` | Standard compressed | Mobile apps |

```python
class StreamingSpeaker(nn.Speaker):
    """Low-latency speaker for chunks."""
    response_format = "opus"

streamer = StreamingSpeaker(model=tts_model)
```

---

## Configuration

### Voice & Speed

Control characteristics via `config`.

```python
class NarratorSpeaker(nn.Speaker):
    """Clear, neutral voice for audiobooks."""
    response_format = "mp3"

narrator = NarratorSpeaker(
    model=tts_model,
    config={
        "voice": "echo",  # Provider-specific voice ID
        "speed": 0.9      # 1.0 is normal speed
    }
)
```

### Guardrails

Validate or sanitize input text before generation to save costs and ensure safety.

```python
def sanitize_input(text: str) -> str:
    """Remove sensitive info or restrict length."""
    if len(text) > 4096:
        raise ValueError("Text too long")
    return text

speaker = nn.Speaker(
    model=tts_model,
    guardrails={
        "input": sanitize_input
    }
)

# Will raise ValueError if too long
audio = speaker(long_text)
```

### Prompt Guidance

Some models accept a system prompt or style guidance.

```python
class StorytellerSpeaker(nn.Speaker):
    """Expressive speaker."""
    prompt = "Speak with dramatic pauses and emotional variation."

storyteller = StorytellerSpeaker(model=tts_model)
```

---

## Streaming

For real-time applications, consume the audio stream generator.

```python
class StreamingSpeaker(nn.Speaker):
    response_format = "opus"

speaker = StreamingSpeaker(
    model=tts_model,
    config={"stream": True}
)

# Get async generator
stream = speaker("This audio will be streamed chunk by chunk.")

async for chunk in stream:
    # Send to client immediately
    await websocket.send(chunk)
```

---

## Integration with Agents

Speakers typically sit at the end of a voice pipeline (Agent -> Speaker).

```python
class VoiceAssistant(nn.Agent):
    """Voice-enabled assistant."""
    model = mf.Model.chat_completion("openai/gpt-4")

class ResponseSpeaker(nn.Speaker):
    """Converts agent responses to speech."""
    model = mf.Model.text_to_speech("openai/tts-1")
    response_format = "mp3"

# simple pipeline
assistant = VoiceAssistant()
speaker = ResponseSpeaker()

text = assistant("What's the weather?")
audio = speaker(text)
```

---

## Message Field Mapping

Automatically extract text from structured messages.

```python
class NotificationSpeaker(nn.Speaker):
    """Reads notifications."""
    response_mode = "message"
    message_fields = {"task_inputs": "notification.text"}

speaker = NotificationSpeaker(model=tts_model)

msg = mf.Message()
msg.set("notification.text", "You have a new meeting.")

result_msg = speaker(msg)
audio = result_msg.get("speaker.audio")
```

---

## Creating Speaker Hierarchies

Share configuration across related speakers.

```python
# Base speaker for announcements
class AnnouncementSpeaker(nn.Speaker):
    response_format = "mp3"
    config = {"voice": "onyx"}

# Urgent announcements
class EmergencySpeaker(AnnouncementSpeaker):
    config = {"voice": "onyx", "speed": 1.1}

# Casual announcements
class CasualSpeaker(AnnouncementSpeaker):
    config = {"voice": "nova", "speed": 1.0}
```
