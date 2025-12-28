# Speaker: Text-to-Speech Module

The `Speaker` module converts text into natural-sounding speech using text-to-speech models.

**All code examples use the recommended import pattern:**

```python
import msgflux as mf
```

## Quick Start

### Traditional Initialization

```python
import msgflux as mf

# Create TTS model
tts_model = mf.Model.text_to_speech("openai/tts-1")

# Traditional speaker initialization
speaker = mf.nn.Speaker(
    model=tts_model,
    response_format="mp3",
    config={"voice": "alloy"}
)

# Generate speech
audio = speaker("Hello, welcome to msgFlux!")

# Save audio file
with open("output.mp3", "wb") as f:
    f.write(audio)
```

### AutoParams Initialization (Recommended)

**This is the preferred and recommended way to define speakers in msgflux.**

```python
import msgflux as mf

class NaturalVoiceSpeaker(mf.nn.Speaker):
    """Natural-sounding speaker for user-facing applications."""

    # AutoParams automatically uses class name as 'name'
    response_format = "mp3"
    response_mode = "plain_response"

# Create TTS model
tts_model = mf.Model.text_to_speech("openai/tts-1")

# Create speaker with defaults
speaker = NaturalVoiceSpeaker(
    model=tts_model,
    config={"voice": "alloy", "speed": 1.0}
)

# Generate speech
audio = speaker("Hello, welcome to msgFlux!")

# Save audio
with open("welcome.mp3", "wb") as f:
    f.write(audio)

# Create variant with different voice
nova_speaker = NaturalVoiceSpeaker(
    model=tts_model,
    config={"voice": "nova", "speed": 1.1}  # Slightly faster
)
```

## Why Use AutoParams?

1. **Voice Personas**: Create distinct speaker personas with consistent configuration
2. **Reusable Voices**: Define voices once, use across application
3. **Clear Purpose**: Class name and docstring document speaker characteristics
4. **Easy Variants**: Create voice variations through inheritance
5. **Better Organization**: Group speakers by use case or audience

## Audio Formats

Speakers support multiple output formats:

```python
import msgflux as mf

class PodcastSpeaker(mf.nn.Speaker):
    """High-quality speaker for podcast production."""

    response_format = "mp3"  # Best for distribution
    response_mode = "plain_response"

class LiveStreamSpeaker(mf.nn.Speaker):
    """Low-latency speaker for live streaming."""

    response_format = "opus"  # Low latency, good compression
    response_mode = "plain_response"

class ArchivalSpeaker(mf.nn.Speaker):
    """Lossless speaker for archival purposes."""

    response_format = "flac"  # Lossless compression
    response_mode = "plain_response"

# Create instances
tts = mf.Model.text_to_speech("openai/tts-1-hd")

podcast = PodcastSpeaker(model=tts, config={"voice": "onyx"})
livestream = LiveStreamSpeaker(model=tts, config={"voice": "alloy"})
archival = ArchivalSpeaker(model=tts, config={"voice": "echo"})

text = "This is a test of the text-to-speech system."

# Generate in different formats
podcast_audio = podcast(text)
stream_audio = livestream(text)
archive_audio = archival(text)
```

## Voice Selection

Different voices for different contexts:

```python
import msgflux as mf

class ProfessionalSpeaker(mf.nn.Speaker):
    """Professional, authoritative voice for business content."""

    response_format = "mp3"
    response_mode = "plain_response"

class FriendlySpeaker(mf.nn.Speaker):
    """Warm, friendly voice for customer interactions."""

    response_format = "mp3"
    response_mode = "plain_response"

class NarratorSpeaker(mf.nn.Speaker):
    """Clear, neutral voice for audiobook narration."""

    response_format = "mp3"
    response_mode = "plain_response"

tts = mf.Model.text_to_speech("openai/tts-1")

# Create speakers with appropriate voices
professional = ProfessionalSpeaker(model=tts, config={"voice": "onyx"})
friendly = FriendlySpeaker(model=tts, config={"voice": "nova"})
narrator = NarratorSpeaker(model=tts, config={"voice": "echo"})

# Use contextually
business_intro = professional("Welcome to our quarterly earnings call.")
customer_greeting = friendly("Hi! How can I help you today?")
audiobook_chapter = narrator("Chapter 1: The Beginning.")
```

## Speed Control

Control speech rate for different use cases:

```python
import msgflux as mf

class SlowSpeaker(mf.nn.Speaker):
    """Slow, clear speech for learning content."""

    response_format = "mp3"
    response_mode = "plain_response"

class NormalSpeaker(mf.nn.Speaker):
    """Normal speed for general content."""

    response_format = "mp3"
    response_mode = "plain_response"

class FastSpeaker(mf.nn.Speaker):
    """Fast speech for efficient content delivery."""

    response_format = "mp3"
    response_mode = "plain_response"

tts = mf.Model.text_to_speech("openai/tts-1")

# Create with different speeds
slow = SlowSpeaker(model=tts, config={"voice": "alloy", "speed": 0.75})
normal = NormalSpeaker(model=tts, config={"voice": "alloy", "speed": 1.0})
fast = FastSpeaker(model=tts, config={"voice": "alloy", "speed": 1.5})

text = "This content is optimized for different listening speeds."

# Generate at different rates
slow_audio = slow(text)
normal_audio = normal(text)
fast_audio = fast(text)
```

## Streaming Audio

For real-time applications, stream audio as it's generated:

```python
import msgflux as mf

class StreamingSpeaker(mf.nn.Speaker):
    """Streaming speaker for real-time applications."""

    response_format = "opus"  # Best for streaming
    response_mode = "plain_response"

tts = mf.Model.text_to_speech("openai/tts-1")

speaker = StreamingSpeaker(
    model=tts,
    config={"voice": "nova", "stream": True}
)

# Get streaming response
stream = speaker("This audio will be streamed chunk by chunk.")

# Process chunks as they arrive
async for chunk in stream:
    # Send chunk to client, play immediately, etc.
    process_audio_chunk(chunk)
```

## Guardrails

Add input validation and filtering:

```python
import msgflux as mf

def sanitize_input(text: str) -> str:
    """Remove sensitive information before TTS."""
    # Remove credit card numbers, SSNs, etc.
    sanitized = remove_pii(text)
    return sanitized

def check_length(text: str) -> str:
    """Ensure text is within reasonable length."""
    if len(text) > 4096:
        raise ValueError("Text too long for TTS")
    return text

class SafeSpeaker(mf.nn.Speaker):
    """Speaker with input guardrails for safety."""

    response_format = "mp3"
    response_mode = "plain_response"

tts = mf.Model.text_to_speech("openai/tts-1")

safe_speaker = SafeSpeaker(
    model=tts,
    guardrails={
        "input": lambda text: check_length(sanitize_input(text))
    },
    config={"voice": "alloy"}
)

# Input is automatically sanitized and validated
audio = safe_speaker("User input that may contain sensitive data...")
```

## Message Field Mapping

Use Message objects for structured processing:

```python
import msgflux as mf

class NotificationSpeaker(mf.nn.Speaker):
    """Speaker that processes structured notification messages."""

    response_format = "mp3"
    response_mode = "message"

tts = mf.Model.text_to_speech("openai/tts-1")

speaker = NotificationSpeaker(
    model=tts,
    message_fields={
        "task_inputs": "notification.text"
    },
    config={"voice": "nova"}
)

# Create notification message
msg = mf.Message()
msg.set("notification.text", "You have 3 new messages")
msg.set("notification.priority", "high")

# Process and get audio in message
result_msg = speaker(msg)
audio = result_msg.get("speaker.audio")

# Save
with open("notification.mp3", "wb") as f:
    f.write(audio)
```

## Creating Speaker Hierarchies

Build specialized speakers through inheritance:

```python
import msgflux as mf

# Base speaker for all announcements
class AnnouncementSpeaker(mf.nn.Speaker):
    """Base speaker for public announcements."""

    response_format = "mp3"
    response_mode = "plain_response"

# Emergency announcements - clear and authoritative
class EmergencySpeaker(AnnouncementSpeaker):
    """Urgent speaker for emergency announcements."""

    # Inherits format and mode from AnnouncementSpeaker

# Casual announcements - friendly tone
class CasualSpeaker(AnnouncementSpeaker):
    """Friendly speaker for casual announcements."""

    # Different voice but same format

tts = mf.Model.text_to_speech("openai/tts-1")

# Create instances
emergency = EmergencySpeaker(
    model=tts,
    config={"voice": "onyx", "speed": 0.9}  # Slower for clarity
)

casual = CasualSpeaker(
    model=tts,
    config={"voice": "nova", "speed": 1.1}  # Slightly faster
)

# Use appropriately
alert = emergency("Attention: Building evacuation in progress.")
update = casual("Reminder: Team meeting at 3 PM.")
```

## Integration with Agents

Speakers can be used in agent workflows:

```python
import msgflux as mf

class ResponseSpeaker(mf.nn.Speaker):
    """Converts agent responses to speech."""

    response_format = "mp3"
    response_mode = "plain_response"

# Create speaker
tts = mf.Model.text_to_speech("openai/tts-1")
speaker = ResponseSpeaker(model=tts, config={"voice": "alloy"})

# Create agent
class VoiceAssistant(mf.nn.Agent):
    """Voice-enabled assistant."""

    temperature = 0.7
    max_tokens = 150

model = mf.Model.chat_completion("openai/gpt-4")
assistant = VoiceAssistant(model=model)

# Get text response from agent
text_response = assistant("What's the weather like today?")

# Convert to speech
audio_response = speaker(text_response)

# Play or save audio
with open("response.mp3", "wb") as f:
    f.write(audio_response)
```

## Prompt Guidance

Guide speech generation patterns:

```python
import msgflux as mf

class StorytellerSpeaker(mf.nn.Speaker):
    """Expressive speaker for storytelling."""

    response_format = "mp3"
    response_mode = "plain_response"

tts = mf.Model.text_to_speech("openai/tts-1-hd")

storyteller = StorytellerSpeaker(
    model=tts,
    prompt="Speak with dramatic pauses and emotional variation, like narrating an audiobook.",
    config={"voice": "echo"}
)

story = """
Once upon a time, in a land far away, there lived a brave knight.
His adventures would become the stuff of legends.
"""

audio = storyteller(story)
```

## Configuration Options

### Complete Parameter Reference

```python
import msgflux as mf

class FullyConfiguredSpeaker(mf.nn.Speaker):
    """Speaker with all configuration options."""

    # Response behavior
    response_format = "mp3"  # "mp3", "opus", "aac", "flac", "wav", "pcm"
    response_mode = "plain_response"  # or "message"

tts = mf.Model.text_to_speech("openai/tts-1")

speaker = FullyConfiguredSpeaker(
    model=tts,                           # Required: TTS model
    guardrails={                         # Optional: input validation
        "input": sanitize_function
    },
    message_fields={                     # Optional: Message field mapping
        "task_inputs": "text.content"
    },
    prompt="Speaker guidance",           # Optional: generation guidance
    config={                             # Optional: TTS-specific config
        "voice": "alloy",                # Voice selection
        "speed": 1.0,                    # Speech rate (0.25 - 4.0)
        "stream": False                  # Enable streaming
    },
    name="custom_speaker"                # Optional: custom name
)
```

## Best Practices

### 1. Match Voice to Purpose

```python
# Good - Voice matches content type
class BusinessSpeaker(mf.nn.Speaker):
    """Professional voice for business content."""
    response_format = "mp3"

business = BusinessSpeaker(
    model=tts,
    config={"voice": "onyx"}  # Deep, authoritative
)

class ChildrensSpeaker(mf.nn.Speaker):
    """Friendly voice for children's content."""
    response_format = "mp3"

children = ChildrensSpeaker(
    model=tts,
    config={"voice": "nova"}  # Warm, friendly
)
```

### 2. Choose Appropriate Format

```python
# High quality for downloads
class DownloadSpeaker(mf.nn.Speaker):
    """High-quality speaker for downloadable content."""
    response_format = "mp3"  # Good quality, widely supported

# Low latency for real-time
class RealtimeSpeaker(mf.nn.Speaker):
    """Low-latency speaker for real-time applications."""
    response_format = "opus"  # Fast, efficient
```

### 3. Use Guardrails for User Input

```python
class UserInputSpeaker(mf.nn.Speaker):
    """Safe speaker for user-generated content."""
    response_format = "mp3"

speaker = UserInputSpeaker(
    model=tts,
    guardrails={
        "input": lambda text: sanitize_and_validate(text)
    }
)
```

## Migration Guide

### From Traditional to AutoParams

**Before (Traditional):**
```python
speaker = mf.nn.Speaker(
    model=tts,
    response_format="mp3",
    config={"voice": "alloy", "speed": 1.0}
)
```

**After (AutoParams - Recommended):**
```python
class MySpeaker(mf.nn.Speaker):
    """Natural speaker for general use."""
    response_format = "mp3"

speaker = MySpeaker(
    model=tts,
    config={"voice": "alloy", "speed": 1.0}
)
```

## Summary

- **Use AutoParams** for defining speakers - create voice personas with consistent configuration
- **Traditional initialization** works for quick, one-off TTS tasks
- Supports **multiple audio formats** (MP3, Opus, AAC, FLAC, WAV, PCM)
- Configure **voice**, **speed**, and **streaming** options
- Add **guardrails** for input validation
- **Streaming support** for real-time applications

The Speaker module provides flexible text-to-speech - use AutoParams to create distinct voice personas for different contexts.
