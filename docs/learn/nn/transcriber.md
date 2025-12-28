# Transcriber: Speech-to-Text Module

The `Transcriber` module converts speech audio into text using speech-to-text models.

**All code examples use the recommended import pattern:**

```python
import msgflux as mf
```

## Quick Start

### Traditional Initialization

```python
import msgflux as mf

# Create STT model
stt_model = mf.Model.speech_to_text("openai/whisper-1")

# Traditional transcriber initialization
transcriber = mf.nn.Transcriber(
    model=stt_model,
    response_format="text",
    config={"language": "en"}
)

# Transcribe audio file
text = transcriber("audio.mp3")
print(text)
```

### AutoParams Initialization (Recommended)

**This is the preferred and recommended way to define transcribers in msgflux.**

```python
import msgflux as mf

class PodcastTranscriber(mf.nn.Transcriber):
    """Transcriber optimized for podcast audio with high accuracy."""

    # AutoParams automatically uses class name as 'name'
    response_format = "text"
    response_mode = "plain_response"

# Create STT model
stt_model = mf.Model.speech_to_text("openai/whisper-1")

# Create transcriber with defaults
transcriber = PodcastTranscriber(
    model=stt_model,
    config={"language": "en", "temperature": 0.0}
)

# Transcribe
text = transcriber("podcast_episode.mp3")
print(text)

# Create variant for different language
spanish_transcriber = PodcastTranscriber(
    model=stt_model,
    config={"language": "es", "temperature": 0.0}
)
```

## Why Use AutoParams?

1. **Domain-Specific Transcribers**: Create specialized transcribers for different audio types
2. **Language-Specific Configuration**: Define transcribers per language with optimal settings
3. **Reusable Configurations**: Share configuration across transcription tasks
4. **Clear Purpose**: Class name and docstring document transcriber characteristics
5. **Easy Variants**: Create transcriber variations through inheritance

## Response Formats

Transcribers support multiple output formats:

```python
import msgflux as mf

class TextTranscriber(mf.nn.Transcriber):
    """Basic transcriber returning plain text."""

    response_format = "text"
    response_mode = "plain_response"

class JSONTranscriber(mf.nn.Transcriber):
    """Transcriber with JSON output for structured processing."""

    response_format = "json"
    response_mode = "plain_response"

class SubtitleTranscriber(mf.nn.Transcriber):
    """Transcriber generating SRT subtitles with timestamps."""

    response_format = "srt"
    response_mode = "plain_response"

class DetailedTranscriber(mf.nn.Transcriber):
    """Transcriber with verbose JSON including word-level timestamps."""

    response_format = "verbose_json"
    response_mode = "plain_response"

# Create instances
stt = mf.Model.speech_to_text("openai/whisper-1")

text_trans = TextTranscriber(model=stt)
json_trans = JSONTranscriber(model=stt)
subtitle_trans = SubtitleTranscriber(model=stt)
detailed_trans = DetailedTranscriber(
    model=stt,
    config={"timestamp_granularities": "word"}
)

# Transcribe in different formats
text = text_trans("audio.mp3")
json_data = json_trans("audio.mp3")
subtitles = subtitle_trans("audio.mp3")
detailed = detailed_trans("audio.mp3")
```

## Language-Specific Transcribers

Create transcribers optimized for specific languages:

```python
import msgflux as mf

class EnglishTranscriber(mf.nn.Transcriber):
    """High-accuracy transcriber for English audio."""

    response_format = "text"
    response_mode = "plain_response"

class SpanishTranscriber(mf.nn.Transcriber):
    """High-accuracy transcriber for Spanish audio."""

    response_format = "text"
    response_mode = "plain_response"

class MultilingualTranscriber(mf.nn.Transcriber):
    """Transcriber for auto-detecting and transcribing multiple languages."""

    response_format = "text"
    response_mode = "plain_response"

stt = mf.Model.speech_to_text("openai/whisper-1")

# Create language-specific transcribers
english = EnglishTranscriber(model=stt, config={"language": "en"})
spanish = SpanishTranscriber(model=stt, config={"language": "es"})
multilingual = MultilingualTranscriber(model=stt)  # Auto-detect

# Use appropriately
en_text = english("english_audio.mp3")
es_text = spanish("spanish_audio.mp3")
multi_text = multilingual("unknown_language.mp3")
```

## Prompt Guidance

Guide transcription with prompts for better accuracy:

```python
import msgflux as mf

class MedicalTranscriber(mf.nn.Transcriber):
    """Transcriber optimized for medical terminology."""

    response_format = "text"
    response_mode = "plain_response"

class LegalTranscriber(mf.nn.Transcriber):
    """Transcriber optimized for legal terminology."""

    response_format = "text"
    response_mode = "plain_response"

stt = mf.Model.speech_to_text("openai/whisper-1")

# Create with domain-specific prompts
medical = MedicalTranscriber(
    model=stt,
    prompt="Medical consultation with terms like: patient, diagnosis, treatment, prescription, symptoms",
    config={"language": "en"}
)

legal = LegalTranscriber(
    model=stt,
    prompt="Legal deposition with terms like: plaintiff, defendant, evidence, testimony, objection",
    config={"language": "en"}
)

# Transcribe with improved accuracy for domain terms
medical_transcript = medical("doctor_patient_conversation.mp3")
legal_transcript = legal("court_hearing.mp3")
```

## Timestamp Granularities

Get word or segment-level timestamps:

```python
import msgflux as mf

class TimestampedTranscriber(mf.nn.Transcriber):
    """Transcriber with word-level timestamps."""

    response_format = "verbose_json"
    response_mode = "plain_response"

stt = mf.Model.speech_to_text("openai/whisper-1")

transcriber = TimestampedTranscriber(
    model=stt,
    config={
        "language": "en",
        "timestamp_granularities": "word"  # or "segment"
    }
)

# Get detailed transcription with timestamps
result = transcriber("speech.mp3")

# Access word-level timestamps
for word_info in result["words"]:
    print(f"{word_info['word']}: {word_info['start']}s - {word_info['end']}s")
```

## Streaming Transcription

For real-time applications, stream transcription as audio is processed:

```python
import msgflux as mf

class StreamingTranscriber(mf.nn.Transcriber):
    """Real-time streaming transcriber."""

    response_format = "text"
    response_mode = "plain_response"

stt = mf.Model.speech_to_text("openai/whisper-1")

transcriber = StreamingTranscriber(
    model=stt,
    config={"language": "en", "stream": True}
)

# Get streaming response
stream = transcriber("live_audio.mp3")

# Process chunks as they arrive
async for chunk in stream:
    print(chunk, end="", flush=True)
```

## Message Field Mapping

Use Message objects for structured processing:

```python
import msgflux as mf

class CallTranscriber(mf.nn.Transcriber):
    """Transcriber that processes call recording messages."""

    response_format = "text"
    response_mode = "message"

stt = mf.Model.speech_to_text("openai/whisper-1")

transcriber = CallTranscriber(
    model=stt,
    message_fields={
        "task_multimodal_inputs": "call.recording"
    },
    config={"language": "en"}
)

# Create call message
msg = mf.Message()
msg.set("call.recording", "customer_call_123.mp3")
msg.set("call.id", "123")
msg.set("call.timestamp", "2024-01-15T10:30:00")

# Transcribe and get result in message
result_msg = transcriber(msg)
transcript = result_msg.get("transcriber.text")
call_id = result_msg.get("call.id")

print(f"Call {call_id}: {transcript}")
```

## Response Templates

Format transcription results using Jinja templates:

```python
import msgflux as mf

class FormattedTranscriber(mf.nn.Transcriber):
    """Transcriber with custom output formatting."""

    response_format = "json"
    response_mode = "plain_response"

stt = mf.Model.speech_to_text("openai/whisper-1")

transcriber = FormattedTranscriber(
    model=stt,
    response_template="""
    Transcription completed:
    Language: {{ language }}
    Duration: {{ duration }}s
    Text: {{ text }}
    """,
    config={"language": "en"}
)

formatted_result = transcriber("audio.mp3")
print(formatted_result)
```

## Creating Transcriber Hierarchies

Build specialized transcribers through inheritance:

```python
import msgflux as mf

# Base transcriber for all meetings
class MeetingTranscriber(mf.nn.Transcriber):
    """Base transcriber for meeting recordings."""

    response_format = "text"
    response_mode = "plain_response"

# Internal meetings - detailed transcription
class InternalMeetingTranscriber(MeetingTranscriber):
    """Detailed transcriber for internal meetings."""

    response_format = "verbose_json"  # Override for timestamps

# Client meetings - clean, formatted output
class ClientMeetingTranscriber(MeetingTranscriber):
    """Clean transcriber for client-facing meetings."""

    # Inherits text format from base

stt = mf.Model.speech_to_text("openai/whisper-1")

# Create instances
internal = InternalMeetingTranscriber(
    model=stt,
    config={
        "language": "en",
        "timestamp_granularities": "segment"
    }
)

client = ClientMeetingTranscriber(
    model=stt,
    prompt="Professional business meeting. Clean transcription without filler words.",
    config={"language": "en"}
)

# Use appropriately
internal_notes = internal("team_standup.mp3")
client_summary = client("client_presentation.mp3")
```

## Integration with Agents

Transcribers can be used in agent workflows:

```python
import msgflux as mf

class VoiceCommandTranscriber(mf.nn.Transcriber):
    """Transcribes voice commands for agent processing."""

    response_format = "text"
    response_mode = "plain_response"

# Create transcriber
stt = mf.Model.speech_to_text("openai/whisper-1")
transcriber = VoiceCommandTranscriber(
    model=stt,
    config={"language": "en"}
)

# Create agent
class VoiceAssistant(mf.nn.Agent):
    """Voice-controlled assistant."""

    temperature = 0.7
    max_tokens = 150

model = mf.Model.chat_completion("openai/gpt-4")
assistant = VoiceAssistant(model=model)

# Process voice command
voice_input = "voice_command.mp3"

# Transcribe
command_text = transcriber(voice_input)
print(f"Command: {command_text}")

# Process with agent
response = assistant(command_text)
print(f"Response: {response}")
```

## Batch Transcription

Transcribe multiple files efficiently:

```python
import msgflux as mf
import asyncio

class BatchTranscriber(mf.nn.Transcriber):
    """Transcriber for batch processing multiple files."""

    response_format = "text"
    response_mode = "plain_response"

stt = mf.Model.speech_to_text("openai/whisper-1")

transcriber = BatchTranscriber(
    model=stt,
    config={"language": "en"}
)

async def transcribe_batch(audio_files):
    """Transcribe multiple audio files concurrently."""
    tasks = [transcriber.aforward(file) for file in audio_files]
    return await asyncio.gather(*tasks)

# Transcribe batch
audio_files = ["file1.mp3", "file2.mp3", "file3.mp3"]
transcripts = asyncio.run(transcribe_batch(audio_files))

for file, transcript in zip(audio_files, transcripts):
    print(f"{file}: {transcript}")
```

## Configuration Options

### Complete Parameter Reference

```python
import msgflux as mf

class FullyConfiguredTranscriber(mf.nn.Transcriber):
    """Transcriber with all configuration options."""

    # Response behavior
    response_format = "text"  # "text", "json", "srt", "verbose_json", "vtt"
    response_mode = "plain_response"  # or "message"

stt = mf.Model.speech_to_text("openai/whisper-1")

transcriber = FullyConfiguredTranscriber(
    model=stt,                           # Required: STT model
    message_fields={                     # Optional: Message field mapping
        "task_multimodal_inputs": "audio.file",
        "model_preference": "model.choice"
    },
    response_template="...",             # Optional: Jinja template
    prompt="Transcription guidance",     # Optional: prompt for accuracy
    config={                             # Optional: STT-specific config
        "language": "en",                # Language code (en, es, fr, etc.)
        "stream": False,                 # Enable streaming
        "timestamp_granularities": "word",  # "word", "segment", or None
        "temperature": 0.0               # Sampling temperature (0-1)
    },
    name="custom_transcriber"            # Optional: custom name
)
```

## Best Practices

### 1. Use Language-Specific Transcribers

```python
# Good - Optimized per language
class EnglishTranscriber(mf.nn.Transcriber):
    """Optimized for English audio."""
    response_format = "text"

class JapaneseTranscriber(mf.nn.Transcriber):
    """Optimized for Japanese audio."""
    response_format = "text"

english = EnglishTranscriber(model=stt, config={"language": "en"})
japanese = JapaneseTranscriber(model=stt, config={"language": "ja"})
```

### 2. Use Prompts for Domain-Specific Audio

```python
# Good - Domain-specific prompts improve accuracy
class TechnicalTranscriber(mf.nn.Transcriber):
    """Transcriber for technical content."""
    response_format = "text"

technical = TechnicalTranscriber(
    model=stt,
    prompt="Technical presentation with terms: API, database, microservices, kubernetes",
    config={"language": "en"}
)
```

### 3. Choose Appropriate Response Format

```python
# Text for simple transcription
class SimpleTranscriber(mf.nn.Transcriber):
    """Simple text transcription."""
    response_format = "text"

# Verbose JSON for detailed analysis
class DetailedTranscriber(mf.nn.Transcriber):
    """Detailed transcription with timestamps."""
    response_format = "verbose_json"

# SRT for video subtitles
class SubtitleTranscriber(mf.nn.Transcriber):
    """Generate subtitles for videos."""
    response_format = "srt"
```

## Migration Guide

### From Traditional to AutoParams

**Before (Traditional):**
```python
transcriber = mf.nn.Transcriber(
    model=stt,
    response_format="text",
    config={"language": "en", "temperature": 0.0}
)
```

**After (AutoParams - Recommended):**
```python
class MyTranscriber(mf.nn.Transcriber):
    """Accurate transcriber for English audio."""
    response_format = "text"

transcriber = MyTranscriber(
    model=stt,
    config={"language": "en", "temperature": 0.0}
)
```

## Summary

- **Use AutoParams** for defining transcribers - create specialized transcribers for different domains
- **Traditional initialization** works for quick, one-off transcriptions
- Supports **multiple output formats** (text, JSON, SRT, verbose JSON, VTT)
- Configure **language**, **timestamps**, and **streaming** options
- Use **prompts** to improve accuracy for domain-specific audio
- **Async support** for batch transcription
- Integrate with **Agents** for voice-controlled workflows

The Transcriber module provides flexible speech-to-text - use AutoParams to organize transcribers by language, domain, and use case.
