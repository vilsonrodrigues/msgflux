# Speech to Text

The `speech_to_text` model transcribes spoken audio into written text. These models enable voice-to-text conversion for accessibility, transcription services, voice commands, and more.

**All code examples use the recommended import pattern:**

```python
import msgflux as mf
```

## Overview

Speech-to-text (STT) models convert audio recordings into text transcripts. They enable:

- **Transcription**: Convert spoken audio to written text
- **Timestamping**: Get word and segment-level timestamps
- **Multiple Formats**: Output as text, JSON, SRT, VTT subtitles
- **Language Detection**: Automatic or manual language specification
- **Context Awareness**: Use prompts to improve accuracy

### Common Use Cases

- **Meeting Transcription**: Convert recordings to searchable text
- **Subtitle Generation**: Create subtitles for videos
- **Voice Commands**: Process spoken user commands
- **Accessibility**: Provide captions for audio content
- **Interview Analysis**: Transcribe interviews and podcasts
- **Call Center**: Analyze customer service calls

## Quick Start

### Basic Usage

```python
import msgflux as mf

# Create STT model
model = mf.Model.speech_to_text("openai/whisper-1")

# Transcribe audio file
response = model("path/to/audio.mp3")

# Get transcript
transcript = response.consume()
print(transcript["text"])
# "Hello, this is a test recording."
```

### From URL

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

# Transcribe from URL
response = model("https://example.com/audio.mp3")
transcript = response.consume()
print(transcript["text"])
```

## Supported Providers

### OpenAI (Whisper)

```python
import msgflux as mf

# Whisper-1 (multilingual, robust)
model = mf.Model.speech_to_text("openai/whisper-1")
```

Whisper supports:
- **98 languages** including English, Spanish, French, German, Chinese, Japanese
- **Multiple audio formats**: mp3, mp4, mpeg, mpga, m4a, wav, webm
- **File size**: Up to 25 MB

## Response Formats

### Text Format (Default)

Simple text output:

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

response = model(
    "audio.mp3",
    response_format="text"
)

transcript = response.consume()
print(transcript["text"])
# "This is the transcribed text."
```

### JSON Format

Structured output:

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

response = model(
    "audio.mp3",
    response_format="json"
)

transcript = response.consume()
print(transcript)
# {"text": "This is the transcribed text."}
```

### Verbose JSON

Detailed output with metadata:

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

response = model(
    "audio.mp3",
    response_format="verbose_json"
)

transcript = response.consume()
print(transcript)
# {
#     "text": "This is the transcribed text.",
#     "language": "en",
#     "duration": 5.2,
#     "segments": [...]
# }
```

### SRT (SubRip) Format

Subtitle format for videos:

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

response = model(
    "audio.mp3",
    response_format="srt"
)

transcript = response.consume()
print(transcript["text"])
# 1
# 00:00:00,000 --> 00:00:02,000
# This is the first subtitle
#
# 2
# 00:00:02,000 --> 00:00:05,000
# This is the second subtitle
```

### VTT (WebVTT) Format

Web-friendly subtitle format:

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

response = model(
    "audio.mp3",
    response_format="vtt"
)

transcript = response.consume()
print(transcript["text"])
# WEBVTT
#
# 00:00:00.000 --> 00:00:02.000
# This is the first subtitle
#
# 00:00:02.000 --> 00:00:05.000
# This is the second subtitle
```

## Timestamp Granularities

### Word-Level Timestamps

Get timestamp for each word:

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

response = model(
    "audio.mp3",
    response_format="verbose_json",
    timestamp_granularities=["word"]
)

transcript = response.consume()
print(transcript["words"])
# [
#     {"word": "Hello", "start": 0.0, "end": 0.5},
#     {"word": "world", "start": 0.6, "end": 1.1}
# ]
```

### Segment-Level Timestamps

Get timestamps for phrases/segments:

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

response = model(
    "audio.mp3",
    response_format="verbose_json",
    timestamp_granularities=["segment"]
)

transcript = response.consume()
print(transcript["segments"])
# [
#     {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello world."},
#     {"id": 1, "start": 2.5, "end": 5.0, "text": "How are you?"}
# ]
```

### Both Word and Segment Timestamps

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

response = model(
    "audio.mp3",
    response_format="verbose_json",
    timestamp_granularities=["word", "segment"]
)

transcript = response.consume()
print("Words:", transcript["words"])
print("Segments:", transcript["segments"])
```

## Language Specification

### Automatic Detection

By default, Whisper auto-detects the language:

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

response = model("audio.mp3")
transcript = response.consume()
# Language automatically detected
```

### Manual Language Specification

Improve accuracy and speed by specifying the language:

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

# English
response = model("audio.mp3", language="en")

# Spanish
response = model("audio.mp3", language="es")

# French
response = model("audio.mp3", language="fr")

# Japanese
response = model("audio.mp3", language="ja")

# Chinese
response = model("audio.mp3", language="zh")
```

### ISO 639-1 Language Codes

Common language codes:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese
- `ar` - Arabic
- `hi` - Hindi

## Context and Prompts

Improve transcription accuracy with context:

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

# Technical content
response = model(
    "meeting.mp3",
    prompt="This is a technical discussion about machine learning, neural networks, and AI"
)

# Names and terminology
response = model(
    "interview.mp3",
    prompt="Interview with Dr. Smith about quantum computing"
)

# Continuing previous segment
response = model(
    "part2.mp3",
    prompt="Previous text ended with: ...and that's how we solved the problem."
)
```

## Temperature Control

Control transcription randomness:

```python
import msgflux as mf

# Deterministic (temperature=0)
model = mf.Model.speech_to_text("openai/whisper-1", temperature=0.0)

# More creative (higher temperature)
model = mf.Model.speech_to_text("openai/whisper-1", temperature=0.3)
```

Note: Lower temperature = more conservative/repetitive, Higher temperature = more creative but potentially less accurate.

## Streaming

Process transcription in real-time:

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

# Stream transcription
response = model("long_audio.mp3", stream=True)

# Process chunks as they arrive
for chunk in response.consume():
    if chunk is None:
        break
    print(chunk, end="", flush=True)
```

## Async Support

Transcribe audio asynchronously:

```python
import msgflux as mf
import asyncio

model = mf.Model.speech_to_text("openai/whisper-1")

async def transcribe_audio(audio_path):
    response = await model.acall(audio_path)
    return response.consume()

async def main():
    # Transcribe multiple files concurrently
    audio_files = ["audio1.mp3", "audio2.mp3", "audio3.mp3"]

    tasks = [transcribe_audio(f) for f in audio_files]
    transcripts = await asyncio.gather(*tasks)

    for file, transcript in zip(audio_files, transcripts):
        print(f"{file}: {transcript['text']}")

asyncio.run(main())
```

## Batch Processing

Transcribe multiple files:

```python
import msgflux as mf
import msgflux.nn.functional as F

model = mf.Model.speech_to_text("openai/whisper-1")

audio_files = [
    "meeting1.mp3",
    "meeting2.mp3",
    "meeting3.mp3"
]

# Process in parallel
results = F.map_gather(
    model,
    args_list=[(f,) for f in audio_files]
)

# Get all transcripts
for file, result in zip(audio_files, results):
    transcript = result.consume()
    print(f"{file}:")
    print(transcript["text"])
    print()
```

## Common Patterns

### Meeting Transcription

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

def transcribe_meeting(audio_path, attendees=None):
    """Transcribe meeting with context."""
    prompt = ""
    if attendees:
        prompt = f"Meeting with {', '.join(attendees)}"

    response = model(
        audio_path,
        prompt=prompt,
        response_format="verbose_json",
        timestamp_granularities=["segment"]
    )

    transcript = response.consume()

    # Format output
    output = f"Meeting Transcript\n{'='*50}\n\n"

    for segment in transcript.get("segments", []):
        timestamp = f"[{segment['start']:.1f}s - {segment['end']:.1f}s]"
        output += f"{timestamp}\n{segment['text']}\n\n"

    return output

# Use it
transcript = transcribe_meeting(
    "meeting.mp3",
    attendees=["Alice", "Bob", "Carol"]
)
print(transcript)
```

### Subtitle Generation

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

def generate_subtitles(video_audio_path, output_path):
    """Generate SRT subtitles for video."""
    response = model(
        video_audio_path,
        response_format="srt",
        language="en"
    )

    transcript = response.consume()

    # Save subtitles
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript["text"])

    return output_path

# Generate subtitles
subtitle_file = generate_subtitles("video_audio.mp3", "subtitles.srt")
print(f"Subtitles saved to: {subtitle_file}")
```

### Podcast Transcription

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

def transcribe_podcast(audio_path, hosts=None, topic=None):
    """Transcribe podcast with metadata."""
    # Build context prompt
    prompt_parts = []
    if hosts:
        prompt_parts.append(f"Podcast hosts: {', '.join(hosts)}")
    if topic:
        prompt_parts.append(f"Topic: {topic}")

    prompt = ". ".join(prompt_parts) if prompt_parts else None

    response = model(
        audio_path,
        prompt=prompt,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"]
    )

    return response.consume()

# Transcribe
transcript = transcribe_podcast(
    "podcast.mp3",
    hosts=["Alice", "Bob"],
    topic="Artificial Intelligence"
)

print("Full text:", transcript["text"])
print(f"Duration: {transcript.get('duration', 'N/A')} seconds")
print(f"Language: {transcript.get('language', 'N/A')}")
```

### Multi-Language Support

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

def transcribe_multilingual(audio_files_with_langs):
    """Transcribe multiple files in different languages."""
    results = {}

    for audio_file, language in audio_files_with_langs:
        response = model(audio_file, language=language)
        transcript = response.consume()
        results[audio_file] = {
            "language": language,
            "text": transcript["text"]
        }

    return results

# Transcribe files in different languages
files_langs = [
    ("english.mp3", "en"),
    ("spanish.mp3", "es"),
    ("french.mp3", "fr")
]

transcripts = transcribe_multilingual(files_langs)

for file, data in transcripts.items():
    print(f"{file} ({data['language']}):")
    print(data['text'])
    print()
```

### Voice Command Processing

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

def process_voice_command(audio_path):
    """Process voice command."""
    response = model(
        audio_path,
        prompt="Voice command for controlling smart home devices",
        language="en",
        temperature=0.0  # Deterministic
    )

    transcript = response.consume()
    command_text = transcript["text"].lower().strip()

    # Parse command
    if "turn on" in command_text:
        device = command_text.replace("turn on", "").strip()
        return {"action": "turn_on", "device": device}
    elif "turn off" in command_text:
        device = command_text.replace("turn off", "").strip()
        return {"action": "turn_off", "device": device}
    else:
        return {"action": "unknown", "text": command_text}

# Process command
command = process_voice_command("command.mp3")
print(command)
# {"action": "turn_on", "device": "the lights"}
```

### Search Transcripts

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

def search_in_audio(audio_path, search_term):
    """Search for specific content in audio."""
    response = model(
        audio_path,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"]
    )

    transcript = response.consume()

    # Search segments
    results = []
    for segment in transcript.get("segments", []):
        if search_term.lower() in segment["text"].lower():
            results.append({
                "timestamp": f"{segment['start']:.1f}s - {segment['end']:.1f}s",
                "text": segment["text"]
            })

    return results

# Search
matches = search_in_audio("meeting.mp3", "budget")
for match in matches:
    print(f"[{match['timestamp']}] {match['text']}")
```

## Best Practices

### 1. Specify Language When Known

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

# Good - Faster and more accurate
response = model("english_audio.mp3", language="en")

# Less optimal - Requires language detection
response = model("english_audio.mp3")
```

### 2. Use Prompts for Context

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

# Good - Provides context
response = model(
    "tech_talk.mp3",
    prompt="Technical presentation about Kubernetes, Docker, and microservices"
)

# Less optimal - No context
response = model("tech_talk.mp3")
```

### 3. Choose Appropriate Response Format

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

# For simple transcription
response = model("audio.mp3", response_format="text")

# For subtitles
response = model("video.mp3", response_format="srt")

# For detailed analysis
response = model("interview.mp3", response_format="verbose_json",
                 timestamp_granularities=["word", "segment"])
```

### 4. Handle Long Audio Files

```python
import msgflux as mf
from pydub import AudioSegment

def split_audio(audio_path, chunk_length_ms=30000):
    """Split long audio into chunks."""
    audio = AudioSegment.from_file(audio_path)
    chunks = []

    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_path = f"/tmp/chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)

    return chunks

def transcribe_long_audio(audio_path):
    """Transcribe long audio file."""
    model = mf.Model.speech_to_text("openai/whisper-1")

    # Split into chunks
    chunks = split_audio(audio_path)

    # Transcribe each chunk
    full_transcript = ""
    previous_text = ""

    for chunk_path in chunks:
        # Use previous text for context
        response = model(
            chunk_path,
            prompt=previous_text[-500:] if previous_text else None
        )
        transcript = response.consume()
        chunk_text = transcript["text"]

        full_transcript += chunk_text + " "
        previous_text = chunk_text

    return full_transcript.strip()

transcript = transcribe_long_audio("long_audio.mp3")
```

### 5. Save Transcripts

```python
import msgflux as mf
import json

model = mf.Model.speech_to_text("openai/whisper-1")

def save_transcript(audio_path, output_path):
    """Transcribe and save with metadata."""
    response = model(
        audio_path,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"]
    )

    transcript = response.consume()

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    # Also save plain text
    text_path = output_path.replace(".json", ".txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(transcript["text"])

    return output_path

save_transcript("meeting.mp3", "meeting_transcript.json")
```

## Audio Format Support

### Supported Formats

- **MP3** (.mp3)
- **MP4** (.mp4, audio track)
- **MPEG** (.mpeg)
- **MPGA** (.mpga)
- **M4A** (.m4a)
- **WAV** (.wav)
- **WEBM** (.webm)

### File Size Limits

- Maximum file size: **25 MB**
- For larger files, split into chunks or compress

### Audio Preprocessing

```python
from pydub import AudioSegment

def prepare_audio(input_path, output_path):
    """Prepare audio for transcription."""
    audio = AudioSegment.from_file(input_path)

    # Normalize volume
    audio = audio.normalize()

    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Set sample rate to 16kHz (optimal for Whisper)
    audio = audio.set_frame_rate(16000)

    # Export as MP3
    audio.export(output_path, format="mp3", bitrate="64k")

    return output_path

# Prepare and transcribe
prepared = prepare_audio("raw_audio.wav", "prepared.mp3")

import msgflux as mf
model = mf.Model.speech_to_text("openai/whisper-1")
response = model(prepared)
```

## Error Handling

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

try:
    response = model("audio.mp3")
    transcript = response.consume()
except ImportError:
    print("Provider not installed")
except ValueError as e:
    print(f"Invalid parameters: {e}")
    # Common issues:
    # - Invalid language code
    # - Invalid response_format
    # - File too large (>25MB)
except FileNotFoundError:
    print("Audio file not found")
except Exception as e:
    print(f"Transcription failed: {e}")
    # Common errors:
    # - Unsupported audio format
    # - Corrupted audio file
    # - Network issues
    # - Rate limits
```

## Cost Optimization

### Efficient Transcription

```python
import msgflux as mf

model = mf.Model.speech_to_text("openai/whisper-1")

# Use language specification to reduce processing time
response = model("audio.mp3", language="en")

# Use simple format when timestamps not needed
response = model("audio.mp3", response_format="text")

# Compress audio before uploading
from pydub import AudioSegment

audio = AudioSegment.from_file("original.wav")
audio.export("compressed.mp3", format="mp3", bitrate="64k")
response = model("compressed.mp3")
```

## See Also

- [Text to Speech](text_to_speech.md) - Convert text to audio
- [Chat Completion](chat_completion.md) - Process transcripts with LLMs
- [Model](model.md) - Model factory and registry
