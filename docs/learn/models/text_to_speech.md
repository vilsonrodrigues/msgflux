# Text to Speech

The `text_to_speech` model converts text into natural-sounding spoken audio. These models enable voice generation for accessibility, content creation, virtual assistants, and more.

**All code examples use the recommended import pattern:**

```python
import msgflux as mf
```

## Overview

Text-to-speech (TTS) models transform written text into spoken audio. They enable:

- **Voice Synthesis**: Convert any text to natural-sounding speech
- **Voice Selection**: Choose from different voice profiles
- **Speed Control**: Adjust speaking rate
- **Format Options**: Generate audio in various formats
- **Streaming**: Real-time audio generation

### Common Use Cases

- **Accessibility**: Convert text to speech for visually impaired users
- **Content Creation**: Generate voiceovers for videos and podcasts
- **Virtual Assistants**: Add voice to chatbots and AI assistants
- **Audio Books**: Convert written content to audio format
- **Language Learning**: Pronunciation examples
- **Notifications**: Voice alerts and announcements

## Quick Start

### Basic Usage

```python
import msgflux as mf

# Create TTS model
model = mf.Model.text_to_speech("openai/tts-1")

# Generate speech
response = model("Hello, how are you today?")

# Get audio file path
audio_path = response.consume()
print(audio_path)  # /tmp/tmpXXXXXX.opus
```

### With Custom Voice

```python
import msgflux as mf

model = mf.Model.text_to_speech(
    "openai/tts-1",
    voice="nova",  # Female voice
    speed=1.0      # Normal speed
)

response = model("Welcome to our service!")
audio_path = response.consume()
```

## Supported Providers

### OpenAI

```python
import msgflux as mf

# Standard quality (faster, cheaper)
model = mf.Model.text_to_speech("openai/tts-1")

# HD quality (higher quality, slower)
model = mf.Model.text_to_speech("openai/tts-1-hd")
```

### Together AI

```python
import msgflux as mf

# Together AI text-to-speech
model = mf.Model.text_to_speech("together/tts-1")
```

## Voice Options

### Available Voices (OpenAI)

```python
import msgflux as mf

# Alloy (neutral)
model = mf.Model.text_to_speech("openai/tts-1", voice="alloy")

# Echo (male)
model = mf.Model.text_to_speech("openai/tts-1", voice="echo")

# Fable (neutral, expressive)
model = mf.Model.text_to_speech("openai/tts-1", voice="fable")

# Onyx (male, deeper)
model = mf.Model.text_to_speech("openai/tts-1", voice="onyx")

# Nova (female, energetic)
model = mf.Model.text_to_speech("openai/tts-1", voice="nova")

# Shimmer (female, warm)
model = mf.Model.text_to_speech("openai/tts-1", voice="shimmer")
```

### Voice Characteristics

| Voice | Gender | Characteristics |
|-------|--------|-----------------|
| **alloy** | Neutral | Balanced, general purpose |
| **echo** | Male | Clear, professional |
| **fable** | Neutral | Expressive, storytelling |
| **onyx** | Male | Deep, authoritative |
| **nova** | Female | Bright, energetic |
| **shimmer** | Female | Warm, friendly |

## Speed Control

Adjust the speaking rate:

```python
import msgflux as mf

model = mf.Model.text_to_speech("openai/tts-1")

# Slow (0.25x - 1.0x)
response = model(
    "This will be spoken slowly.",
    speed=0.5
)

# Normal (default)
response = model(
    "This will be spoken at normal speed.",
    speed=1.0
)

# Fast (1.0x - 4.0x)
response = model(
    "This will be spoken quickly.",
    speed=2.0
)
```

Note: Speed parameter accepts values from 0.25 to 4.0.

## Audio Formats

### Supported Formats

```python
import msgflux as mf

model = mf.Model.text_to_speech("openai/tts-1")

# Opus (default, best for streaming)
response = model(
    "Hello world",
    response_format="opus"
)

# MP3 (universal compatibility)
response = model(
    "Hello world",
    response_format="mp3"
)

# AAC (good quality, small size)
response = model(
    "Hello world",
    response_format="aac"
)

# FLAC (lossless, large)
response = model(
    "Hello world",
    response_format="flac"
)

# WAV (uncompressed)
response = model(
    "Hello world",
    response_format="wav"
)

# PCM (raw audio)
response = model(
    "Hello world",
    response_format="pcm"
)
```

### Format Comparison

| Format | Quality | Size | Use Case |
|--------|---------|------|----------|
| **opus** | High | Small | Streaming, real-time |
| **mp3** | Good | Medium | Universal playback |
| **aac** | High | Small | Mobile, web |
| **flac** | Lossless | Large | Archival, editing |
| **wav** | Lossless | Large | Professional audio |
| **pcm** | Raw | Largest | Audio processing |

## Voice Instructions

Control voice characteristics with prompts:

```python
import msgflux as mf

model = mf.Model.text_to_speech("openai/tts-1")

# Add emotional tone
response = model(
    "I'm so excited about this!",
    prompt="Speak with enthusiasm and energy"
)

# Control pacing
response = model(
    "This is an important announcement.",
    prompt="Speak slowly and clearly, emphasizing each word"
)

# Set context
response = model(
    "Welcome to the show!",
    prompt="Speak as a radio host, upbeat and friendly"
)
```

Note: Voice instructions work best with tts-1-hd model.

## Streaming Audio

Generate and play audio in real-time:

```python
import msgflux as mf

model = mf.Model.text_to_speech("openai/tts-1")

# Stream audio
response = model(
    "This is a long text that will be streamed as audio chunks...",
    stream=True
)

# Process chunks as they arrive
for chunk in response.consume():
    if chunk is None:  # End of stream
        break
    # chunk is bytes - play or save incrementally
    process_audio_chunk(chunk)
```

### Streaming to File

```python
import msgflux as mf

model = mf.Model.text_to_speech("openai/tts-1")

response = model(
    "This will be streamed to a file.",
    stream=True,
    response_format="mp3"
)

# Write chunks to file
with open("output.mp3", "wb") as f:
    for chunk in response.consume():
        if chunk is None:
            break
        f.write(chunk)

print("Audio saved to output.mp3")
```

### Streaming Playback

```python
import msgflux as mf
import pyaudio  # pip install pyaudio

model = mf.Model.text_to_speech("openai/tts-1")

# Setup audio playback
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=24000,  # 24kHz for TTS
    output=True
)

# Stream and play
response = model(
    "This will be played in real-time.",
    stream=True,
    response_format="pcm"
)

for chunk in response.consume():
    if chunk is None:
        break
    stream.write(chunk)

stream.stop_stream()
stream.close()
p.terminate()
```

## Async Support

Generate audio asynchronously:

```python
import msgflux as mf
import asyncio

model = mf.Model.text_to_speech("openai/tts-1")

async def generate_speech(text):
    response = await model.acall(text, voice="nova")
    return response.consume()

async def main():
    # Generate multiple audio files concurrently
    texts = [
        "First announcement",
        "Second announcement",
        "Third announcement"
    ]

    tasks = [generate_speech(text) for text in texts]
    audio_paths = await asyncio.gather(*tasks)

    for text, path in zip(texts, audio_paths):
        print(f"{text}: {path}")

asyncio.run(main())
```

### Async Streaming

```python
import msgflux as mf
import asyncio

model = mf.Model.text_to_speech("openai/tts-1")

async def stream_speech(text):
    response = await model.acall(text, stream=True)

    async for chunk in response.consume():
        if chunk is None:
            break
        # Process chunk asynchronously
        await process_chunk(chunk)

asyncio.run(stream_speech("Hello world"))
```

## Batch Processing

Generate multiple audio files:

```python
import msgflux as mf
import msgflux.nn.functional as F

model = mf.Model.text_to_speech("openai/tts-1", voice="nova")

texts = [
    "Welcome to chapter one.",
    "Welcome to chapter two.",
    "Welcome to chapter three."
]

# Generate in parallel
results = F.map_gather(
    model,
    args_list=[(text,) for text in texts]
)

# Save all files
for i, result in enumerate(results):
    audio_path = result.consume()
    # Copy to permanent location
    import shutil
    shutil.copy(audio_path, f"chapter_{i+1}.opus")
```

## Working with Audio Files

### Save to Specific Location

```python
import msgflux as mf
import shutil

model = mf.Model.text_to_speech("openai/tts-1")

response = model(
    "Save this audio",
    response_format="mp3"
)

# Get temporary file path
temp_path = response.consume()

# Copy to desired location
shutil.copy(temp_path, "output.mp3")
print("Saved to output.mp3")
```

### Play Audio

```python
import msgflux as mf
import subprocess

model = mf.Model.text_to_speech("openai/tts-1")

response = model("Play this message")
audio_path = response.consume()

# Play with system player
subprocess.run(["mpv", audio_path])  # Or use "afplay" on macOS
```

### Get Audio Info

```python
import msgflux as mf
from pydub import AudioSegment

model = mf.Model.text_to_speech("openai/tts-1")

response = model(
    "Get information about this audio",
    response_format="mp3"
)

audio_path = response.consume()

# Load with pydub
audio = AudioSegment.from_mp3(audio_path)

print(f"Duration: {len(audio) / 1000:.2f} seconds")
print(f"Channels: {audio.channels}")
print(f"Frame rate: {audio.frame_rate} Hz")
print(f"Sample width: {audio.sample_width} bytes")
```

## Common Patterns

### Multi-Voice Narration

```python
import msgflux as mf
from pydub import AudioSegment

# Create models with different voices
narrator = mf.Model.text_to_speech("openai/tts-1", voice="fable")
character1 = mf.Model.text_to_speech("openai/tts-1", voice="nova")
character2 = mf.Model.text_to_speech("openai/tts-1", voice="onyx")

# Generate dialogue
narration = narrator("The story begins").consume()
line1 = character1("Hello there!").consume()
line2 = character2("Hi, how are you?").consume()

# Combine audio
from pydub import AudioSegment

combined = AudioSegment.from_file(narration)
combined += AudioSegment.from_file(line1)
combined += AudioSegment.from_file(line2)

combined.export("dialogue.mp3", format="mp3")
```

### Text-to-Audio Book

```python
import msgflux as mf
from pydub import AudioSegment

model = mf.Model.text_to_speech("openai/tts-1-hd", voice="fable")

# Split book into chapters
chapters = [
    "Chapter 1: Once upon a time...",
    "Chapter 2: The adventure begins...",
    "Chapter 3: A challenge appears..."
]

# Generate audio for each chapter
audio_segments = []

for i, chapter_text in enumerate(chapters):
    print(f"Generating chapter {i+1}...")
    response = model(chapter_text, response_format="mp3")
    audio_path = response.consume()
    audio_segments.append(AudioSegment.from_mp3(audio_path))

# Combine with silence between chapters
combined = audio_segments[0]
silence = AudioSegment.silent(duration=2000)  # 2 seconds

for segment in audio_segments[1:]:
    combined += silence + segment

combined.export("audiobook.mp3", format="mp3")
print("Audiobook created!")
```

### Real-time Chat TTS

```python
import msgflux as mf
import subprocess
import tempfile

model = mf.Model.text_to_speech("openai/tts-1", voice="nova")

def speak(text):
    """Convert text to speech and play immediately."""
    response = model(text, response_format="mp3")
    audio_path = response.consume()

    # Play immediately
    subprocess.run(
        ["mpv", "--really-quiet", audio_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# Use in chat
speak("Hello! How can I help you today?")
user_input = input("You: ")
speak(f"You said: {user_input}")
```

### Language Learning

```python
import msgflux as mf

model_slow = mf.Model.text_to_speech("openai/tts-1", voice="nova", speed=0.7)
model_normal = mf.Model.text_to_speech("openai/tts-1", voice="nova", speed=1.0)

phrase = "The quick brown fox jumps over the lazy dog"

# Slow version for learning
slow_audio = model_slow(phrase, response_format="mp3").consume()

# Normal speed for practice
normal_audio = model_normal(phrase, response_format="mp3").consume()

print(f"Slow: {slow_audio}")
print(f"Normal: {normal_audio}")
```

### Podcast Generation

```python
import msgflux as mf
from pydub import AudioSegment

# Create hosts with different voices
host1 = mf.Model.text_to_speech("openai/tts-1-hd", voice="echo")
host2 = mf.Model.text_to_speech("openai/tts-1-hd", voice="shimmer")

# Podcast script
intro = host1("Welcome to our podcast!").consume()
response1 = host2("Thanks for having me!").consume()
discussion = host1("Let's talk about AI...").consume()

# Combine with music/effects
podcast = AudioSegment.from_file(intro)
podcast += AudioSegment.from_file(response1)
podcast += AudioSegment.from_file(discussion)

podcast.export("podcast.mp3", format="mp3")
```

## Best Practices

### 1. Choose the Right Model

```python
import msgflux as mf

# For production/client work - HD quality
model_hd = mf.Model.text_to_speech("openai/tts-1-hd", voice="nova")

# For testing/iteration - standard quality
model_std = mf.Model.text_to_speech("openai/tts-1", voice="nova")
```

### 2. Text Preprocessing

```python
import msgflux as mf

model = mf.Model.text_to_speech("openai/tts-1")

def prepare_text(text):
    """Prepare text for better TTS output."""
    # Remove excessive whitespace
    text = " ".join(text.split())

    # Add pauses with punctuation
    text = text.replace(". ", "... ")

    # Spell out abbreviations if needed
    text = text.replace("Dr.", "Doctor")
    text = text.replace("Mr.", "Mister")

    return text

text = "Dr. Smith said hello.  How are you?"
clean_text = prepare_text(text)
response = model(clean_text)
```

### 3. Handle Long Texts

```python
import msgflux as mf

model = mf.Model.text_to_speech("openai/tts-1")

def split_text(text, max_length=4000):
    """Split long text into chunks."""
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

long_text = "..." * 10000  # Very long text
chunks = split_text(long_text)

audio_files = []
for chunk in chunks:
    response = model(chunk)
    audio_files.append(response.consume())
```

### 4. Use Streaming for Large Content

```python
import msgflux as mf

model = mf.Model.text_to_speech("openai/tts-1")

# For long content, stream to avoid memory issues
long_text = "..." * 5000

response = model(long_text, stream=True, response_format="mp3")

with open("long_audio.mp3", "wb") as f:
    for chunk in response.consume():
        if chunk is None:
            break
        f.write(chunk)
```

### 5. Cache Common Phrases

```python
import msgflux as mf
import hashlib

model = mf.Model.text_to_speech("openai/tts-1")

audio_cache = {}

def cached_tts(text):
    """Cache generated audio for reuse."""
    cache_key = hashlib.md5(text.encode()).hexdigest()

    if cache_key in audio_cache:
        return audio_cache[cache_key]

    response = model(text)
    audio_path = response.consume()
    audio_cache[cache_key] = audio_path

    return audio_path

# These will only generate once
audio1 = cached_tts("Welcome!")
audio2 = cached_tts("Welcome!")  # Uses cache
```

## Error Handling

```python
import msgflux as mf

model = mf.Model.text_to_speech("openai/tts-1")

try:
    response = model("Hello world")
    audio_path = response.consume()
except ImportError:
    print("Provider not installed")
except ValueError as e:
    print(f"Invalid parameters: {e}")
    # Common issues:
    # - Invalid voice name
    # - Speed out of range (0.25-4.0)
    # - Invalid response_format
except Exception as e:
    print(f"Generation failed: {e}")
    # Common errors:
    # - Rate limits
    # - Network issues
    # - Text too long
```

## Limitations

### OpenAI TTS Limitations

- **Text Length**: Maximum ~4096 characters per request
- **Speed Range**: 0.25 to 4.0 only
- **Language**: Supports multiple languages but optimized for English
- **Voice Count**: Limited to 6 predefined voices
- **Real-time Factor**: Not exactly real-time (some processing delay)

## Cost Optimization

### Efficient TTS Usage

```python
import msgflux as mf

# Use standard model for development
dev_model = mf.Model.text_to_speech("openai/tts-1")

# Use HD only for production
prod_model = mf.Model.text_to_speech("openai/tts-1-hd")

# Reuse audio for common phrases
common_phrases = {
    "welcome": dev_model("Welcome!").consume(),
    "goodbye": dev_model("Goodbye!").consume()
}
```

## See Also

- [Speech to Text](speech_to_text.md) - Transcribe audio to text
- [Chat Completion](chat_completion.md) - Generate text for TTS
- [Model](model.md) - Model factory and registry
