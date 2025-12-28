# Quickstart: PIX --- [Voice, Text]

This example demonstrates how to create a simple PIX transaction workflow that can handle both text and audio inputs.

## Setup

```python
import msgflux as mf
import msgflux.nn as nn

# Set up API key (use environment variables in production)
from google.colab import userdata
api_key = userdata.get("OPENAI_API_KEY")
mf.set_envs(OPENAI_API_KEY=api_key)

# Create models
chat_model = mf.Model.chat_completion("openai/gpt-4o-mini")
stt_model = mf.Model.speech_to_text("openai/whisper-1")
```

## Define PIX Workflow

```python
# Define signature for PIX extraction
signature = """text -> amount: float, key_type: Literal['cpf', 'cnpj', 'email', 'phone_number', 'name'], key_id: str"""

class PIX(nn.Module):
    def __init__(self):
        super().__init__()

        # Transcriber agent - converts audio to text
        self.transcriber = nn.Agent(
            name="transcriber",
            model=stt_model,
            message_fields={
                "task_multimodal_inputs": {"audio": "user_audio"}
            },
            config={
                "return_content_only": True
            }
        )

        # Extractor agent - extracts PIX information
        self.extractor = nn.Agent(
            name="extractor",
            model=chat_model,
            signature=signature,
            message_fields={
                "task_inputs": "content",
                "task_multimodal_inputs": {"image": "user_image"}
            },
            config={
                "return_extraction_only": True
            }
        )

        # Define workflow: if audio exists, transcribe first, then extract
        self.components = nn.ModuleDict({
            "transcriber": self.transcriber,
            "extractor": self.extractor
        })

        self.register_buffer("flux", "{user_audio is not None? transcriber} -> extractor")

    def forward(self, msg):
        return mf.inline(self.flux, self.components, msg)

# Create PIX instance
pix = PIX()

```

## Usage Examples

### Text Input

```python
# Create message with text content
# en: "Send 22.40 to 123.456.789-00"
# cpf: Brazilian personal ID (CPF)
msg = mf.Message(content="Envie 22,40 para 123.456.789-00")

# Process through PIX workflow
result = pix(msg)

# View extracted information
print(result.response)
# {
#     'amount': 22.40,
#     'key_type': 'cpf',
#     'key_id': '123.456.789-00'
# }
```

### Audio Input

```python
# Create message with audio file
msg = mf.Message()
msg.set("user_audio", "audio-pix-direct-pt-br.ogg")

# Process: transcribe audio -> extract PIX info
result = pix(msg)

# Transcription is stored in content
print(result.content)  # Transcribed text

# Extraction result
print(result.response)
# {
#     'amount': 23.50,
#     'key_type': 'phone_number',
#     'key_id': '84999242111'
# }
```

### Text + Image Input

```python
# Create message with text and image
msg = mf.Message(
    content="Pague o valor para o destinatário na imagem",
    user_image="pix_qrcode.png"
)

# Extractor can use both text instruction and image content
result = pix(msg)

print(result.response)
```

## How It Works

1. **Conditional Flow**: The workflow checks if `user_audio` exists
   - If audio exists: `transcriber` → `extractor`
   - If no audio: `extractor` only

2. **Transcriber Agent**:
   - Takes audio from `msg.user_audio`
   - Uses speech-to-text model
   - Outputs transcribed text to `msg.content`

3. **Extractor Agent**:
   - Takes text from `msg.content`
   - Optionally takes image from `msg.user_image`
   - Uses signature to extract structured PIX data
   - Returns: amount, key_type, and key_id

## Inspecting the Module

```python
# View module structure
print(pix)

# View state dictionary
state = pix.state_dict()
print(state)

# Save module configuration
mf.save(state, "pix_workflow.json")

# Load later
loaded_state = mf.load("pix_workflow.json")
pix_loaded = PIX()
pix_loaded.load_state_dict(loaded_state)
```