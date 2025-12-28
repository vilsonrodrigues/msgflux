# Installation

This guide covers how to install msgFlux using different package managers.

## Requirements

- **Python**: 3.10+
- **Operating System**: Linux, macOS or Windows

## Quick Installation

=== "uv (recommended)"

    [uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver, significantly faster than pip.

    **Install uv:**

    === "Linux/macOS"
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

    === "Windows"
        ```powershell
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

    === "With pip"
        ```bash
        pip install uv
        ```

    **Install msgFlux:**

    ```bash
    # Basic installation
    uv pip install msgflux

    # With OpenAI support
    uv pip install msgflux[openai]

    # With multiple providers
    uv pip install "msgflux[openai,google,anthropic]"
    ```

=== "pip"

    Standard installation using pip:

    ```bash
    # Basic installation
    pip install msgflux

    # With OpenAI support
    pip install msgflux[openai]

    # With multiple providers
    pip install "msgflux[openai,google,anthropic]"
    ```

## Provider Support

Install extras for specific AI providers:

=== "uv"

    ```bash
    # OpenAI (GPT, DALL-E, Whisper, TTS)
    uv pip install msgflux[openai]

    # Google (Gemini)
    uv pip install msgflux[google]

    # Anthropic (Claude)
    uv pip install msgflux[anthropic]

    # Groq
    uv pip install msgflux[groq]

    # Together AI
    uv pip install msgflux[together]

    # Ollama (local models)
    uv pip install msgflux[ollama]

    # Multiple providers
    uv pip install "msgflux[openai,google,anthropic]"
    ```

=== "pip"

    ```bash
    # OpenAI (GPT, DALL-E, Whisper, TTS)
    pip install msgflux[openai]

    # Google (Gemini)
    pip install msgflux[google]

    # Anthropic (Claude)
    pip install msgflux[anthropic]

    # Groq
    pip install msgflux[groq]

    # Together AI
    pip install msgflux[together]

    # Ollama (local models)
    pip install msgflux[ollama]

    # Multiple providers
    pip install "msgflux[openai,google,anthropic]"
    ```

## Installation from GitHub

Install directly from source without cloning:

=== "uv"

    ```bash
    # Latest from main branch
    uv pip install git+https://github.com/msgflux/msgflux.git

    # With extras
    uv pip install "git+https://github.com/msgflux/msgflux.git#egg=msgflux[openai]"

    # Specific branch
    uv pip install git+https://github.com/msgflux/msgflux.git@branch-name

    # Specific tag/release
    uv pip install git+https://github.com/msgflux/msgflux.git@v0.1.0

    # Specific commit
    uv pip install git+https://github.com/msgflux/msgflux.git@abc123def

    # From a fork
    uv pip install git+https://github.com/your-username/msgflux.git@feature-branch
    ```

=== "pip"

    ```bash
    # Latest from main branch
    pip install git+https://github.com/msgflux/msgflux.git

    # With extras
    pip install "git+https://github.com/msgflux/msgflux.git#egg=msgflux[openai]"

    # Specific branch
    pip install git+https://github.com/msgflux/msgflux.git@branch-name

    # Specific tag/release
    pip install git+https://github.com/msgflux/msgflux.git@v0.1.0

    # Specific commit
    pip install git+https://github.com/msgflux/msgflux.git@abc123def

    # From a fork
    pip install git+https://github.com/your-username/msgflux.git@feature-branch
    ```

### Clone and Install (Development)

For local development with editable installation:

=== "uv"

    ```bash
    git clone https://github.com/msgflux/msgflux.git
    cd msgflux
    uv pip install -e ".[dev]"
    ```

=== "pip"

    ```bash
    git clone https://github.com/msgflux/msgflux.git
    cd msgflux
    pip install -e ".[dev]"
    ```

## Configuration

### Set API Keys

=== "Environment Variables"

    ```bash
    export OPENAI_API_KEY="sk-..."
    export GOOGLE_API_KEY="..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    ```

=== ".env File"

    Create a `.env` file in your project:

    ```env
    OPENAI_API_KEY=sk-...
    GOOGLE_API_KEY=...
    ANTHROPIC_API_KEY=sk-ant-...
    ```

=== "In Code"

    ```python
    import msgflux as mf

    mf.set_envs(
        OPENAI_API_KEY="sk-...",
        GOOGLE_API_KEY="...",
        ANTHROPIC_API_KEY="sk-ant-..."
    )
    ```

## Verify Installation

```python
import msgflux as mf

# Check version
print(mf.__version__)

# List available providers
providers = mf.Model.providers()
print(providers['chat_completion'])
```

## Quick Example

```python
import msgflux as mf

# Set API key
mf.set_envs(OPENAI_API_KEY="sk-...")

# Create model
model = mf.Model.chat_completion("openai/gpt-4o")

# Generate response
response = model(messages=[
    {"role": "user", "content": "Hello!"}
])

print(response.consume())
```

## Upgrade

=== "uv"

    ```bash
    uv pip install --upgrade msgflux
    uv pip install --upgrade "msgflux[openai,google]"
    ```

=== "pip"

    ```bash
    pip install --upgrade msgflux
    pip install --upgrade "msgflux[openai,google]"
    ```

## Troubleshooting

### Import Error

```bash
# Check installation
pip list | grep msgflux
```

### Provider Not Available

```bash
# Install provider extra
uv pip install msgflux[openai]
```

### API Key Not Found

```bash
# Set environment variable
export OPENAI_API_KEY="sk-..."
```

## Next Steps

- **[Quick Start](quickstart.md)** - Get started with examples
- **[Models](models/model.md)** - Learn about model types
- **[Chat Completion](models/chat_completion.md)** - Build conversational AI
