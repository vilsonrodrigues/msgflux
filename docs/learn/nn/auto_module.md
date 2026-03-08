# AutoModule

`AutoModule` loads `nn.Module` from remote repositories (GitHub or Hugging Face Hub) with a single line of code, enabling module sharing across projects and teams.

## Overview

`AutoModule` follows a Hub-based distribution model: the module author publishes a repository with a `config.json` and a Python file, and consumers load it by repository identifier.

**Key Features**:

- **Multi-source**: Load from GitHub or Hugging Face Hub
- **Local cache**: Downloaded files are cached at `~/.cache/msgflux/auto/`
- **Revision pinning**: Lock to a specific commit, tag, or branch
- **Model config**: Authors define default models in `config.json`; users override at instantiation
- **Security modes**: Choose between class-only loading or pre-instantiated objects

## Quick Start

```python
import msgflux as mf

# Load a class and instantiate it
AgentClass = mf.AutoModule("owner/repo")
agent = AgentClass()

# Load from Hugging Face Hub
AgentClass = mf.AutoModule("hf://owner/repo")

# Override model defaults at instantiation
agent = AgentClass(models={"lm": {"model_id": "groq/llama-3.1-8b"}})
```

## Creating a Shareable Module

### Repository structure

```
your-repo/
├── config.json
└── modeling.py
```

### `config.json`

```json
{
  "msgflux_class": "MyAgent",
  "msgflux_entrypoint": "modeling.py",
  "msgflux_version": ">=0.1.0",
  "sharing_mode": "class",
  "models": {
    "lm": {
      "model_id": "openai/gpt-5-mini",
      "temperature": 0.7
    },
    "embedder": {
      "model_id": "openai/text-embedding-3-small",
      "dimensions": 1536
    }
  }
}
```

| Field | Description |
|---|---|
| `msgflux_class` | Class name to import from the entrypoint file |
| `msgflux_entrypoint` | Python file containing the module |
| `msgflux_version` | Required msgflux version specifier (PEP 440) |
| `sharing_mode` | `"class"` or `"instance"` |
| `models` | Default model configs (optional) |

### `modeling.py`

```python
from msgflux.auto import load_model_configs
from msgflux.nn import LM, Embedder, Module


class MyAgent(Module):
    def __init__(self, models=None):
        super().__init__()
        cfg = load_model_configs(__file__, overrides=models)

        lm_cfg  = cfg["lm"]
        emb_cfg = cfg["embedder"]

        self.lm      = LM(lm_cfg["model_id"], temperature=lm_cfg.get("temperature", 0.7))
        self.embedder = Embedder(emb_cfg["model_id"])

    def forward(self, query: str) -> str:
        return self.lm(query)
```

`load_model_configs(__file__, overrides=models)` reads the `models` section from the `config.json` located next to the module file, and merges it with any overrides provided by the user. Unspecified keys keep their `config.json` defaults.

## Sharing Modes

### `class` — user controls instantiation

`AutoModule` returns the **class**. The user decides when and how to instantiate it.

```json
{ "sharing_mode": "class" }
```

```python
AgentClass = mf.AutoModule("owner/repo")

agent = AgentClass()
agent = AgentClass(models={"lm": {"model_id": "groq/llama-3.1-8b", "temperature": 0.3}})
```

This is the recommended mode: no remote code is executed until the user explicitly calls the class.

### `instance` — pre-instantiated object

`AutoModule` returns an **already created instance**. The module file is executed during import, which requires `trust_remote_code=True`.

```json
{
  "sharing_mode": "instance",
  "msgflux_entrypoint": "modeling.py:agent"
}
```

```python
# modeling.py
agent = MyAgent()  # instantiated at module level
```

```python
agent = mf.AutoModule("owner/repo", trust_remote_code=True)
```

Only use `instance` mode when you trust the repository author, as it executes arbitrary code.

## Model Configs

Each entry in the `models` field is a dict of configurations for that model. At minimum it contains `model_id`, but any additional key is valid and passed to the module constructor:

```json
{
  "models": {
    "lm": {
      "model_id": "openai/gpt-5-mini",
      "temperature": 0.7,
      "max_tokens": 2048
    },
    "embedder": {
      "model_id": "openai/text-embedding-3-small",
      "dimensions": 1536
    }
  }
}
```

The module author loads these defaults via `load_model_configs` and accepts a `models` parameter so users can override specific keys:

```python
class MyAgent(Module):
    def __init__(self, models=None):
        super().__init__()
        cfg = load_model_configs(__file__, overrides=models)

        lm_cfg = cfg["lm"]
        self.lm = LM(lm_cfg["model_id"], temperature=lm_cfg.get("temperature", 0.7))

# uses defaults from config.json
agent = AgentClass()

# overrides only "lm"; "embedder" keeps its config.json defaults
agent = AgentClass(models={"lm": {"model_id": "groq/llama-3.1-8b", "temperature": 0.3}})
```

Partial overrides are supported: only the specified keys are replaced; the rest fall back to `config.json` defaults.

## Loading Options

```python
# Explicit source
AgentClass = mf.AutoModule("hf://owner/repo")           # Hugging Face Hub
AgentClass = mf.AutoModule("gh://owner/repo")           # GitHub
AgentClass = mf.AutoModule("github.com/owner/repo")     # GitHub (URL format)
AgentClass = mf.AutoModule("huggingface.co/owner/repo") # Hugging Face (URL format)

# Pin revision (recommended for production)
AgentClass = mf.AutoModule("owner/repo", revision="abc123")

# Force re-download even if cached
AgentClass = mf.AutoModule("owner/repo", force_download=True)

# Offline mode — use only cached files
AgentClass = mf.AutoModule("owner/repo", local_files_only=True)

# Custom cache directory
AgentClass = mf.AutoModule("owner/repo", cache_dir="/path/to/cache")
```

## Checking Requirements

Inspect a module's requirements without loading or executing any code:

```python
info = mf.AutoModule.check_requirements("owner/repo")

print(info["config"].models)            # default model configs
print(info["dependencies"]["missing"])  # packages not installed
print(info["dependencies"]["available"])
print(info["version_ok"])              # True if version is compatible
```

## Local Cache

Downloaded files are stored at:

```
~/.cache/msgflux/auto/{source}/{owner}--{repo}/{revision}/
├── config.json
└── modeling.py
```

The `MSGFLUX_AUTO_CACHE_DIR` environment variable overrides the default cache path.

## Security

- **`sharing_mode: "class"`** — safe by default; no remote code runs until the user instantiates the class.
- **`sharing_mode: "instance"`** — the module file is fully executed at import time; requires `trust_remote_code=True`.
- Always pin `revision` in production to prevent unexpected updates from the repository author.
