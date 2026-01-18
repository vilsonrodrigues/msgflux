# AutoModule

Load `nn.Module` from remote repositories (GitHub or Hugging Face Hub).

## Quick Start

```python
import msgflux as mf

# Load class (sharing_mode: class)
AgentClass = mf.AutoModule("owner/repo")
agent = AgentClass()

# Load pre-instantiated object (sharing_mode: instance)
agent = mf.AutoModule("owner/repo", trust_remote_code=True)
```

## Creating a Shareable Module

### Repository Structure

```
your-repo/
├── config.json
├── modeling.py
└── README.md (optional)
```

### config.json

**For class sharing:**
```json
{
  "msgflux_class": "MyAgent",
  "msgflux_entrypoint": "modeling.py",
  "msgflux_version": ">=0.1.0",
  "sharing_mode": "class"
}
```

**For instance sharing:**
```json
{
  "msgflux_class": "MyAgent",
  "msgflux_entrypoint": "modeling.py:agent",
  "msgflux_version": ">=0.1.0",
  "sharing_mode": "instance"
}
```

### modeling.py

**Class mode:**
```python
from msgflux.nn import Module

class MyAgent(Module):
    def __init__(self, name: str = "default"):
        super().__init__()
        self.name = name

    def forward(self, input: str) -> str:
        return f"[{self.name}] {input}"
```

**Instance mode** (object created in file):
```python
from msgflux.nn import Module

class MyAgent(Module):
    def __init__(self, name: str = "default"):
        super().__init__()
        self.name = name

    def forward(self, input: str) -> str:
        return f"[{self.name}] {input}"

# Pre-instantiated object
agent = MyAgent(name="ProductionAgent")
# Can load state_dict here if needed
```

## Loading Options

```python
# Pin revision (recommended)
Agent = mf.AutoModule("owner/repo", revision="abc123")

# Explicit source
Agent = mf.AutoModule("hf://owner/repo")  # Hugging Face
Agent = mf.AutoModule("gh://owner/repo")  # GitHub

# Force re-download
Agent = mf.AutoModule("owner/repo", force_download=True)

# Offline mode
Agent = mf.AutoModule("owner/repo", local_files_only=True)
```

## Security

- `sharing_mode: "class"` - Safe, user controls instantiation
- `sharing_mode: "instance"` - Requires `trust_remote_code=True`

Always pin `revision` for reproducibility and security.
