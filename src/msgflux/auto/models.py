"""Utility for loading model configs defined in AutoModule config.json."""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_model_configs(
    file: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load model configs from the config.json co-located with the module file.

    Reads the ``models`` section of ``config.json`` from the same directory as
    ``file``. When ``overrides`` are provided, matching keys replace the
    values from config.json; unmatched keys from both dicts are preserved.

    Intended to be called inside a ``Module.__init__`` so the developer does
    not need to hard-code model IDs and the user can override them at
    instantiation time.

    Args:
        file: Path to the calling module file. Pass ``__file__``.
        overrides: Optional dict whose keys override entries in
            ``config.json["models"]``. Partial overrides are supported —
            only the supplied keys are replaced.

    Returns:
        Merged dict of model configs. Empty dict if ``config.json`` has no
        ``models`` key and no ``overrides`` were given.

    Example:
        In ``modeling.py``::

            from msgflux.auto import load_model_configs

            class MyAgent(Module):
                def __init__(self, models=None):
                    super().__init__()
                    cfg = load_model_configs(__file__, overrides=models)
                    self.lm = LM(cfg["lm"])

        User-side override::

            AgentClass = mf.AutoModule("owner/repo")
            agent = AgentClass(models={"lm": "groq/llama-3.1-8b"})
    """
    config_path = Path(file).parent / "config.json"

    models: Dict[str, Any] = {}
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            models = data.get("models", {})
        except (json.JSONDecodeError, OSError):
            pass

    if overrides:
        models = {**models, **overrides}

    return models


__all__ = ["load_model_configs"]
