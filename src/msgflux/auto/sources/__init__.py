"""Source implementations for AutoModule."""

from msgflux.auto.sources.base import Source
from msgflux.auto.sources.github import GitHubSource
from msgflux.auto.sources.huggingface import HuggingFaceSource

__all__ = ["Source", "GitHubSource", "HuggingFaceSource"]
