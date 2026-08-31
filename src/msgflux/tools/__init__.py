"""Tools module for msgflux.

This module provides tool-related functionality including
ToolFlowControl for managing tool execution flow.
"""

from importlib import import_module

from msgflux.generation.control_flow import ToolFlowControl
from msgflux.tools.dataclasses import ToolMetadata
from msgflux.tools.definitions import ToolCatalog, ToolSpec
from msgflux.tools.guidance import BUILTIN_TOOL_USAGE_GUIDANCE, apply_tool_guidance
from msgflux.tools.handles import ToolBucketHandle, ToolLibraryHandle
from msgflux.tools.runtime import FeedbackSpec, ToolError, ToolIntent, ToolOutcome
from msgflux.tools.types import (
    Hidden,
    ToolBackground,
    ToolBucket,
    ToolLibraryOperator,
)

__all__ = [
    "BUILTIN_TOOL_USAGE_GUIDANCE",
    "FeedbackSpec",
    "Hidden",
    "ToolBackground",
    "ToolCatalog",
    "ToolCatalogEntry",
    "ToolCatalogView",
    "ToolChoice",
    "ToolError",
    "ToolIntent",
    "ToolOutcome",
    "ToolRef",
    "ToolSpec",
    "ToolBucket",
    "ToolBucketHandle",
    "ToolFlowControl",
    "ToolLibraryHandle",
    "ToolLibraryOperator",
    "ToolMetadata",
    "apply_tool_guidance",
]


def __getattr__(name: str):
    if name in {"ToolCatalogEntry", "ToolCatalogView", "ToolChoice", "ToolRef"}:
        value = getattr(import_module("msgflux.nn.modules.tool_runtime"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
