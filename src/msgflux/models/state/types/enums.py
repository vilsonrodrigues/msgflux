"""Enums for message roles and lifecycle types."""

from enum import Enum


class Role(str, Enum):
    """Message roles - universal across all providers."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LifecycleType(str, Enum):
    """Lifecycle types for message retention policies."""

    PERMANENT = "permanent"
    EPHEMERAL_TURNS = "ephemeral_turns"
    EPHEMERAL_SCOPE = "ephemeral_scope"
    SUMMARIZABLE = "summarizable"
