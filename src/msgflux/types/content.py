"""Content block types for messages.

Basic content types without dependencies - used across the codebase.
"""

from typing import Optional, Union

import msgspec


class TextContent(msgspec.Struct, tag="text", tag_field="content_type", kw_only=True):
    """Plain text content block."""

    text: str

    @property
    def type(self) -> str:
        return "text"


class ImageContent(msgspec.Struct, tag="image", tag_field="content_type", kw_only=True):
    """Image content block - supports URL or base64."""

    url: Optional[str] = None
    base64: Optional[str] = None
    media_type: Optional[str] = None
    detail: Optional[str] = None

    @property
    def type(self) -> str:
        return "image"


class AudioContent(msgspec.Struct, tag="audio", tag_field="content_type", kw_only=True):
    """Audio content block."""

    url: Optional[str] = None
    base64: Optional[str] = None
    format: Optional[str] = None

    @property
    def type(self) -> str:
        return "audio"


class VideoContent(msgspec.Struct, tag="video", tag_field="content_type", kw_only=True):
    """Video content block."""

    url: Optional[str] = None
    base64: Optional[str] = None
    media_type: Optional[str] = None

    @property
    def type(self) -> str:
        return "video"


class FileContent(msgspec.Struct, tag="file", tag_field="content_type", kw_only=True):
    """File content block."""

    filename: str
    data: str
    media_type: Optional[str] = None

    @property
    def type(self) -> str:
        return "file"


ContentBlock = Union[TextContent, ImageContent, AudioContent, VideoContent, FileContent]
