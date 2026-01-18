"""Multimodal data types for msgflux.

This module provides data types for handling multimodal content (images, audio,
video, files) with support for both sync and async processing.

Usage:
    >>> from msgflux import Image, Audio, Video, File
    >>> # or
    >>> from msgflux.data.types import Image, Audio, Video, File

    # Create and process
    >>> img = Image("/path/to/photo.jpg", detail="high")
    >>> block = img()  # sync
    >>> block = await img.acall()  # async
"""

from msgflux.data.types.audio import Audio
from msgflux.data.types.base import MediaType
from msgflux.data.types.file import File
from msgflux.data.types.image import Image
from msgflux.data.types.video import Video

__all__ = [
    "MediaType",
    "Image",
    "Audio",
    "Video",
    "File",
]
