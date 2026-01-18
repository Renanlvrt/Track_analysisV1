"""IO module for video and image loading."""

from .video import (
    load_video_from_uploaded_file,
    load_image_from_uploaded_file,
    get_video_properties,
    sample_frames,
)

__all__ = [
    "load_video_from_uploaded_file",
    "load_image_from_uploaded_file", 
    "get_video_properties",
    "sample_frames",
]
