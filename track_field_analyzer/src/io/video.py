"""Video and image loading utilities.

Handles Streamlit uploaded files by writing to temp files,
then reading via OpenCV. Includes frame sampling for performance.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator, Any

import cv2
import numpy as np


def load_video_from_uploaded_file(uploaded_file: Any) -> str:
    """
    Write Streamlit uploaded video file to a temporary location.
    
    Streamlit's UploadedFile object doesn't have a filesystem path,
    so we must write it to a temp file for OpenCV to read.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        str: Path to temporary video file
        
    Raises:
        ValueError: If file cannot be written
    """
    # Get file extension from original filename
    suffix = Path(uploaded_file.name).suffix.lower()
    
    # Create temp file with correct extension
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )
    
    try:
        # Write uploaded content to temp file
        temp_file.write(uploaded_file.getbuffer())
        temp_file.flush()
        temp_path = temp_file.name
    finally:
        temp_file.close()
    
    # Verify the file can be opened by OpenCV
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {uploaded_file.name}")
    cap.release()
    
    return temp_path


def load_image_from_uploaded_file(uploaded_file: Any) -> np.ndarray:
    """
    Load an image from Streamlit uploaded file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        np.ndarray: Image in RGB format
        
    Raises:
        ValueError: If image cannot be decoded
    """
    # Read file bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Decode image
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image_bgr is None:
        raise ValueError(f"Could not decode image: {uploaded_file.name}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    return image_rgb


def get_video_properties(video_path: str) -> dict:
    """
    Get video metadata properties.
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict with keys:
            - fps: Frames per second
            - frame_count: Total number of frames
            - width: Frame width in pixels
            - height: Frame height in pixels
            - duration_sec: Video duration in seconds
            
    Raises:
        ValueError: If video cannot be opened
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate duration
        duration_sec = frame_count / fps if fps > 0 else 0.0
        
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": duration_sec,
        }
    finally:
        cap.release()


def sample_frames(
    video_path: str,
    sample_rate: int = 5,
    max_frames: int | None = None
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Generator that yields sampled frames from a video.
    
    Memory-efficient: only one frame loaded at a time.
    
    Args:
        video_path: Path to video file
        sample_rate: Process every Nth frame (1 = all frames)
        max_frames: Optional maximum number of frames to yield
        
    Yields:
        tuple: (frame_index, frame_rgb as np.ndarray)
        
    Raises:
        ValueError: If video cannot be opened
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    try:
        frame_index = 0
        yielded_count = 0
        
        while True:
            ret, frame_bgr = cap.read()
            
            if not ret:
                break
            
            # Check if we should yield this frame
            if frame_index % sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield (frame_index, frame_rgb)
                yielded_count += 1
                
                # Check max frames limit
                if max_frames is not None and yielded_count >= max_frames:
                    break
            
            frame_index += 1
    finally:
        cap.release()


def cleanup_temp_file(temp_path: str) -> None:
    """
    Clean up a temporary file.
    
    Args:
        temp_path: Path to temporary file to delete
    """
    try:
        Path(temp_path).unlink(missing_ok=True)
    except Exception:
        pass  # Best effort cleanup
