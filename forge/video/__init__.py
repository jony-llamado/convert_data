"""Video encoding/decoding module for Forge.

Provides PyAV-based video encoding for converting robotics image
sequences to MP4 format.
"""

from forge.video.encoder import VideoEncoder, VideoEncoderConfig, encode_video

__all__ = ["VideoEncoder", "VideoEncoderConfig", "encode_video"]
