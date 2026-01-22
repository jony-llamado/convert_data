"""Core domain models and protocols for Forge."""

from forge.core.exceptions import (
    ForgeError,
    FormatDetectionError,
    InspectionError,
    UnsupportedFormatError,
    ValidationError,
)
from forge.core.models import (
    CameraInfo,
    DatasetInfo,
    Dtype,
    Episode,
    FieldSchema,
    Frame,
    LazyArray,
    LazyImage,
)
from forge.core.protocols import FormatReader, FormatWriter, VideoDecoder, VideoEncoder

__all__ = [
    # Models
    "CameraInfo",
    "DatasetInfo",
    "Dtype",
    "Episode",
    "FieldSchema",
    "Frame",
    "LazyArray",
    "LazyImage",
    # Protocols
    "FormatReader",
    "FormatWriter",
    "VideoDecoder",
    "VideoEncoder",
    # Exceptions
    "ForgeError",
    "FormatDetectionError",
    "InspectionError",
    "UnsupportedFormatError",
    "ValidationError",
]
