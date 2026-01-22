"""Zarr format support for Forge.

Zarr is commonly used for robotics datasets like UMI (Universal Manipulation Interface).
It stores data as chunked, compressed arrays with episode boundaries in metadata.
"""

from forge.formats.zarr.reader import ZarrReader

__all__ = ["ZarrReader"]
