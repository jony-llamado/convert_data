"""HDF5 format support for Forge.

This module provides reading for HDF5-based robotics datasets,
commonly used by robomimic, ACT/ALOHA, and Mobile ALOHA.
"""

from forge.formats.hdf5.reader import HDF5Reader

__all__ = ["HDF5Reader"]
