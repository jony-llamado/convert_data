"""Dataset inspection module for Forge.

This module provides tools for analyzing robotics datasets without full loading,
including schema inference, statistics collection, and format detection.
"""

from forge.inspect.inspector import InspectionOptions, Inspector
from forge.inspect.schema_analyzer import SchemaAnalyzer
from forge.inspect.stats_collector import StatsCollector

__all__ = [
    "Inspector",
    "InspectionOptions",
    "SchemaAnalyzer",
    "StatsCollector",
]
