"""Rosbag format support for Forge.

Supports both ROS1 (.bag) and ROS2 (MCAP, SQLite3) bag formats.
Uses the rosbags library for pure Python reading without ROS dependencies.
"""

from forge.formats.rosbag.reader import RosbagReader

__all__ = ["RosbagReader"]
