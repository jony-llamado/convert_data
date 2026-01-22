"""LeRobot v2 format support for Forge.

LeRobot v2 uses HuggingFace datasets with video files.
"""

from forge.formats.lerobot_v2.reader import LeRobotV2Reader

__all__ = ["LeRobotV2Reader"]
