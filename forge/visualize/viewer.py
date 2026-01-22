"""Dataset visualizer for Forge.

A visualization tool supporting LeRobot v3, Zarr, and RLDS datasets using matplotlib.
Displays camera feeds and observation/action data in an interactive window.
Supports side-by-side comparison mode for original vs converted datasets.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.exceptions import MissingDependencyError


def _check_matplotlib() -> tuple[Any, Any, Any]:
    """Check and import matplotlib dependencies."""
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button
    except ImportError:
        raise MissingDependencyError(
            dependency="matplotlib",
            feature="Dataset visualization",
            install_hint="pip install matplotlib",
        )
    return plt, Slider, Button


def _check_av() -> Any:
    """Check and import PyAV."""
    try:
        import av
        return av
    except ImportError:
        raise MissingDependencyError(
            dependency="av",
            feature="Video frame extraction",
            install_hint="pip install av",
        )


def _check_pyarrow() -> Any:
    """Check and import PyArrow."""
    try:
        import pyarrow.parquet as pq
        return pq
    except ImportError:
        raise MissingDependencyError(
            dependency="pyarrow",
            feature="Parquet reading",
            install_hint="pip install pyarrow",
        )


def _check_zarr() -> Any:
    """Check and import Zarr."""
    try:
        import zarr
        return zarr
    except ImportError:
        raise MissingDependencyError(
            dependency="zarr",
            feature="Zarr dataset support",
            install_hint="pip install zarr",
        )


class DatasetBackend(ABC):
    """Abstract backend for loading dataset data."""

    @abstractmethod
    def get_num_episodes(self) -> int:
        """Get total number of episodes."""
        pass

    @abstractmethod
    def get_episode_length(self, episode_idx: int) -> int:
        """Get number of frames in an episode."""
        pass

    @abstractmethod
    def get_frame_image(self, episode_idx: int, frame_idx: int, camera_key: str) -> np.ndarray | None:
        """Get image frame as numpy array (H, W, C)."""
        pass

    @abstractmethod
    def get_frame_data(self, episode_idx: int, frame_idx: int, feature_key: str) -> np.ndarray | None:
        """Get numeric feature data for a frame."""
        pass

    @abstractmethod
    def get_episode_data(self, episode_idx: int, feature_key: str) -> np.ndarray | None:
        """Get all data for a feature in an episode."""
        pass

    @abstractmethod
    def get_camera_keys(self) -> list[str]:
        """Get list of camera/image feature keys."""
        pass

    @abstractmethod
    def get_numeric_keys(self) -> list[str]:
        """Get list of numeric feature keys."""
        pass

    @abstractmethod
    def get_image_shape(self, camera_key: str) -> tuple[int, int, int]:
        """Get shape (H, W, C) for a camera."""
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """Get dataset FPS."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get dataset name for display."""
        pass

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class LeRobotV3Backend(DatasetBackend):
    """Backend for LeRobot v3 datasets."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.pq = _check_pyarrow()
        self.av = _check_av()

        # Load info
        info_path = self.dataset_path / "meta" / "info.json"
        with open(info_path) as f:
            self.info = json.load(f)

        # Load data
        self.df = self._load_data()

        # Video readers cache
        self.video_readers: dict[str, Any] = {}
        self.video_streams: dict[str, Any] = {}
        self.video_frame_cache: dict[str, dict[int, np.ndarray]] = {}  # cache_key -> {frame_idx -> frame}
        self.video_last_decoded: dict[str, int] = {}  # cache_key -> last decoded frame index

        # Episode boundaries
        self._episode_starts: dict[int, int] = {}
        self._episode_lengths: dict[int, int] = {}
        for ep_idx in self.df["episode_index"].unique():
            ep_data = self.df[self.df["episode_index"] == ep_idx]
            self._episode_starts[ep_idx] = ep_data["index"].min()
            self._episode_lengths[ep_idx] = len(ep_data)

    def _load_data(self) -> Any:
        """Load parquet data."""
        import pyarrow as pa

        data_dir = self.dataset_path / "data"
        parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))

        tables = []
        for pf in parquet_files:
            tables.append(self.pq.read_table(pf))

        combined = pa.concat_tables(tables)
        return combined.to_pandas()

    def get_num_episodes(self) -> int:
        return len(self._episode_lengths)

    def get_episode_length(self, episode_idx: int) -> int:
        return self._episode_lengths.get(episode_idx, 0)

    def get_frame_image(self, episode_idx: int, frame_idx: int, camera_key: str) -> np.ndarray | None:
        # Calculate global index
        global_index = self._episode_starts.get(episode_idx, 0) + frame_idx

        # Determine chunk
        chunks_size = self.info.get("chunks_size", 1000)
        chunk_index = episode_idx // chunks_size

        video_path = (
            self.dataset_path / "videos" / camera_key /
            f"chunk-{chunk_index:03d}" / "file-000.mp4"
        )

        if not video_path.exists():
            return None

        # Calculate video frame offset
        chunk_start_ep = chunk_index * chunks_size
        chunk_mask = self.df["episode_index"] >= chunk_start_ep
        chunk_mask &= self.df["episode_index"] < (chunk_index + 1) * chunks_size
        chunk_df = self.df[chunk_mask]

        if chunk_df.empty:
            return None

        chunk_start_index = chunk_df["index"].min()
        video_frame_idx = global_index - chunk_start_index

        # Open video
        cache_key = str(video_path)
        if cache_key not in self.video_readers:
            try:
                container = self.av.open(str(video_path))
                self.video_readers[cache_key] = container
                self.video_streams[cache_key] = container.streams.video[0]
                self.video_frame_cache[cache_key] = {}
                self.video_last_decoded[cache_key] = -1
            except Exception:
                return None

        # Check cache first
        if cache_key in self.video_frame_cache:
            if video_frame_idx in self.video_frame_cache[cache_key]:
                return self.video_frame_cache[cache_key][video_frame_idx]

        container = self.video_readers[cache_key]
        last_decoded = self.video_last_decoded.get(cache_key, -1)

        try:
            # If we need to go backwards or jump far, seek to beginning
            if video_frame_idx < last_decoded or video_frame_idx > last_decoded + 100:
                container.seek(0)
                last_decoded = -1
                # Clear cache for this video to save memory
                self.video_frame_cache[cache_key] = {}

            # Decode frames sequentially until we reach target
            for frame in container.decode(video=0):
                frame_idx = frame.pts // 1024 if frame.pts else last_decoded + 1  # pts increments by 1024
                frame_array = frame.to_ndarray(format="rgb24")

                # Cache the frame
                self.video_frame_cache[cache_key][frame_idx] = frame_array
                self.video_last_decoded[cache_key] = frame_idx

                if frame_idx >= video_frame_idx:
                    return frame_array

                # Limit cache size to ~200 frames per video
                if len(self.video_frame_cache[cache_key]) > 200:
                    oldest = min(self.video_frame_cache[cache_key].keys())
                    del self.video_frame_cache[cache_key][oldest]

        except Exception:
            pass

        return None

    def get_frame_data(self, episode_idx: int, frame_idx: int, feature_key: str) -> np.ndarray | None:
        ep_data = self.df[self.df["episode_index"] == episode_idx]
        if frame_idx >= len(ep_data) or feature_key not in ep_data.columns:
            return None
        val = ep_data.iloc[frame_idx][feature_key]
        if isinstance(val, (list, np.ndarray)):
            return np.array(val)
        return np.array([val])

    def get_episode_data(self, episode_idx: int, feature_key: str) -> np.ndarray | None:
        ep_data = self.df[self.df["episode_index"] == episode_idx]
        if feature_key not in ep_data.columns:
            return None
        values = ep_data[feature_key].tolist()
        return np.array(values)

    def get_camera_keys(self) -> list[str]:
        return [
            name for name, feat in self.info.get("features", {}).items()
            if feat.get("dtype") == "video"
        ]

    def get_numeric_keys(self) -> list[str]:
        excluded = {"episode_index", "frame_index", "index", "task_index", "timestamp"}
        all_keys = [
            name for name, feat in self.info.get("features", {}).items()
            if feat.get("dtype") in ("float32", "float64", "int32", "int64")
            and name not in excluded
        ]
        # Prioritize action and state features for consistent comparison
        priority_keys = []
        other_keys = []
        for key in all_keys:
            key_lower = key.lower()
            if 'action' in key_lower or 'state' in key_lower or 'obs' in key_lower:
                priority_keys.append(key)
            else:
                other_keys.append(key)
        return priority_keys + other_keys

    def get_image_shape(self, camera_key: str) -> tuple[int, int, int]:
        feat = self.info.get("features", {}).get(camera_key, {})
        shape = feat.get("shape", [96, 96, 3])
        return tuple(shape)

    def get_fps(self) -> float:
        return self.info.get("fps", 10)

    def get_name(self) -> str:
        return f"{self.dataset_path.name} (LeRobot v3)"

    def cleanup(self) -> None:
        for container in self.video_readers.values():
            container.close()


class ZarrBackend(DatasetBackend):
    """Backend for Zarr datasets."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        zarr = _check_zarr()

        self.root = zarr.open(str(dataset_path), 'r')

        # Detect structure
        if 'data' in self.root and 'meta' in self.root:
            self.data = self.root['data']
            self.meta = self.root['meta']
        else:
            self.data = self.root
            self.meta = None

        # Get episode boundaries
        if self.meta and 'episode_ends' in self.meta:
            self.episode_ends = self.meta['episode_ends'][:]
        else:
            # Assume single episode
            first_key = list(self.data.keys())[0]
            self.episode_ends = np.array([len(self.data[first_key])])

        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])

        # Detect features
        self._camera_keys: list[str] = []
        self._numeric_keys: list[str] = []
        self._image_shapes: dict[str, tuple[int, int, int]] = {}

        for key in self.data.keys():
            arr = self.data[key]
            shape = arr.shape

            # Image detection: shape (N, H, W, C) where C is 1, 3, or 4
            if len(shape) == 4 and shape[-1] in (1, 3, 4):
                self._camera_keys.append(key)
                self._image_shapes[key] = (shape[1], shape[2], shape[3])
            elif key.lower() in ('img', 'image', 'images', 'rgb', 'observation_image'):
                self._camera_keys.append(key)
                if len(shape) == 4:
                    self._image_shapes[key] = (shape[1], shape[2], shape[3])
                else:
                    self._image_shapes[key] = (96, 96, 3)
            elif len(shape) >= 1:
                self._numeric_keys.append(key)

    def get_num_episodes(self) -> int:
        return len(self.episode_ends)

    def get_episode_length(self, episode_idx: int) -> int:
        if episode_idx >= len(self.episode_ends):
            return 0
        start = self.episode_starts[episode_idx]
        end = self.episode_ends[episode_idx]
        return int(end - start)

    def get_frame_image(self, episode_idx: int, frame_idx: int, camera_key: str) -> np.ndarray | None:
        if camera_key not in self._camera_keys:
            return None

        global_idx = int(self.episode_starts[episode_idx] + frame_idx)
        arr = self.data[camera_key]

        if global_idx >= len(arr):
            return None

        img = arr[global_idx]

        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def get_frame_data(self, episode_idx: int, frame_idx: int, feature_key: str) -> np.ndarray | None:
        if feature_key not in self.data:
            return None

        global_idx = int(self.episode_starts[episode_idx] + frame_idx)
        arr = self.data[feature_key]

        if global_idx >= len(arr):
            return None

        return np.array(arr[global_idx])

    def get_episode_data(self, episode_idx: int, feature_key: str) -> np.ndarray | None:
        if feature_key not in self.data:
            return None

        start = int(self.episode_starts[episode_idx])
        end = int(self.episode_ends[episode_idx])

        return np.array(self.data[feature_key][start:end])

    def get_camera_keys(self) -> list[str]:
        return self._camera_keys

    def get_numeric_keys(self) -> list[str]:
        # Prioritize action and state features for consistent comparison
        priority_keys = []
        other_keys = []
        for key in self._numeric_keys:
            key_lower = key.lower()
            if 'action' in key_lower or 'state' in key_lower or 'obs' in key_lower:
                priority_keys.append(key)
            else:
                other_keys.append(key)
        return priority_keys + other_keys

    def get_image_shape(self, camera_key: str) -> tuple[int, int, int]:
        return self._image_shapes.get(camera_key, (96, 96, 3))

    def get_fps(self) -> float:
        return 10  # Default for zarr

    def get_name(self) -> str:
        return f"{self.dataset_path.name} (Zarr)"


def _check_tensorflow() -> Any:
    """Check and import TensorFlow."""
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        return tf
    except ImportError:
        raise MissingDependencyError(
            dependency="tensorflow",
            feature="RLDS dataset support",
            install_hint="pip install tensorflow tensorflow-datasets",
        )


def _check_tfds() -> Any:
    """Check and import tensorflow_datasets."""
    try:
        import tensorflow_datasets as tfds
        return tfds
    except ImportError:
        raise MissingDependencyError(
            dependency="tensorflow-datasets",
            feature="RLDS dataset support",
            install_hint="pip install tensorflow-datasets",
        )


class RLDSBackend(DatasetBackend):
    """Backend for RLDS/TFDS datasets."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.tf = _check_tensorflow()
        self.tfds = _check_tfds()

        # Determine data_dir and dataset_name
        data_dir, dataset_name = self._resolve_dataset_path()

        # Load dataset
        ds = self.tfds.load(dataset_name, data_dir=str(data_dir))
        if isinstance(ds, dict):
            ds = ds.get("train", list(ds.values())[0])

        # Pre-load episodes (RLDS requires iteration)
        self._episodes: list[dict] = []
        self._camera_keys: list[str] = []
        self._numeric_keys: list[str] = []
        self._image_shapes: dict[str, tuple[int, int, int]] = {}
        self._episode_lengths: list[int] = []

        # Load first few episodes for visualization
        max_episodes = 50  # Limit for memory
        for ep_data in ds.take(max_episodes):
            steps = ep_data.get("steps", ep_data)
            frames = list(steps)
            self._episodes.append(frames)
            self._episode_lengths.append(len(frames))

            # Detect features from first episode
            if len(self._episodes) == 1 and frames:
                self._detect_features(frames[0])

    def _resolve_dataset_path(self) -> tuple[Path, str]:
        """Resolve TFDS data_dir and dataset_name from path."""
        path = self.dataset_path

        # TFDS datasets have structure: data_dir/dataset_name/config/version/
        # We need to find the data_dir and construct dataset_name/config

        # Check if current dir has version subdirs with dataset_info.json
        for subdir in path.iterdir():
            if subdir.is_dir():
                # Check if subdir is a version (e.g., 1.0.1)
                if (subdir / "dataset_info.json").exists():
                    # Structure: data_dir/dataset/config/version
                    # path is config level, parent is dataset, grandparent is data_dir
                    return path.parent.parent, f"{path.parent.name}/{path.name}"

        # Check if parent dir is a config (structure: data_dir/dataset_name/config)
        if (path.parent / "dataset_info.json").exists():
            data_dir = path.parent.parent.parent
            dataset_name = f"{path.parent.parent.name}/{path.parent.name}"
            return data_dir, dataset_name

        # Check direct subdirectories for version dirs
        for subdir in path.iterdir():
            if subdir.is_dir():
                for version_dir in subdir.iterdir():
                    if version_dir.is_dir() and (version_dir / "dataset_info.json").exists():
                        # Structure: path/config/version/
                        return path.parent, f"{path.name}/{subdir.name}"

        # Default: assume path is dataset name at top level
        if (path / "dataset_info.json").exists():
            return path.parent, path.name

        return path.parent, path.name

    def _detect_features(self, step: dict) -> None:
        """Detect camera and numeric features from a step."""
        obs = step.get("observation", step)

        if isinstance(obs, dict):
            for key, value in obs.items():
                if hasattr(value, "shape"):
                    shape = value.shape
                    # Image detection: 3D shape with last dim in [1, 3, 4]
                    if len(shape) == 3 and shape[-1] in (1, 3, 4):
                        self._camera_keys.append(key)
                        self._image_shapes[key] = (int(shape[0]), int(shape[1]), int(shape[2]))
                    elif self._looks_like_image(key):
                        self._camera_keys.append(key)
                        if len(shape) == 3:
                            self._image_shapes[key] = (int(shape[0]), int(shape[1]), int(shape[2]))
                        else:
                            self._image_shapes[key] = (480, 640, 3)
                    else:
                        self._numeric_keys.append(key)

        # Check for action
        if "action" in step and hasattr(step["action"], "shape"):
            self._numeric_keys.append("action")

    def _looks_like_image(self, key: str) -> bool:
        """Check if key looks like an image field."""
        key_lower = key.lower()
        return any(x in key_lower for x in ["image", "rgb", "camera", "img", "frame"])

    def get_num_episodes(self) -> int:
        return len(self._episodes)

    def get_episode_length(self, episode_idx: int) -> int:
        if episode_idx >= len(self._episode_lengths):
            return 0
        return self._episode_lengths[episode_idx]

    def get_frame_image(self, episode_idx: int, frame_idx: int, camera_key: str) -> np.ndarray | None:
        if episode_idx >= len(self._episodes):
            return None
        frames = self._episodes[episode_idx]
        if frame_idx >= len(frames):
            return None

        step = frames[frame_idx]
        obs = step.get("observation", step)

        if isinstance(obs, dict) and camera_key in obs:
            img = obs[camera_key].numpy() if hasattr(obs[camera_key], "numpy") else np.array(obs[camera_key])

            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)

            return img

        return None

    def get_frame_data(self, episode_idx: int, frame_idx: int, feature_key: str) -> np.ndarray | None:
        if episode_idx >= len(self._episodes):
            return None
        frames = self._episodes[episode_idx]
        if frame_idx >= len(frames):
            return None

        step = frames[frame_idx]

        # Check action first
        if feature_key == "action" and "action" in step:
            return step["action"].numpy() if hasattr(step["action"], "numpy") else np.array(step["action"])

        # Check observation
        obs = step.get("observation", step)
        if isinstance(obs, dict) and feature_key in obs:
            val = obs[feature_key]
            return val.numpy() if hasattr(val, "numpy") else np.array(val)

        return None

    def get_episode_data(self, episode_idx: int, feature_key: str) -> np.ndarray | None:
        if episode_idx >= len(self._episodes):
            return None
        frames = self._episodes[episode_idx]

        data = []
        for frame_idx in range(len(frames)):
            val = self.get_frame_data(episode_idx, frame_idx, feature_key)
            if val is not None:
                data.append(val)

        if data:
            return np.array(data)
        return None

    def get_camera_keys(self) -> list[str]:
        return self._camera_keys

    def get_numeric_keys(self) -> list[str]:
        # Prioritize action and state features
        priority_keys = []
        other_keys = []
        for key in self._numeric_keys:
            key_lower = key.lower()
            if 'action' in key_lower or 'state' in key_lower or 'obs' in key_lower:
                priority_keys.append(key)
            else:
                other_keys.append(key)
        return priority_keys + other_keys

    def get_image_shape(self, camera_key: str) -> tuple[int, int, int]:
        return self._image_shapes.get(camera_key, (480, 640, 3))

    def get_fps(self) -> float:
        return 10  # Default for RLDS

    def get_name(self) -> str:
        return f"{self.dataset_path.name} (RLDS)"


def detect_format(path: Path) -> str:
    """Detect dataset format from path."""
    path = Path(path)

    # LeRobot v3
    if (path / "meta" / "info.json").exists():
        return "lerobot-v3"

    # Zarr
    if path.suffix == ".zarr" or (path / ".zarray").exists() or (path / ".zgroup").exists():
        return "zarr"

    # Check for zarr structure inside
    if path.is_dir():
        for subdir in path.iterdir():
            if subdir.suffix == ".zarr":
                return "zarr"
            if (subdir / ".zarray").exists() or (subdir / ".zgroup").exists():
                return "zarr"

    # RLDS (TFRecord files or dataset_info.json)
    if path.is_dir():
        # Check for tfrecord files
        if any(path.glob("*.tfrecord*")):
            return "rlds"
        # Check for dataset_info.json (TFDS format)
        if (path / "dataset_info.json").exists():
            return "rlds"
        # Check subdirectories (version dirs)
        for subdir in path.iterdir():
            if subdir.is_dir():
                if any(subdir.glob("*.tfrecord*")):
                    return "rlds"
                if (subdir / "dataset_info.json").exists():
                    return "rlds"

    raise ValueError(f"Unknown dataset format at {path}")


def create_backend(path: Path) -> DatasetBackend:
    """Create appropriate backend for dataset."""
    fmt = detect_format(path)

    if fmt == "lerobot-v3":
        return LeRobotV3Backend(path)
    elif fmt == "zarr":
        return ZarrBackend(path)
    elif fmt == "rlds":
        return RLDSBackend(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


class DatasetViewer:
    """Interactive viewer for robotics datasets.

    Supports LeRobot v3, Zarr, and RLDS formats, with optional side-by-side comparison.

    Example:
        >>> viewer = DatasetViewer("path/to/dataset")
        >>> viewer.show()

        # Comparison mode
        >>> viewer = DatasetViewer("original.zarr", "converted_lerobot_v3")
        >>> viewer.show()
    """

    def __init__(
        self,
        dataset_path: str | Path,
        compare_path: str | Path | None = None,
    ):
        """Initialize viewer.

        Args:
            dataset_path: Path to primary dataset.
            compare_path: Optional path to second dataset for comparison.
        """
        self.plt, self.Slider, self.Button = _check_matplotlib()

        self.backend = create_backend(Path(dataset_path))
        self.compare_backend = create_backend(Path(compare_path)) if compare_path else None
        self.comparison_mode = compare_path is not None

        # State
        self.current_episode = 0
        self.current_frame = 0
        self.playing = False
        self.fps = self.backend.get_fps()

    def show(self) -> None:
        """Display the interactive visualization window."""
        plt = self.plt

        backends = [self.backend]
        if self.compare_backend:
            backends.append(self.compare_backend)

        # Determine layout
        n_backends = len(backends)
        camera_keys = self.backend.get_camera_keys()
        numeric_keys = self.backend.get_numeric_keys()[:2]

        n_cameras = len(camera_keys)
        n_plots = len(numeric_keys)

        if n_cameras == 0 and n_plots == 0:
            print("No cameras or numeric features to display")
            return

        # Layout: rows for cameras + plots, columns for backends (comparison)
        n_rows = (1 if n_cameras > 0 else 0) + (1 if n_plots > 0 else 0)
        n_cols = max(n_cameras, n_plots) * n_backends

        fig_width = 4 * max(n_cameras, n_plots) * n_backends
        fig_height = 4 * n_rows

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        title = "Forge Viewer"
        if self.comparison_mode:
            title += " - Comparison Mode"
        fig.canvas.manager.set_window_title(title)

        # Normalize axes to 2D array
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Store display elements
        self.camera_displays: list[list[tuple[Any, Any, str]]] = []  # Per backend
        self.plot_displays: list[list[tuple[Any, list, Any, str]]] = []  # Per backend

        row_idx = 0

        # Setup camera displays
        if n_cameras > 0:
            for b_idx, backend in enumerate(backends):
                backend_cameras = []
                b_camera_keys = backend.get_camera_keys()

                for c_idx, cam_key in enumerate(b_camera_keys):
                    col_idx = b_idx * max(n_cameras, 1) + c_idx
                    if col_idx >= n_cols:
                        break

                    ax = axes[row_idx, col_idx]
                    short_name = cam_key.replace("observation.images.", "")
                    ax.set_title(f"{backend.get_name()}\n{short_name}", fontsize=9)
                    ax.axis("off")

                    shape = backend.get_image_shape(cam_key)
                    placeholder = np.zeros((shape[0], shape[1], shape[2]), dtype=np.uint8)
                    im = ax.imshow(placeholder)
                    backend_cameras.append((ax, im, cam_key))

                self.camera_displays.append(backend_cameras)

            # Hide unused camera axes
            for col_idx in range(len(backends) * n_cameras, n_cols):
                axes[row_idx, col_idx].axis("off")

            row_idx += 1

        # Setup plot displays
        if n_plots > 0 and row_idx < n_rows:
            for b_idx, backend in enumerate(backends):
                backend_plots = []
                b_numeric_keys = backend.get_numeric_keys()[:2]

                for p_idx, feat_key in enumerate(b_numeric_keys):
                    col_idx = b_idx * max(n_plots, 1) + p_idx
                    if col_idx >= n_cols:
                        break

                    ax = axes[row_idx, col_idx]
                    ax.set_title(f"{backend.get_name()}\n{feat_key}", fontsize=9)
                    ax.set_xlabel("Frame")

                    # Initial plot
                    ep_data = backend.get_episode_data(self.current_episode, feat_key)
                    lines = []
                    if ep_data is not None:
                        # Flatten if more than 2D
                        if len(ep_data.shape) > 2:
                            ep_data = ep_data.reshape(ep_data.shape[0], -1)

                        if len(ep_data.shape) > 1:
                            for dim in range(min(ep_data.shape[1], 7)):
                                line, = ax.plot(ep_data[:, dim], alpha=0.7)
                                lines.append(line)
                        else:
                            line, = ax.plot(ep_data)
                            lines.append(line)

                    marker = ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)
                    backend_plots.append((ax, lines, marker, feat_key))

                self.plot_displays.append(backend_plots)

            # Hide unused plot axes
            for col_idx in range(len(backends) * n_plots, n_cols):
                if col_idx < axes.shape[1]:
                    axes[row_idx, col_idx].axis("off")

        # Controls
        plt.subplots_adjust(bottom=0.2)

        num_episodes = self.backend.get_num_episodes()

        ax_episode = plt.axes([0.15, 0.10, 0.7, 0.03])
        self.episode_slider = self.Slider(
            ax_episode, "Episode",
            0, max(num_episodes - 1, 1),
            valinit=0, valstep=1
        )
        self.episode_slider.on_changed(self._on_episode_change)

        ep_len = self.backend.get_episode_length(0)
        ax_frame = plt.axes([0.15, 0.05, 0.7, 0.03])
        self.frame_slider = self.Slider(
            ax_frame, "Frame",
            0, max(ep_len - 1, 1),
            valinit=0, valstep=1
        )
        self.frame_slider.on_changed(self._on_frame_change)

        ax_play = plt.axes([0.4, 0.01, 0.08, 0.03])
        self.play_button = self.Button(ax_play, "Play")
        self.play_button.on_clicked(self._on_play_click)

        ax_prev = plt.axes([0.28, 0.01, 0.08, 0.03])
        self.prev_button = self.Button(ax_prev, "< Prev")
        self.prev_button.on_clicked(self._on_prev_click)

        ax_next = plt.axes([0.52, 0.01, 0.08, 0.03])
        self.next_button = self.Button(ax_next, "Next >")
        self.next_button.on_clicked(self._on_next_click)

        # Initial update
        self._update_display()

        # Timer
        self.timer = fig.canvas.new_timer(interval=int(1000 / self.fps))
        self.timer.add_callback(self._on_timer)

        plt.show()

        # Cleanup
        self.backend.cleanup()
        if self.compare_backend:
            self.compare_backend.cleanup()

    def _on_episode_change(self, val: float) -> None:
        self.current_episode = int(val)
        self.current_frame = 0

        ep_len = self.backend.get_episode_length(self.current_episode)
        self.frame_slider.valmax = max(ep_len - 1, 1)
        self.frame_slider.set_val(0)

        self._update_plots()
        self._update_display()

    def _on_frame_change(self, val: float) -> None:
        self.current_frame = int(val)
        self._update_display()

    def _on_play_click(self, event: Any) -> None:
        self.playing = not self.playing
        self.play_button.label.set_text("Pause" if self.playing else "Play")

        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _on_prev_click(self, event: Any) -> None:
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.set_val(self.current_frame)

    def _on_next_click(self, event: Any) -> None:
        ep_len = self.backend.get_episode_length(self.current_episode)
        if self.current_frame < ep_len - 1:
            self.current_frame += 1
            self.frame_slider.set_val(self.current_frame)

    def _on_timer(self) -> None:
        if not self.playing:
            return

        ep_len = self.backend.get_episode_length(self.current_episode)
        if self.current_frame < ep_len - 1:
            self.current_frame += 1
            self.frame_slider.set_val(self.current_frame)
        else:
            self.playing = False
            self.play_button.label.set_text("Play")
            self.timer.stop()

    def _update_plots(self) -> None:
        backends = [self.backend]
        if self.compare_backend:
            backends.append(self.compare_backend)

        for b_idx, backend in enumerate(backends):
            if b_idx >= len(self.plot_displays):
                continue

            for ax, lines, marker, feat_key in self.plot_displays[b_idx]:
                ep_data = backend.get_episode_data(self.current_episode, feat_key)
                if ep_data is None:
                    continue

                # Flatten if more than 2D
                if len(ep_data.shape) > 2:
                    ep_data = ep_data.reshape(ep_data.shape[0], -1)

                if len(ep_data.shape) > 1:
                    for dim, line in enumerate(lines):
                        if dim < ep_data.shape[1]:
                            line.set_data(range(len(ep_data)), ep_data[:, dim])
                elif lines:
                    lines[0].set_data(range(len(ep_data)), ep_data)

                ax.relim()
                ax.autoscale_view()

    def _update_display(self) -> None:
        backends = [self.backend]
        if self.compare_backend:
            backends.append(self.compare_backend)

        # Update cameras
        for b_idx, backend in enumerate(backends):
            if b_idx >= len(self.camera_displays):
                continue

            for ax, im, cam_key in self.camera_displays[b_idx]:
                frame = backend.get_frame_image(
                    self.current_episode, self.current_frame, cam_key
                )
                if frame is not None:
                    im.set_data(frame)

        # Update plot markers
        for b_idx, backend in enumerate(backends):
            if b_idx >= len(self.plot_displays):
                continue

            for ax, lines, marker, feat_key in self.plot_displays[b_idx]:
                marker.set_xdata([self.current_frame, self.current_frame])

        self.plt.gcf().canvas.draw_idle()


# Backwards compatibility
class LeRobotV3Viewer(DatasetViewer):
    """Viewer for LeRobot v3 datasets (backwards compatible)."""

    def __init__(self, dataset_path: str | Path):
        super().__init__(dataset_path)


def visualize(
    dataset_path: str | Path,
    compare_path: str | Path | None = None,
) -> None:
    """Visualize a dataset, optionally with comparison.

    Args:
        dataset_path: Path to primary dataset.
        compare_path: Optional path to second dataset for side-by-side comparison.
    """
    viewer = DatasetViewer(dataset_path, compare_path)
    viewer.show()
