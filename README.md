# Forge

Forge is the normalization layer for robotics data. It converts between robotics episode formats (RLDS, LeRobot v2/v3, Zarr, Rosbag) with minimal friction.

## Installation

```bash
git clone https://github.com/your-org/forge.git
cd forge
pip install -e ".[all]"
```

## Quick Start

```python
import forge

# Inspect a dataset
info = forge.inspect("/path/to/dataset")
print(info)

# Convert between formats
forge.convert("/path/to/rlds", "/path/to/lerobot", target_format="lerobot-v3")

# Visualize any format using the unified viewer
from forge.visualize import unified_visualize
unified_visualize("/path/to/dataset")
```

## CLI Usage

```bash
# Inspect a dataset (local or from HuggingFace Hub)
forge inspect /path/to/dataset
forge inspect hf://lerobot/pusht

# Quick inspect Hub datasets (metadata only, no download)
forge inspect hf://lerobot/aloha_sim_insertion_human --quick

# Convert between formats
forge convert /path/to/input /path/to/output --format lerobot-v3

# Convert directly from HuggingFace Hub
forge convert hf://lerobot/pusht ./output --format lerobot-v3

# Convert with camera name mapping
forge convert input/ output/ --format lerobot-v3 --camera wrist_cam=img

# Convert with parallel processing (faster on multi-core systems)
forge convert input/ output/ --format lerobot-v3 --workers 4

# Visualize a dataset
forge visualize /path/to/dataset

# Visualize with comparison (two datasets side by side)
forge visualize /path/to/dataset1 --compare /path/to/dataset2

# Compute dataset statistics
forge stats /path/to/dataset
forge stats /path/to/dataset --plot              # Show distribution plots
forge stats /path/to/dataset --output stats.json # Export to JSON

# List supported formats (shows Read/Write/Visualize support)
forge formats

# Show version
forge version
```

## HuggingFace Hub Integration

Forge can download and work with datasets directly from HuggingFace Hub:

```bash
# Search for robotics datasets
forge hub                              # Popular robotics datasets
forge hub "robot manipulation"         # Search by query
forge hub --author lerobot             # Filter by author/org

# Download a dataset
forge hub --download lerobot/pusht

# Inspect directly from Hub (downloads and caches automatically)
forge inspect hf://lerobot/pusht

# Convert directly from Hub
forge convert hf://lerobot/pusht ./output --format lerobot-v3
```

Datasets are cached in `~/.cache/forge/datasets/` for reuse.

## Supported Formats

| Format | Read | Write | Visualize | Notes |
|--------|------|-------|-----------|-------|
| RLDS | Yes | Yes | Yes | TensorFlow Datasets format |
| LeRobot v2 | Yes | - | Yes | Parquet + MP4 |
| LeRobot v3 | Yes | Yes | Yes | Parquet + MP4 with chunks |
| Zarr | Yes | Yes | Yes | Diffusion Policy, UMI style |
| HDF5 | Yes | - | Yes | robomimic, ACT/ALOHA |
| Rosbag | Yes | - | - | ROS1 .bag, ROS2 MCAP/.db3 |

## Conversion Examples

```bash
# RLDS to LeRobot v3
forge convert sample_data/robomimic_ph/can_ph_image/1.0.1 output/ --format lerobot-v3

# LeRobot v3 to RLDS (for OpenVLA/Octo training)
forge convert lerobot_dataset/ output_rlds/ --format rlds

# Zarr to LeRobot v3
forge convert sample_data/pusht_zarr output/ --format lerobot-v3

# Zarr to LeRobot v3 with camera mapping
forge convert input.zarr output/ --format lerobot-v3 --camera observation=img
```

## Configuration

For complex conversions with custom field/camera mappings, use a YAML config file:

```bash
# Generate a config template from your dataset
forge inspect my_dataset/ --generate-config config.yaml

# Edit config.yaml to customize mappings, then convert
forge convert my_dataset/ output/ --config config.yaml

# Preview conversion without writing files
forge convert my_dataset/ output/ --config config.yaml --dry-run
```

Example config:
```yaml
target_format: lerobot-v3
fps: 30
robot_type: franka

cameras:
  wrist_camera: wrist_cam
  overhead_camera: front_cam

fields:
  action: data/actions
  state: data/proprio
```

See [docs/configuration.md](docs/configuration.md) for the full configuration reference.

## Visualization

Forge includes a dataset viewer that works with any supported format:

```bash
# Basic visualization
forge visualize /path/to/dataset

# Compare two datasets (e.g., before/after conversion)
forge visualize original/ --compare converted/
```

```python
# Python API - unified viewer works with any format
from forge.visualize import unified_visualize

unified_visualize("/path/to/dataset")

# Comparison mode
unified_visualize("dataset1", "dataset2")
```

The viewer displays camera feeds, action/state plots, and allows scrubbing through episodes.

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed architecture diagrams and information on adding new formats.

## Development

```bash
# Create virtual environment
make venv
source .venv/bin/activate

# Install with dev dependencies
make install-dev

# Run tests
make test

# Run linter and type checker
make check
```

## License

MIT
