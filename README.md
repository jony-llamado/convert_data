# Forge

Convert between robotics dataset formats with one command.

| Format | Read | Write | Notes |
|--------|:----:|:-----:|-------|
| RLDS | ✓ | ✓ | Open-X, TensorFlow Datasets |
| LeRobot v2/v3 | ✓ | ✓ | HuggingFace, Parquet + MP4 |
| Zarr | ✓ | ✓ | Diffusion Policy, UMI |
| HDF5 | ✓ | - | robomimic, ACT/ALOHA |
| Rosbag | ✓ | - | ROS1 .bag, ROS2 MCAP |

## Quick Start

```bash
pip install forge-robotics[all]

# See what's in a dataset
forge inspect /path/to/dataset

# Convert it
forge convert /path/to/rlds ./output --format lerobot-v3
```

Works with HuggingFace Hub too:

```bash
forge inspect hf://lerobot/pusht
forge convert hf://lerobot/pusht ./output --format lerobot-v3
```

## Installation

```bash
# Full install (all formats)
pip install forge-robotics[all]

# Minimal install (specific formats only)
pip install forge-robotics[lerobot]    # LeRobot + Zarr
pip install forge-robotics[rlds]       # RLDS/TensorFlow
pip install forge-robotics[ros]        # Rosbag support
```

Or from source:

```bash
git clone https://github.com/your-org/forge.git
cd forge && pip install -e ".[all]"
```

## Python API

```python
import forge

# Inspect
info = forge.inspect("/path/to/dataset")
print(info.format, info.num_episodes, info.cameras)

# Convert
forge.convert(
    "/path/to/rlds",
    "/path/to/output",
    target_format="lerobot-v3"
)
```

## CLI Reference

See [docs/cli.md](docs/cli.md) for the full command reference including:

- `forge inspect` - Dataset inspection and schema analysis
- `forge convert` - Format conversion with camera mapping
- `forge visualize` - Interactive dataset viewer
- `forge stats` - Compute dataset statistics
- `forge hub` - Search and download from HuggingFace

## Configuration

For complex conversions, use a YAML config:

```bash
forge inspect my_dataset/ --generate-config config.yaml
forge convert my_dataset/ output/ --config config.yaml
```

See [docs/configuration.md](docs/configuration.md) for details.

## Development

```bash
make venv && source .venv/bin/activate
make install-dev
make test
```

## License

MIT
