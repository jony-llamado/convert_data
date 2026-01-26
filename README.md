<p align="center">
<pre>
███████╗ ██████╗ ██████╗  ██████╗ ███████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
█████╗  ██║   ██║██████╔╝██║  ███╗█████╗
██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
</pre>
<b>⚒ Robotics Data Format Converter ⚒</b>
<br><br>
<code>RLDS ═══╗         ╔═══► LeRobot</code><br>
<code>Zarr ═══╬════⚙════╬═══► RoboDM</code><br>
<code>HDF5 ═══╝         ╚═══► RLDS</code>
</p>

Convert between robotics dataset formats with one command.

| Format | Read | Write | Visualize | Notes |
|--------|:----:|:-----:|:---------:|-------|
| RLDS | ✓ | ✓ | ✓ | Open-X, TensorFlow Datasets |
| LeRobot v2/v3 | ✓ | ✓ | ✓ | HuggingFace, Parquet + MP4 |
| GR00T | ✓ | - | ✓ | NVIDIA Isaac, LeRobot v2 with embodiment metadata |
| RoboDM | ✓ | ✓ | ✓ | Berkeley's .vla format, up to 70x compression* |
| Zarr | ✓ | - | ✓ | Diffusion Policy, UMI |
| HDF5 | ✓ | - | ✓ | robomimic, ACT/ALOHA |
| Rosbag | ✓ | - | ✓ | ROS1 .bag, ROS2 MCAP |

*\*RoboDM requires manual installation from GitHub (see below)*

See [docs/model_formats.md](docs/model_formats.md) for which models (Octo, OpenVLA, ACT, Diffusion Policy, etc.) use which format. See [docs/format_reference.md](docs/format_reference.md) for detailed format specifications.

## Why Forge?

Every robotics lab has their own data format: Open-X uses RLDS, HuggingFace uses LeRobot, Diffusion Policy uses Zarr, robomimic uses HDF5. Want to train Octo on your ALOHA data? Write a converter. Want to use LeRobot on Open-X datasets? Write another.

Forge uses a hub-and-spoke architecture — one intermediate representation, O(n) format support:

```
Any Reader → Episode/Frame → Any Writer
```

Add a reader, get all writers for free. Add a writer, get all readers for free. No N×M conversion logic. See [docs/architecture.md](docs/architecture.md) for details.

## Quick Start

```bash
git clone https://github.com/arpitg1304/forge.git
cd forge
pip install -e ".[all]"
```

### RoboDM Support (Optional)

RoboDM requires manual installation from GitHub (PyPI version has a codec bug):

```bash
git clone https://github.com/BerkeleyAutomation/robodm.git
pip install -e robodm
```

### Usage

```bash
# See what's in a dataset
forge inspect /path/to/dataset

# Convert it
forge convert /path/to/rlds ./output --format lerobot-v3
forge convert hf://arpitg1304/stack_lego ./stack_lego_rlds --format rlds --workers 4 --visualize
forge convert hf://lerobot/pusht ./pusht_robodm --format robodm
```

Works with HuggingFace Hub too:

```bash
forge inspect hf://lerobot/pusht
forge convert hf://lerobot/pusht ./output --format lerobot-v3
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
- `forge export-video` - Extract camera videos as MP4
- `forge hub` - Search and download from HuggingFace

## Configuration

For complex conversions, use a YAML config:

```bash
forge inspect my_dataset/ --generate-config config.yaml
forge convert my_dataset/ output/ --config config.yaml
```

See [docs/configuration.md](docs/configuration.md) for details.

## Roadmap

Planned features (contributions welcome!):

- [ ] **Streaming reads** - Process HuggingFace datasets without full download
- [ ] **Episode filtering** - Convert only specific episodes (`--episodes 100-200`)
- [ ] **Depth/point cloud support** - Preserve depth streams from RLDS/Open-X
- [ ] **GR00T writer** - Write to NVIDIA Isaac GR00T training format (read support complete)
- [ ] **Distributed conversion** - Scale to 100K+ episode datasets across nodes
- [ ] **Conversion verification** - Automated diff between source and converted data

## Development

```bash
make venv && source .venv/bin/activate
make install-dev
make test
```

## License

MIT
