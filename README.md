<p align="center">
<pre>
███████╗ ██████╗ ██████╗  ██████╗ ███████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
█████╗  ██║   ██║██████╔╝██║  ███╗█████╗
██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
</pre>
<h2>⚒ Robotics Data Toolkit ⚒</h2>
<i>Convert, inspect, visualize, and score robotics datasets across every major format.</i>
<br><br>
<a href="https://github.com/arpitg1304/forge"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square"></a>
<a href="https://github.com/arpitg1304/forge/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green?style=flat-square"></a>
<br><br>
<code>RLDS ═══╗         ╔═══► LeRobot</code><br>
<code>Zarr ═══╬════⚙════╬═══► RoboDM</code><br>
<code>HDF5 ═══╝         ╚═══► RLDS</code>
</p>

Convert between robotics dataset formats with one command. Score demonstration quality with research-backed metrics.

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

## Quality Metrics

Automated episode-level quality scoring from proprioception data alone — no video processing needed.

```bash
forge quality ./my_dataset
forge quality hf://lerobot/aloha_sim_cube --export report.json
```

Scores each episode 0-10 based on 8 research-backed metrics:

- **Smoothness (LDLJ)** — jerk-based smoothness from motor control literature (Hogan & Sternad, 2009)
- **Dead actions** — zero/constant action detection (Kim et al. "OpenVLA", 2024)
- **Gripper chatter** — rapid open/close transitions (Sakr et al., 2024)
- **Static detection** — idle periods where the robot isn't moving (Liu et al. "SCIZOR", 2025)
- **Timestamp regularity** — dropped frames and frequency jitter
- **Action saturation** — time spent at hardware limits
- **Action entropy** — diversity vs repetitiveness (Belkhale et al. "DemInf", 2025)
- **Path length** — wandering/hesitation in joint space

See [forge/quality/README.md](forge/quality/README.md) for full metric details, paper references, and how to add new metrics.

## CLI Reference

See [docs/cli.md](docs/cli.md) for the full command reference including:

- `forge inspect` - Dataset inspection and schema analysis
- `forge convert` - Format conversion with camera mapping
- `forge visualize` - Interactive dataset viewer
- `forge quality` - Episode-level quality scoring ([details](forge/quality/README.md))
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

- [ ] **Dataset merging** - Combine multiple datasets into one (`forge merge ds1/ ds2/ --output combined/`)
- [ ] **Train/val/test splitting** - Split datasets with stratification (`--split 80/10/10`)
- [ ] **Streaming reads** - Process HuggingFace datasets without full download
- [ ] **Episode filtering** - Filter by quality score, flags, or episode IDs (`forge filter --min-quality 6.0`)
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
