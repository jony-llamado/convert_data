# CLI Reference

Complete reference for all Forge commands.

## forge inspect

Analyze a dataset's structure and schema.

```bash
# Basic inspection
forge inspect /path/to/dataset

# Inspect from HuggingFace Hub
forge inspect hf://lerobot/pusht

# Quick inspect (metadata only, no download)
forge inspect hf://lerobot/pusht --quick

# Generate a conversion config template
forge inspect /path/to/dataset --generate-config config.yaml

# Output as JSON
forge inspect /path/to/dataset --output json

# Deep scan (reads all episodes, slower but more accurate)
forge inspect /path/to/dataset --deep

# Sample specific number of episodes
forge inspect /path/to/dataset --samples 10
```

**Output includes:**
- Detected format
- Episode and frame counts
- Camera names and resolutions
- State/action dimensions
- Language instruction samples

---

## forge convert

Convert datasets between formats.

```bash
# Basic conversion
forge convert /path/to/input /path/to/output --format lerobot-v3

# From HuggingFace Hub
forge convert hf://lerobot/pusht ./output --format lerobot-v3

# With camera name mapping
forge convert input/ output/ --format lerobot-v3 --camera wrist_cam=img

# Multiple camera mappings
forge convert input/ output/ --format lerobot-v3 \
    --camera agentview=front \
    --camera eye_in_hand=wrist

# Parallel processing (faster on multi-core systems)
forge convert input/ output/ --format lerobot-v3 --workers 4

# Using a config file
forge convert input/ output/ --config config.yaml

# Dry run (preview without writing)
forge convert input/ output/ --format lerobot-v3 --dry-run

# Open visualizer after conversion to compare
forge convert input/ output/ --format lerobot-v3 --visualize

# Specify source format (if auto-detection fails)
forge convert input/ output/ --format lerobot-v3 --source-format hdf5
```

**Target formats:**
- `lerobot-v3` - LeRobot v3 (recommended for HuggingFace)
- `rlds` - RLDS/TensorFlow Datasets

---

## forge visualize

Interactive dataset viewer.

```bash
# View a dataset
forge visualize /path/to/dataset

# Compare two datasets side by side
forge visualize /path/to/original --compare /path/to/converted

# Specify starting episode
forge visualize /path/to/dataset --episode 5
```

**Viewer controls:**
- Episode slider: Navigate between episodes
- Frame slider: Navigate frames within an episode
- Play/Pause button: Auto-play frames

---

## forge stats

Compute dataset statistics.

```bash
# Basic statistics
forge stats /path/to/dataset

# With distribution plots (requires matplotlib)
forge stats /path/to/dataset --plot

# Export to JSON
forge stats /path/to/dataset --output stats.json

# Sample subset of episodes (faster)
forge stats /path/to/dataset --sample 100
```

**Statistics include:**
- Episode counts (total, min/max/mean frames)
- Coverage metrics (language, success labels, rewards)
- Action/state distributions (min, max, mean, std per dimension)

---

## forge export-video

Export camera videos from any dataset format.

```bash
# Export first episode (all cameras as grid)
forge export-video /path/to/dataset -o demo.mp4

# Export specific episode
forge export-video /path/to/dataset -e 5 -o episode5.mp4

# Export specific camera only
forge export-video /path/to/dataset -c wrist_cam -o wrist.mp4

# Export all episodes to a directory
forge export-video /path/to/dataset --all -o ./videos/

# From HuggingFace Hub
forge export-video hf://lerobot/pusht -o pusht_demo.mp4

# Override FPS
forge export-video /path/to/dataset -o demo.mp4 --fps 30

# Force grid layout for multiple cameras
forge export-video /path/to/dataset -o demo.mp4 --grid
```

**Options:**
- `-e, --episode`: Episode index (default: 0)
- `-c, --camera`: Export only this camera
- `-a, --all`: Export all episodes
- `-f, --fps`: Override frames per second
- `-g, --grid`: Combine cameras into grid layout

---

## forge hub

Search and download datasets from HuggingFace Hub.

```bash
# List popular robotics datasets
forge hub

# Search by query
forge hub "robot manipulation"
forge hub "droid"

# Filter by author/organization
forge hub --author lerobot
forge hub --author berkeley-humanoid

# Download a dataset
forge hub --download lerobot/pusht
```

Downloaded datasets are cached in `~/.cache/forge/datasets/`.

---

## forge formats

List supported formats and their capabilities.

```bash
forge formats
```

Shows read/write/visualize support for each format.

---

## forge version

Show version information.

```bash
forge version
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FORGE_CACHE_DIR` | Dataset cache location | `~/.cache/forge` |
| `FORGE_LOG_LEVEL` | Logging verbosity | `INFO` |

---

## Examples

### Convert Open-X dataset for LeRobot training

```bash
# Download from HuggingFace
forge hub --download openvla/droid_100

# Inspect to understand structure
forge inspect ~/.cache/forge/datasets/openvla/droid_100

# Convert to LeRobot v3
forge convert ~/.cache/forge/datasets/openvla/droid_100 \
    ./droid_lerobot \
    --format lerobot-v3 \
    --workers 4
```

### Convert LeRobot dataset for OpenVLA/Octo training

```bash
# LeRobot → RLDS
forge convert hf://lerobot/pusht ./pusht_rlds --format rlds
```

### Verify conversion quality

```bash
# Compare original vs converted
forge visualize original_dataset/ --compare converted_dataset/

# Check statistics match
forge stats original_dataset/ --output original.json
forge stats converted_dataset/ --output converted.json
diff original.json converted.json
```
