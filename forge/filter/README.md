# Forge Filter

Filter dataset episodes based on quality scores, flags, or explicit episode lists. Writes the same format as input — no format conversion.

## Usage

### CLI

```bash
# Dry-run: preview what passes/fails (no output written)
forge filter ./my_dataset --min-quality 6.0

# Write filtered dataset
forge filter ./my_dataset ./filtered --min-quality 6.0

# Exclude episodes with specific quality flags
forge filter ./my_dataset ./filtered --exclude-flags jerky,mostly_static

# Use pre-computed quality report (faster — skips re-analysis)
forge quality ./my_dataset --export report.json
forge filter ./my_dataset ./filtered --from-report report.json --min-quality 7.0

# Explicit episode selection
forge filter ./my_dataset ./filtered --include-episodes ep_000,ep_001,ep_005
forge filter ./my_dataset ./filtered --exclude-episodes ep_003,ep_010
```

### Python API

```python
from forge.filter import FilterEngine, FilterConfig

# Filter by quality score
config = FilterConfig(min_quality=6.0)
engine = FilterEngine(config)
result = engine.filter("./my_dataset", output="./filtered")
print(result.episodes_kept, result.episodes_excluded)

# Dry-run (no output path)
result = engine.filter("./my_dataset")
print(result.kept_ids, result.excluded_ids)

# With pre-computed report
config = FilterConfig(
    min_quality=7.0,
    exclude_flags=["jerky", "mostly_static"],
    from_report="report.json",
)
engine = FilterEngine(config)
result = engine.filter("./my_dataset", output="./filtered")
```

## Filter Criteria

| Option | Description |
|--------|-------------|
| `--min-quality` | Keep episodes with overall score >= threshold (0-10) |
| `--exclude-flags` | Exclude episodes with any matching flag (comma-separated) |
| `--include-episodes` | Only keep these episode IDs (comma-separated) |
| `--exclude-episodes` | Remove these episode IDs (comma-separated) |
| `--from-report` | Load quality data from a JSON report instead of re-analyzing |

### Available Flags

These are the flags produced by `forge quality`:

- `jerky` — LDLJ smoothness below threshold
- `mostly_static` — >50% of episode is idle
- `gripper_chatter` — rapid open/close transitions
- `dead_actions` — >10% zero-action frames
- `timestamp_jitter` — irregular frame timing
- `low_entropy` — repetitive actions
- `saturated` — actions hitting hardware limits
- `nan_in_actions` / `nan_in_states` — NaN values detected

## Workflow

Typical quality-based filtering workflow:

```bash
# 1. Score the dataset
forge quality ./my_dataset --export report.json

# 2. Preview what would be filtered
forge filter ./my_dataset --from-report report.json --min-quality 6.0

# 3. Write filtered dataset
forge filter ./my_dataset ./filtered --from-report report.json --min-quality 6.0
```

## Module Structure

```
forge/filter/
    __init__.py       Public API exports
    engine.py         FilterEngine, FilterConfig, FilterResult
```
