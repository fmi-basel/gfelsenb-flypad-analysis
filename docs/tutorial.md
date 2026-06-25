# Tutorial — analysing the `20240215` sample

This walks through a full analysis of the bundled sample (two 96-channel recordings,
five starvation conditions) from both the CLI and the GUI.

## 1. The data

`data/sample/20240215/` holds a complete fixture:

- two raw `CapacitanceData_*` recordings (git-ignored; ask the maintainer for them),
- their `exp_*.txt` / `LOG.txt` sidecars (the real condition/substrate layout),
- the MATLAB ground-truth `20240215.mat`, and
- the derived `channel_condition_map.csv`.

The matching config is `configs/example_experiment.yaml` (96 channels, 5 conditions).

## 2. Run the whole pipeline

```bash
uv run flypad run data/sample/20240215 \
    -c configs/example_experiment.yaml \
    -o results/
```

`results/` now contains:

```
events.{parquet,csv}        # one row per sip (+ burst membership, metadata)
per_fly.{parquet,csv}       # one row per channel (metrics + non_eater QC flag)
per_condition.{parquet,csv} # mean / median / SEM / CI per condition
run_info.json               # flypad version, config hash, inputs (provenance)
config.used.yaml            # the fully-resolved config
figures/                    # dashboard, boxplot, ccdf, raster, timecourse, substrate
```

!!! tip "The `-c` flag matters"
    Without `-c`, flypad falls back to the 64-channel default and the 96-channel sample
    will error. Always pass the experiment config (or `--set hardware.n_channels=96`).

## 3. Re-derive stats and figures

The tables are the keystone — recompute downstream artefacts without re-detecting:

```bash
uv run flypad stats results/                 # re-print + rewrite per_condition
uv run flypad plot  results/ --kind dashboard,timecourse --metric n_feeding_bursts
```

## 4. Check parity against MATLAB

```bash
uv run flypad validate data/sample/20240215 \
    --against data/sample/20240215/20240215.mat \
    -c configs/example_experiment.yaml
```

→ ~92% precision / ~93% recall, a constant −53 sample frame offset.

## 5. The desktop GUI

```bash
uv sync --group gui
uv run flypad gui
```

Drag the recordings folder onto the drop area, point *Config YAML* at
`configs/example_experiment.yaml`, pick a metric, and press **Run analysis**. Progress
streams to the bar; when it finishes the per-condition table and an interactive dashboard
appear, and the same files are written to the output directory.
