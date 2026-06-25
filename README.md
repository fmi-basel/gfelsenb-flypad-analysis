# flypad

Python analysis suite for **flyPAD** (fly Proboscis and Activity Detector) capacitance recordings —
a clean re-implementation of the MATLAB `AnalyseFlypadData v2.2` pipeline (raw capacitance → sips &
activity bouts → feeding bursts & QC → statistics & figures).

- **Dual-mode** engine: `matlab_compat` (faithful to v2.2) and `corrected` (documented bug-fixes), selected in config.
- **Fully YAML-configurable** (Pydantic-validated); no hard-coded parameters.
- **Library + CLI + GUI** over one Qt-free core.

> Status: **M0–M7 done** — the full scientific pipeline (detect → post-process → stats → figures) runs
> end-to-end from the CLI. GUI (M8) and packaged release (M9) remain. See
> `flypad_new_software_design.html` for the architecture, locked decisions, and milestone roadmap.

## Quickstart (uv)

```bash
uv sync                 # create .venv + install (core + dev groups)
uv run flypad --help    # CLI entry point
uv run pytest           # tests
uv run ruff check .     # lint
uv run mypy src         # types
```

Optional groups: `uv sync --group gui` (PySide6), `uv sync --group docs` (MkDocs).

## Run the pipeline

```bash
# full pipeline: raw → events + per-fly/per-condition tables + figures
uv run flypad run data/sample/20240215 -c configs/example_experiment.yaml -o results/

# detection only; (re)compute stats and figures from saved results
uv run flypad detect data/sample/20240215 -c configs/example_experiment.yaml -o results/
uv run flypad stats results/
uv run flypad plot  results/ --kind dashboard,boxplot,cdf,raster

# parity against the MATLAB ground truth
uv run flypad validate data/sample/20240215 \
    --against data/sample/20240215/20240215.mat -c configs/example_experiment.yaml
```

Every command takes `-c/--config`, `--mode matlab_compat|corrected`, and repeatable `--set key=value`
overrides. Outputs land as tidy Parquet/CSV tables plus PNG/EPS figures in the results directory.

## Sample data

`data/sample/20240215/` holds a complete fixture — two raw `CapacitanceData` recordings (96-channel
"48 flies" config), their `exp_*.txt` / `LOG.txt` sidecars, the MATLAB ground-truth `20240215.mat`, and the
derived `channel_condition_map.csv`. The large raw binaries are git-ignored; the rest is tracked. With them
present, `flypad run` reproduces the starvation response (fully fed ≪ 24 h ≪ 44 h sips per fly).

## License

GPL-3.0-or-later (inherited from the original flyPAD analysis suite).
