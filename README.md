# flypad

Python analysis suite for **flyPAD** (fly Proboscis and Activity Detector) capacitance recordings —
a clean re-implementation of the MATLAB `AnalyseFlypadData v2.2` pipeline (raw capacitance → sips &
activity bouts → feeding bursts & QC → statistics & figures).

- **Dual-mode** engine: `matlab_compat` (faithful to v2.2) and `corrected` (documented bug-fixes), selected in config.
- **Fully YAML-configurable** (Pydantic-validated); no hard-coded parameters.
- **Library + CLI + GUI** over one Qt-free core.

> Status: **v0.1.0 — M0–M9 complete.** The full scientific pipeline (detect → post-process →
> stats → figures) runs end-to-end from the CLI and the PySide6 GUI, with docs, packaging, and a
> tag-triggered release workflow in place. See `CHANGELOG.md` for the release notes and
> `flypad_new_software_design.html` for the architecture and decisions log.

## Quickstart (uv)

```bash
uv sync                 # create .venv + install (core + dev groups)
uv run flypad --help    # CLI entry point
uv run pytest           # tests
uv run ruff check .     # lint
uv run mypy src         # types
```

Optional groups: `uv sync --group gui` (PySide6), `uv sync --group docs` (MkDocs).
Published extras for end users: `pip install flypad[gui]`. Build the docs site with
`uv run --group docs mkdocs build` (or `mkdocs serve`).

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
overrides. Outputs land as tidy Parquet/CSV tables plus PNG/PDF figures in the results directory.

## Desktop GUI

```bash
uv sync --group gui     # install PySide6 + qtpy (one-time)
uv run flypad gui       # drag-drop a folder, tweak config, Run
```

A thin Qt view over the same pipeline: drag-drop a recordings folder, adjust the schema-driven
config panel, and run on a worker thread; the per-condition table and dashboard appear when it
finishes. GUI tests run headless (offscreen Qt) via `uv run --group gui pytest`.

## Sample data

`data/sample/20240215/` holds a complete fixture — two raw `CapacitanceData` recordings (96-channel
"48 flies" config), their `exp_*.txt` / `LOG.txt` sidecars, the MATLAB ground-truth `20240215.mat`, and the
derived `channel_condition_map.csv`. The large raw binaries are git-ignored; the rest is tracked. With them
present, `flypad run` reproduces the starvation response (fully fed ≪ 24 h ≪ 44 h sips per fly).

## License

GPL-3.0-or-later (inherited from the original flyPAD analysis suite).
