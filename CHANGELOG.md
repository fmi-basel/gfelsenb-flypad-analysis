# Changelog

All notable changes to **flypad** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-26

First public release — the full flyPAD analysis pipeline, faithful to the MATLAB
`AnalyseFlypadData v2.2` suite in `matlab_compat` mode, with a documented `corrected`
mode for the known bug-fixes.

### Added
- **I/O & config** — raw `uint16` reader → labelled xarray; layered Pydantic config
  (preset → YAML → `--set`) with JSON-Schema export.
- **Detection** — vectorised de-trend / RMS / derivative-threshold; greedy (compat) and
  adjacent (corrected) sip pairing; activity bouts. Reproduces the MATLAB ground truth on
  the `20240215` sample at ~92% precision / ~93% recall (onsets ±2 samples), counts within
  ~1%.
- **Post-processing** — feeding bursts (`GET_FEEDING_BURSTS`), spill/unconnected QC and
  non-eater removal, channel-to-channel transitions, and the `exp_*.txt`/`LOG.txt`
  board→condition metadata parser.
- **Statistics** — per-fly / per-condition summaries (mean/median/SEM/CI), ICDF/CCDF
  distributions, permutation tests, polynomial fits; two-choice preference index and
  cumulative time courses.
- **Figures** — tilted box plots (per-fly dots, N, tilt), CDF/CCDF, rasters, cumulative
  time courses (mean ± SEM), two-choice substrate plots, and the standalone dashboard,
  with a stable per-condition palette and PNG + PDF export.
- **CLI** — `flypad run / detect / stats / plot / validate / config / gui`, writing tidy
  Parquet/CSV tables, figures, and run provenance (`run_info.json` + `config.used.yaml`).
- **GUI** — PySide6 desktop app (drag-drop, schema-driven config panel, progress bar,
  metric selector, embedded interactive dashboard) over the same pipeline.

[0.1.0]: https://github.com/dgoldschmidt/flypad/releases/tag/v0.1.0
