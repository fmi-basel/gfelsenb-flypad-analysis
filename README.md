# flypad

Python analysis suite for **flyPAD** (fly Proboscis and Activity Detector) capacitance recordings —
a clean re-implementation of the MATLAB `AnalyseFlypadData v2.2` pipeline (raw capacitance → sips &
activity bouts → feeding bursts & QC → statistics & figures).

- **Dual-mode** engine: `matlab_compat` (faithful to v2.2) and `corrected` (documented bug-fixes), selected in config.
- **Fully YAML-configurable** (Pydantic-validated); no hard-coded parameters.
- **Library + CLI + GUI** over one Qt-free core.

> Status: **M0 — scaffolding.** See `flypad_new_software_design.html` for the full architecture,
> the locked design decisions, and the milestone roadmap.

## Quickstart (uv)

```bash
uv sync                 # create .venv + install (core + dev groups)
uv run flypad --help    # CLI entry point
uv run pytest           # tests
uv run ruff check .     # lint
uv run mypy src         # types
```

Optional groups: `uv sync --group gui` (PySide6), `uv sync --group docs` (MkDocs).

## Sample data

`data/sample/20240215/` holds a complete fixture — two raw `CapacitanceData` recordings (96-channel
"48 flies" config), their `exp_*.txt` / `LOG.txt` sidecars, the MATLAB ground-truth `20240215.mat`, and the
derived `channel_condition_map.csv`. The large raw binaries are git-ignored; the rest is tracked.

## License

GPL-3.0-or-later (inherited from the original flyPAD analysis suite).
