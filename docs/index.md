# flypad

A ground-up Python re-implementation of the MATLAB **`AnalyseFlypadData v2.2`** pipeline
for **flyPAD** (fly Proboscis and Activity Detector) capacitance recordings:

> raw capacitance → sips & activity bouts → feeding bursts & QC → statistics & figures

- **Dual-mode** engine: `matlab_compat` (faithful to v2.2, quirks included) and
  `corrected` (documented bug-fixes), selected in config.
- **Fully YAML-configurable** (Pydantic-validated, JSON-Schema-backed); no hard-coded
  parameters.
- **Library + CLI + GUI** over one Qt-free scientific core.

## Install

```bash
pip install flypad          # core (CLI + library)
pip install flypad[gui]     # + PySide6 desktop GUI
```

Or, for development with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group gui --group docs
```

## At a glance

```bash
uv run flypad run data/sample/20240215 -c configs/example_experiment.yaml -o results/
```

produces tidy `events` / `per_fly` / `per_condition` tables (Parquet + CSV), a figure set
(PNG + PDF), and run provenance (`run_info.json` + `config.used.yaml`).

See the [tutorial](tutorial.md) for a full walkthrough, the [CLI](cli.md) reference, the
[configuration](configuration.md) guide, and the [API reference](api.md).

## Parity

In `matlab_compat` mode the detector reproduces the MATLAB ground truth on the `20240215`
sample at **~92% precision / ~93% recall** (onsets ±2 samples), with total sip counts
within ~1% — and the per-condition feeding response reproduces the expected starvation
gradient (fully fed ≪ 24 h ≪ 44 h sips per fly).
