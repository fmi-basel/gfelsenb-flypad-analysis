"""Experiment-level orchestration: discover → detect → postprocess → summarise → write.

The per-recording science (``detect_recording`` + ``detect_feeding_bursts``) runs once
per file; the channel→condition map, the tidy tables, and the figures are assembled at
the experiment level. This is the single entry point the CLI (and later the GUI) drive —
they contain no science themselves.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from flypad import __version__
from flypad.config.models import Config
from flypad.datamodel import load_recording
from flypad.detect.results import ChannelBouts, ChannelSips
from flypad.detect.run import detect_recording
from flypad.io.discovery import find_capacitance_files
from flypad.postprocess.bursts import ChannelBursts, detect_feeding_bursts
from flypad.postprocess.metadata import (
    channel_condition_map_for_dir,
    channel_map_from_filenames,
)
from flypad.stats.summaries import (
    apply_qc_removal,
    build_event_table,
    mark_non_eaters,
    per_condition_summary,
    per_fly_summary,
)

Progress = Callable[[str], None]

#: Tables written to a results directory; preferred read order for re-loading.
TABLE_NAMES = ("events", "per_fly", "per_condition")
_TABLE_FORMATS = ("parquet", "csv")


@dataclass
class DetectionResult:
    """Raw per-file detection output plus the channel→condition map."""

    files: list[Path]
    sips_by_file: list[list[ChannelSips]]
    bouts_by_file: list[list[ChannelBouts]]
    bursts_by_file: list[list[ChannelBursts]]
    channel_map: pd.DataFrame


def _emit(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _default_channel_map(files: Sequence[Path], config: Config) -> pd.DataFrame:
    """Minimal channel map when no ``exp_*.txt`` sidecars exist (single condition)."""
    n = config.hardware.n_channels
    step = config.metadata.channels_per_board_position
    rows: list[dict[str, object]] = []
    for file_index, path in enumerate(files):
        for channel in range(n):
            is_left = channel % 2 == 0
            rows.append(
                {
                    "file_index": file_index,
                    "file_name": path.name,
                    "exp_file": "",
                    "board_position": channel // step + 1,
                    "channel": channel,
                    "condition": 1,
                    "condition_label": "all",
                    "condition_short": "all",
                    "sex": "",
                    "substrate": 1 if is_left else 2,
                    "substrate_side": "left" if is_left else "right",
                    "substrate_label": "",
                }
            )
    return pd.DataFrame(rows)


def _resolve_channel_map(
    data_dir: str | Path, files: Sequence[Path], config: Config
) -> pd.DataFrame:
    try:
        return channel_condition_map_for_dir(
            data_dir,
            n_channels=config.hardware.n_channels,
            channels_per_board_position=config.metadata.channels_per_board_position,
        )
    except FileNotFoundError:
        pass
    # No exp_*.txt sidecars: fall back to the filename condition spans (the MATLAB
    # mechanism), then to a single "all" condition if the filenames carry none.
    from flypad.io.discovery import parse_filename

    if any(parse_filename(f.name).condition_spans for f in files):
        return channel_map_from_filenames(
            files,
            n_channels=config.hardware.n_channels,
            channels_per_board_position=config.metadata.channels_per_board_position,
        )
    return _default_channel_map(files, config)


def detect_experiment(
    data_dir: str | Path,
    config: Config,
    *,
    progress: Progress | None = None,
) -> DetectionResult:
    """Discover recordings and run detection + feeding-burst grouping on each."""
    files = find_capacitance_files(data_dir)
    if not files:
        raise FileNotFoundError(f"no CapacitanceData* files in {data_dir}")

    sips_by_file: list[list[ChannelSips]] = []
    bouts_by_file: list[list[ChannelBouts]] = []
    bursts_by_file: list[list[ChannelBursts]] = []
    for i, path in enumerate(files, 1):
        _emit(progress, f"detect [{i}/{len(files)}] {path.name}")
        recording = load_recording(path, config)
        result = detect_recording(recording.capacitance, config)
        _, bursts = detect_feeding_bursts(result.sips, config.feeding_bursts)
        sips_by_file.append(result.sips)
        bouts_by_file.append(result.bouts)
        bursts_by_file.append(bursts)

    _emit(progress, "build channel→condition map")
    channel_map = _resolve_channel_map(data_dir, files, config)
    return DetectionResult(
        files=files,
        sips_by_file=sips_by_file,
        bouts_by_file=bouts_by_file,
        bursts_by_file=bursts_by_file,
        channel_map=channel_map,
    )


def build_tables(detection: DetectionResult, config: Config) -> dict[str, pd.DataFrame]:
    """Build the ``events`` / ``per_fly`` / ``per_condition`` tables.

    ``per_fly`` keeps every channel with a ``non_eater`` QC flag; ``per_condition``
    aggregates only the kept (non-removed) flies.
    """
    rate = config.hardware.sampling_rate_hz
    events = build_event_table(
        detection.sips_by_file, detection.bursts_by_file, detection.channel_map, rate
    )
    per_fly = per_fly_summary(
        detection.sips_by_file,
        detection.bouts_by_file,
        detection.bursts_by_file,
        detection.channel_map,
        rate,
    )
    per_fly = mark_non_eaters(per_fly, config.non_eaters)
    kept = apply_qc_removal(per_fly)
    per_condition = per_condition_summary(kept, ci_level=config.stats.ci_level)
    return {"events": events, "per_fly": per_fly, "per_condition": per_condition}


def write_tables(
    tables: dict[str, pd.DataFrame],
    out_dir: str | Path,
    *,
    formats: Sequence[str] = ("parquet", "csv"),
) -> list[Path]:
    """Write each table in the requested formats (only ``parquet``/``csv`` apply)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    use = [f for f in formats if f in _TABLE_FORMATS] or ["csv"]
    written: list[Path] = []
    for name, df in tables.items():
        for fmt in use:
            path = out / f"{name}.{fmt}"
            if fmt == "parquet":
                df.to_parquet(path, index=False)
            else:
                df.to_csv(path, index=False)
            written.append(path)
    return written


def config_hash(config: Config) -> str:
    """A stable SHA-256 over the resolved config (provenance / reproducibility)."""
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def write_provenance(
    out_dir: str | Path,
    config: Config,
    *,
    files: Sequence[Path],
    command: str = "run",
    extra: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> list[Path]:
    """Write ``run_info.json`` (version, config hash, inputs) + ``config.used.yaml``.

    Together these make any results directory self-describing and reproducible.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {
        "flypad_version": __version__,
        "command": command,
        "timestamp": timestamp or datetime.now(UTC).isoformat(),
        "mode": config.mode.value,
        "config_hash": config_hash(config),
        "n_files": len(files),
        "files": [Path(f).name for f in files],
    }
    if extra:
        info.update(extra)
    run_info = out / "run_info.json"
    used = out / "config.used.yaml"
    run_info.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    used.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return [run_info, used]


def read_table(results_dir: str | Path, name: str) -> pd.DataFrame:
    """Read a saved table by name, preferring parquet over csv."""
    out = Path(results_dir)
    for fmt in _TABLE_FORMATS:
        path = out / f"{name}.{fmt}"
        if path.exists():
            return pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path)
    raise FileNotFoundError(f"no {name}.(parquet|csv) in {results_dir}")


def _kept(per_fly: pd.DataFrame) -> pd.DataFrame:
    return apply_qc_removal(per_fly) if "non_eater" in per_fly.columns else per_fly


#: Figure kinds renderable by :func:`render_figures`.
FIGURE_KINDS = ("dashboard", "boxplot", "cdf", "raster", "timecourse", "substrate")


def render_figures(
    per_fly: pd.DataFrame,
    events: pd.DataFrame | None,
    out_dir: str | Path,
    config: Config,
    *,
    kinds: Sequence[str] = FIGURE_KINDS,
    metric: str = "n_sips",
    progress: Progress | None = None,
) -> list[Path]:
    """Render the requested figure kinds into ``out_dir/figures``."""
    from flypad.plotting import (
        ccdf_plot,
        condition_palette,
        raster_plot,
        save_figure,
        set_theme,
        shaded_lines,
        standalone_dashboard,
        substrate_comparison,
        tilted_boxplot,
    )
    from flypad.stats import cumulative_timecourse_by_condition

    set_theme()
    figdir = Path(out_dir) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    vector = config.plotting.vector_format
    formats = ("png",) if vector == "none" else ("png", vector)
    dpi = config.plotting.dpi
    rate = config.hardware.sampling_rate_hz
    kept = _kept(per_fly)
    group_col = "condition_label" if "condition_label" in kept.columns else "condition"
    groups = {
        str(label): grp[metric].to_numpy() for label, grp in kept.groupby(group_col, dropna=False)
    }
    palette = condition_palette(groups.keys())
    written: list[Path] = []

    def save(fig: object, stem: str) -> None:
        _emit(progress, f"figure {stem}")
        written.extend(save_figure(fig, figdir / stem, formats=formats, dpi=dpi, close=True))

    if "dashboard" in kinds:
        save(standalone_dashboard(kept, metric, central="median"), f"dashboard_{metric}")
    if "boxplot" in kinds:
        save(tilted_boxplot(groups, palette=palette, ylabel=metric).figure, f"boxplot_{metric}")
    if "cdf" in kinds:
        save(ccdf_plot(groups, palette=palette, xlabel=metric).figure, f"ccdf_{metric}")
    if "substrate" in kinds and "substrate_side" in kept.columns:
        save(substrate_comparison(kept, metric, ylabel=metric).figure, f"substrate_{metric}")
    if "timecourse" in kinds and events is not None and not events.empty:
        series = cumulative_timecourse_by_condition(
            events, config.acquisition.duration_samples, group_col=group_col, sampling_rate_hz=rate
        )
        save(shaded_lines(series, palette=palette).figure, "timecourse")
    if "raster" in kinds and events is not None and not events.empty:
        # Full plate of the first file: every channel (silent or QC-removed too).
        fi0 = int(events["file_index"].min())
        first = events[events["file_index"] == fi0]
        channels = list(range(config.hardware.n_channels))
        onsets_by_ch = {int(c): g["onset"].to_numpy() for c, g in first.groupby("channel")}
        rows = [onsets_by_ch.get(c, []) for c in channels]
        pf0 = per_fly[per_fly["file_index"] == fi0].drop_duplicates("channel").set_index("channel")
        row_conditions = (
            [str(pf0.loc[c, group_col]) if c in pf0.index else "" for c in channels]
            if group_col in pf0.columns
            else None
        )
        raster = raster_plot(
            rows, row_conditions=row_conditions, palette=palette, sampling_rate_hz=rate
        )
        save(raster.figure, "raster")

    return written
