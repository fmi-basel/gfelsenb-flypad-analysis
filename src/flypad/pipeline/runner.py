"""Experiment-level orchestration: discover → detect → postprocess → summarise → write.

The per-recording science (``detect_recording`` + ``detect_feeding_bursts``) runs once
per file; the channel→condition map, the tidy tables, and the figures are assembled at
the experiment level. This is the single entry point the CLI (and later the GUI) drive —
they contain no science themselves.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from flypad.config.models import Config
from flypad.datamodel import load_recording
from flypad.detect.results import ChannelBouts, ChannelSips
from flypad.detect.run import detect_recording
from flypad.io.discovery import find_capacitance_files
from flypad.postprocess.bursts import ChannelBursts, detect_feeding_bursts
from flypad.postprocess.metadata import channel_condition_map_for_dir
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


def render_figures(
    per_fly: pd.DataFrame,
    events: pd.DataFrame | None,
    out_dir: str | Path,
    config: Config,
    *,
    kinds: Sequence[str] = ("dashboard", "boxplot", "cdf", "raster"),
    metric: str = "n_sips",
    progress: Progress | None = None,
) -> list[Path]:
    """Render the requested figure kinds into ``out_dir/figures``."""
    from flypad.plotting import (
        cdf_plot,
        raster_plot,
        save_figure,
        set_theme,
        standalone_dashboard,
        tilted_boxplot,
    )

    set_theme()
    figdir = Path(out_dir) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    formats = ("png", "eps") if config.plotting.save_eps else ("png",)
    dpi = config.plotting.dpi
    kept = _kept(per_fly)
    group_col = "condition_label" if "condition_label" in kept.columns else "condition"
    groups = {
        str(label): grp[metric].to_numpy() for label, grp in kept.groupby(group_col, dropna=False)
    }
    written: list[Path] = []

    def save(fig: object, stem: str) -> None:
        _emit(progress, f"figure {stem}")
        written.extend(save_figure(fig, figdir / stem, formats=formats, dpi=dpi, close=True))

    if "dashboard" in kinds:
        save(standalone_dashboard(kept, metric, central="median"), f"dashboard_{metric}")
    if "boxplot" in kinds:
        save(tilted_boxplot(groups, ylabel=metric).figure, f"boxplot_{metric}")
    if "cdf" in kinds:
        save(cdf_plot(groups, complementary=True, xlabel=metric).figure, f"ccdf_{metric}")
    if "raster" in kinds and events is not None and not events.empty:
        first = events[events["file_index"] == events["file_index"].min()]
        channels = sorted(first["channel"].unique())[:32]
        rows = [first.loc[first["channel"] == c, "onset"].to_numpy() for c in channels]
        save(raster_plot(rows, row_labels=[f"ch{c}" for c in channels]).figure, "raster")

    return written
