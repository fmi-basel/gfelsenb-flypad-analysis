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

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from flypad import __version__
from flypad.config.models import Config
from flypad.datamodel import load_recording
from flypad.detect.results import ChannelBouts, ChannelSips, shift_bouts, shift_sips
from flypad.detect.run import detect_recording
from flypad.io.discovery import find_capacitance_files
from flypad.postprocess.bursts import ChannelBursts, detect_feeding_bursts
from flypad.postprocess.metadata import (
    apply_label_overrides,
    channel_condition_map_for_dir,
    channel_map_from_filenames,
)
from flypad.postprocess.quality import saturation_fraction, zero_fraction
from flypad.stats.comparisons import build_comparisons
from flypad.stats.summaries import (
    apply_qc_removal,
    build_event_table,
    mark_bad_channels,
    mark_non_eaters,
    per_condition_summary,
    per_fly_summary,
)

Progress = Callable[[str], None]
FloatArray = npt.NDArray[np.float64]

#: Tables written to a results directory; preferred read order for re-loading.
TABLE_NAMES = ("events", "per_fly", "per_condition", "comparisons")
_TABLE_FORMATS = ("parquet", "csv")


@dataclass
class DetectionResult:
    """Raw per-file detection output plus the channel→condition map.

    ``spill_by_file`` / ``zero_by_file`` hold the per-channel spill (saturated-sample)
    and zero-sample fractions used for spill/unconnected channel QC. ``n_samples`` is the
    *actual* recorded length (the shortest file, as MATLAB's ``Events.Dur``), which can be
    shorter than the configured ``acquisition.duration_samples``.
    """

    files: list[Path]
    sips_by_file: list[list[ChannelSips]]
    bouts_by_file: list[list[ChannelBouts]]
    bursts_by_file: list[list[ChannelBursts]]
    channel_map: pd.DataFrame
    spill_by_file: list[FloatArray]
    zero_by_file: list[FloatArray]
    n_samples: int = 0
    #: Pre-alignment sips in absolute file time — kept only when arena alignment ran, so
    #: the raster can show what the crop removed. ``None`` when nothing was aligned.
    absolute_sips_by_file: list[list[ChannelSips]] | None = None


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
    """Discover the channel→condition map, then apply the config label overrides."""
    channel_map = _discover_channel_map(data_dir, files, config)
    return apply_label_overrides(
        channel_map,
        conditions=config.metadata.conditions,
        substrates=config.metadata.substrates,
    )


def _discover_channel_map(
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


def _arena_starts(
    data_dir: str | Path, file_name: str, config: Config
) -> npt.NDArray[np.int64] | None:
    """Per-channel arena start samples for one recording, or ``None`` without a sidecar."""
    if not config.alignment.enabled:
        return None
    from flypad.io.timestamps import find_arena_fills
    from flypad.postprocess.alignment import channel_start_samples

    fills = find_arena_fills(data_dir, file_name)
    if fills is None or fills.empty:
        return None
    return channel_start_samples(
        fills, config.hardware.n_channels, config.metadata.channels_per_board_position
    )


def _raw_length(path: Path, config: Config) -> int:
    """Recording length in samples, from the file size alone (no read)."""
    itemsize = np.dtype(config.acquisition.dtype).itemsize
    on_disk = path.stat().st_size // (itemsize * config.hardware.n_channels)
    return int(min(on_disk, config.acquisition.duration_samples))


def _window_duration(
    starts_by_file: Sequence[npt.NDArray[np.int64] | None],
    lengths: Sequence[int],
    config: Config,
) -> int:
    """Common per-channel window: the configured one, else the longest all arenas allow."""
    from flypad.postprocess.alignment import available_duration

    if config.alignment.window_samples:
        return int(config.alignment.window_samples)
    zeros = np.zeros(config.hardware.n_channels, dtype=np.int64)
    return min(
        available_duration(s if s is not None else zeros, n)
        for s, n in zip(starts_by_file, lengths, strict=True)
    )


def detect_experiment(
    data_dir: str | Path,
    config: Config,
    *,
    progress: Progress | None = None,
) -> DetectionResult:
    """Discover recordings and run detection + feeding-burst grouping on each."""
    data_dir = Path(data_dir).expanduser()
    files = find_capacitance_files(data_dir)
    if not files:
        raise FileNotFoundError(f"no CapacitanceData* files in {data_dir}")

    from flypad.postprocess.alignment import align_channels

    # Cheap pre-pass (file sizes + the small timestamp sidecars) so the common analysis
    # window is known before any raw data is read — every output can then be computed
    # over that window in a single pass.
    lengths = [_raw_length(p, config) for p in files]
    starts_by_file = [_arena_starts(data_dir, p.name, config) for p in files]
    aligned = config.alignment.enabled and any(s is not None for s in starts_by_file)
    duration = _window_duration(starts_by_file, lengths, config) if aligned else 0
    if aligned and duration <= 0:
        _emit(progress, "arena alignment skipped (no usable common window)")
        aligned = False
    if aligned:
        _emit(progress, f"align arenas → {duration} samples per channel")

    zeros = np.zeros(config.hardware.n_channels, dtype=np.int64)
    sips_by_file: list[list[ChannelSips]] = []
    bouts_by_file: list[list[ChannelBouts]] = []
    bursts_by_file: list[list[ChannelBursts]] = []
    spill_by_file: list[FloatArray] = []
    zero_by_file: list[FloatArray] = []
    absolute_sips_by_file: list[list[ChannelSips]] = []
    for i, path in enumerate(files, 1):
        _emit(progress, f"detect [{i}/{len(files)}] {path.name}")
        recording = load_recording(path, config)
        raw = np.asarray(recording.capacitance.values)
        file_starts = starts_by_file[i - 1]
        starts = (file_starts if file_starts is not None else zeros) if aligned else None
        window = duration if aligned else None
        # QC fractions honour the same window, so the pre-fill handling period cannot
        # flag a channel that is clean while its fly is actually being recorded.
        spill_by_file.append(
            saturation_fraction(
                raw,
                config.quality_control.spill_saturation_value,
                starts=starts,
                duration=window,
            )
        )
        zero_by_file.append(zero_fraction(raw, starts=starts, duration=window))
        result = detect_recording(recording.capacitance, config)
        # Detection indexes the de-trended trace, which "crop" edge handling starts
        # crop_offset samples into the file; shift back so every downstream index
        # (tables, figures, arena-fill alignment) is a raw-file sample position.
        sips = [shift_sips(s, result.crop_offset) for s in result.sips]
        bouts = [shift_bouts(b, result.crop_offset) for b in result.bouts]
        if aligned and starts is not None:
            absolute_sips_by_file.append(sips)  # keep the uncropped view for the raster
            sips, bouts = align_channels(sips, bouts, starts, duration)
        sips_by_file.append(sips)
        bouts_by_file.append(bouts)

    n_samples = duration if aligned else (min(lengths) if lengths else 0)
    for sips in sips_by_file:
        _, bursts = detect_feeding_bursts(sips, config.feeding_bursts)
        bursts_by_file.append(bursts)

    _emit(progress, "build channel→condition map")
    channel_map = _resolve_channel_map(data_dir, files, config)
    return DetectionResult(
        files=files,
        sips_by_file=sips_by_file,
        bouts_by_file=bouts_by_file,
        bursts_by_file=bursts_by_file,
        channel_map=channel_map,
        spill_by_file=spill_by_file,
        zero_by_file=zero_by_file,
        n_samples=n_samples,
        absolute_sips_by_file=absolute_sips_by_file or None,
    )


def absolute_onsets(detection: DetectionResult) -> pd.DataFrame | None:
    """Minimal ``file_index / file_name / channel / onset`` frame in absolute file time.

    Built from the pre-alignment sips so the raster can show the uncropped recording next
    to the aligned one. ``None`` when no alignment ran (the ``events`` table is already
    in absolute time).
    """
    if not detection.absolute_sips_by_file:
        return None
    names = (
        detection.channel_map.drop_duplicates("file_index").set_index("file_index")["file_name"]
        if "file_name" in detection.channel_map.columns
        else None
    )
    blocks: list[pd.DataFrame] = []
    for file_index, per_channel in enumerate(detection.absolute_sips_by_file):
        for channel, sips in enumerate(per_channel):
            if not len(sips):
                continue
            blocks.append(
                pd.DataFrame(
                    {
                        "file_index": file_index,
                        "file_name": (names.get(file_index, "") if names is not None else ""),
                        "channel": channel,
                        "onset": np.asarray(sips.onsets, dtype=np.int64),
                    }
                )
            )
    if not blocks:
        return None
    return pd.concat(blocks, ignore_index=True)


def build_tables(detection: DetectionResult, config: Config) -> dict[str, pd.DataFrame]:
    """Build the ``events`` / ``per_fly`` / ``per_condition`` tables.

    ``per_fly`` keeps every channel, carrying the ``non_eater`` / ``spill`` /
    ``unconnected`` QC flags plus the ``spill_fraction`` / ``zero_fraction`` diagnostics;
    ``per_condition`` aggregates only the kept (non-removed) flies, and ``comparisons``
    holds the pairwise permutation tests (faceted per ``plotting.facet_by``).
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
        spill_by_file=detection.spill_by_file,
        zero_by_file=detection.zero_by_file,
    )
    per_fly = mark_non_eaters(per_fly, config.non_eaters)
    per_fly = mark_bad_channels(per_fly, config.quality_control)
    kept = apply_qc_removal(per_fly)
    per_condition = per_condition_summary(kept, ci_level=config.stats.ci_level)
    comparisons = build_comparisons(kept, config)
    return {
        "events": events,
        "per_fly": per_fly,
        "per_condition": per_condition,
        "comparisons": comparisons,
    }


def write_tables(
    tables: dict[str, pd.DataFrame],
    out_dir: str | Path,
    *,
    formats: Sequence[str] = ("parquet", "csv"),
) -> list[Path]:
    """Write each table in the requested formats (only ``parquet``/``csv`` apply)."""
    out = Path(out_dir).expanduser()
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
    out = Path(out_dir).expanduser()
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


@dataclass
class ExperimentResult:
    """Everything a completed run produced: tables, written paths, headline counts."""

    out_dir: Path
    files: list[Path]
    tables: dict[str, pd.DataFrame]
    written: list[Path]
    n_sips: int
    n_flies_kept: int
    n_samples: int

    @property
    def n_files(self) -> int:
        return len(self.files)

    @property
    def per_fly(self) -> pd.DataFrame:
        return self.tables["per_fly"]

    @property
    def per_condition(self) -> pd.DataFrame:
        return self.tables["per_condition"]


def run_experiment(
    data_dir: str | Path,
    config: Config,
    out_dir: str | Path,
    *,
    make_plots: bool = True,
    command: str = "run",
    progress: Progress | None = None,
) -> ExperimentResult:
    """Run the full pipeline: detect → tables → figures → write, provenance included.

    The single orchestration behind both ``flypad run`` and the GUI, so a results
    directory is the same whichever produced it. ``command`` is recorded in
    ``run_info.json`` to say which one did (``"run"`` / ``"gui"``).

    Keeping this in one place is deliberate: when the CLI and the GUI each had their own
    copy of the sequence, the GUI's silently omitted :func:`write_provenance`, which also
    made ``flypad stats`` fall back to a default config on those directories (#1).
    """
    detection = detect_experiment(data_dir, config, progress=progress)
    _emit(progress, "building tables")
    tables = build_tables(detection, config)
    written = write_tables(tables, out_dir, formats=config.output.formats)
    if make_plots and config.plotting.enabled:
        written += render_figures(
            tables["per_fly"],
            tables["events"],
            out_dir,
            config,
            comparisons=tables["comparisons"],
            n_samples=detection.n_samples,
            data_dir=data_dir,
            events_absolute=absolute_onsets(detection),
            progress=progress,
        )
    kept = len(apply_qc_removal(tables["per_fly"]))
    n_sips = len(tables["events"])
    written += write_provenance(
        out_dir,
        config,
        files=detection.files,
        command=command,
        extra={
            "n_sips": n_sips,
            "n_flies_kept": kept,
            "n_samples_recorded": detection.n_samples,
        },
    )
    return ExperimentResult(
        out_dir=Path(out_dir),
        files=detection.files,
        tables=tables,
        written=written,
        n_sips=n_sips,
        n_flies_kept=kept,
        n_samples=detection.n_samples,
    )


def read_table(results_dir: str | Path, name: str) -> pd.DataFrame:
    """Read a saved table by name, preferring parquet over csv."""
    out = Path(results_dir).expanduser()
    for fmt in _TABLE_FORMATS:
        path = out / f"{name}.{fmt}"
        if path.exists():
            return pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path)
    raise FileNotFoundError(f"no {name}.(parquet|csv) in {results_dir}")


def _kept(per_fly: pd.DataFrame) -> pd.DataFrame:
    return apply_qc_removal(per_fly)


def _relabel_conditions(df: pd.DataFrame, column: str, mapping: dict[str, str]) -> pd.DataFrame:
    """Return a copy of ``df`` with ``column`` values remapped via ``mapping``.

    Values absent from ``mapping`` are kept as-is (matched on their string form, so an
    integer ``condition`` column can be renamed too). Used for the plot-only
    ``config.plotting.condition_labels`` display rename.
    """
    if not mapping or column not in df.columns:
        return df
    out = df.copy()
    out[column] = out[column].map(lambda v: mapping.get(str(v), v))
    return out


def _condition_pairs(
    comparisons: pd.DataFrame, metric: str, strata: str, labels_map: dict[str, str]
) -> list[tuple[str, str, float]]:
    """Condition ``(a, b, p_adjusted)`` triples for one ``strata`` of the comparisons table.

    Labels pass through the plot display rename so brackets match the axis tick labels.
    """
    sub = comparisons[
        (comparisons["metric"] == metric)
        & (comparisons["contrast"] == "condition")
        & (comparisons["strata"].astype(str) == str(strata))
    ]

    def disp(x: object) -> str:
        return labels_map.get(str(x), str(x))

    return [(disp(r.group_a), disp(r.group_b), float(r.p_adjusted)) for r in sub.itertuples()]


def _arena_fill_markers(
    data_dir: str | Path | None,
    file_events: pd.DataFrame,
    config: Config,
    duration: int | None = None,
) -> list[tuple[float, float, float]] | None:
    """Red-line markers bounding each channel's analysis window in the raster.

    Every arena contributes a **start** and an **end** marker spanning only its own
    channels. In arena-aligned coordinates all windows are ``[0, duration)``, so the pair
    becomes two full-height lines delimiting the analysed span; in absolute time they form
    the loading staircase and its parallel end staircase. ``None`` without a sidecar.
    """
    if data_dir is None or "file_name" not in file_events.columns or file_events.empty:
        return None
    from flypad.io.timestamps import find_arena_fills

    fills = find_arena_fills(data_dir, str(file_events["file_name"].iloc[0]))
    if fills is None or fills.empty:
        return None
    step = config.metadata.channels_per_board_position
    n_channels = config.hardware.n_channels
    if config.alignment.enabled:
        if not duration:
            return None
        return [(0.0, 0.0, float(n_channels - 1)), (float(duration), 0.0, float(n_channels - 1))]

    from flypad.postprocess.alignment import arena_start_samples

    # Unaligned: every channel is analysed over the whole recording, so only the arena
    # starts carry information — an "end" would just be the right-hand axis limit.
    n_positions = -(-n_channels // step)
    starts = arena_start_samples(fills, n_positions)
    markers: list[tuple[float, float, float]] = []
    for position, start in sorted(starts.items()):
        lo = float((position - 1) * step)
        hi = float(min((position - 1) * step + step - 1, n_channels - 1))
        markers.append((float(start), lo, hi))
    return markers or None


def _absolute_rows(
    events_absolute: pd.DataFrame | None, file_index: int, channels: Sequence[int]
) -> list[Any] | None:
    """Per-channel absolute onsets for one recording, ordered like ``channels``."""
    if events_absolute is None or events_absolute.empty:
        return None
    one = events_absolute[events_absolute["file_index"] == file_index]
    if one.empty:
        return None
    by_channel = {int(c): g["onset"].to_numpy() for c, g in one.groupby("channel")}
    return [by_channel.get(c, []) for c in channels]


def _absolute_fill_markers(
    data_dir: str | Path | None, file_events: pd.DataFrame, config: Config
) -> list[tuple[float, float, float]] | None:
    """Arena-start markers in absolute time, regardless of the alignment setting."""
    unaligned = config.model_copy(update={"alignment": config.alignment.model_copy()})
    unaligned.alignment.enabled = False
    return _arena_fill_markers(data_dir, file_events, unaligned)


def _minutes(samples: int, rate: int) -> str:
    return f"{samples / rate / 60:.0f} min per channel"


def _slug(text: str) -> str:
    """Filesystem-safe stem fragment (``2026-07-09 10:01:52`` -> ``2026-07-09_10-01-52``)."""
    cleaned = "".join(c if c.isalnum() else ("_" if c == " " else "-") for c in text.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-_") or "file"


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
    comparisons: pd.DataFrame | None = None,
    n_samples: int | None = None,
    data_dir: str | Path | None = None,
    events_absolute: pd.DataFrame | None = None,
    progress: Progress | None = None,
) -> list[Path]:
    """Render the requested figure kinds into ``out_dir/figures``.

    ``data_dir`` is the *input* folder; when given, each raster overlays that recording's
    manual arena-fill timestamps (``timestamps_manual_*.csv``) as red vertical markers.

    When ``comparisons`` (from :func:`flypad.stats.build_comparisons`) is supplied and
    ``config.plotting.annotate_stats`` is on, the box plot's condition comparisons are
    drawn as significance brackets. ``n_samples`` is the recording length used for the
    time-course axis; pass the *measured* length (``DetectionResult.n_samples``) so the
    axis ends with the data rather than at a longer configured duration.
    """
    from flypad.plotting import (
        ccdf_plot,
        condition_palette,
        faceted_boxplot,
        faceted_ccdf,
        faceted_dashboard,
        faceted_timecourse,
        metric_label,
        raster_panels,
        raster_plot,
        resolve_facets,
        save_figure,
        set_theme,
        shaded_lines,
        standalone_dashboard,
        substrate_comparison,
        tilted_boxplot,
        with_file_labels,
    )
    from flypad.stats import (
        cumulative_timecourse_by_condition,
        is_two_choice,
        ordered_condition_labels,
    )

    set_theme()
    figdir = Path(out_dir).expanduser() / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    vector = config.plotting.vector_format
    formats = ("png",) if vector == "none" else ("png", vector)
    dpi = config.plotting.dpi
    rate = config.hardware.sampling_rate_hz
    kept = _kept(per_fly)
    group_col = "condition_label" if "condition_label" in kept.columns else "condition"
    # Presentation-only condition rename (plots keep display names; tables are untouched).
    labels_map = config.plotting.condition_labels
    if labels_map:
        kept = _relabel_conditions(kept, group_col, labels_map)
        per_fly = _relabel_conditions(per_fly, group_col, labels_map)
        if events is not None:
            events = _relabel_conditions(events, group_col, labels_map)
    # Conditions are laid out in experiment order (from the condition number), not
    # alphabetically, so a starvation series reads in its intended sequence.
    order = ordered_condition_labels(kept, group_col)
    by_label = {
        str(label): grp[metric].to_numpy() for label, grp in kept.groupby(group_col, dropna=False)
    }
    groups = {label: by_label[label] for label in order if label in by_label}
    palette = condition_palette(groups.keys(), sort=False)
    ylabel = metric_label(metric)
    # Box plot / CCDF / time course are split into one axis per facet as configured by
    # ``plotting.facet_by`` (substrate | file | none); single-axis when <2 facets exist.
    if config.plotting.facet_by == "file":
        kept = with_file_labels(kept)
        if events is not None:
            events = with_file_labels(events)
    facet_col, facet_values, facet_noun = resolve_facets(kept, config.plotting.facet_by)
    comp = (
        comparisons
        if config.plotting.annotate_stats and comparisons is not None and not comparisons.empty
        else None
    )
    # Condition-comparison significance brackets for the box plot / dashboard box panel.
    box_pairs = None if comp is None else _condition_pairs(comp, metric, "all", labels_map)
    box_pairs_by_facet = (
        {v: _condition_pairs(comp, metric, v, labels_map) for v in facet_values}
        if comp is not None and facet_col is not None
        else None
    )
    # Time-course length: prefer the measured recording, then the events, then the config
    # (which may be a longer nominal duration and would pad the axis with a flat tail).
    if n_samples is None and events is not None and not events.empty:
        n_samples = int(events["onset"].max()) + 1
    duration = n_samples or config.acquisition.duration_samples
    written: list[Path] = []
    # Bracket / scale styling shared by the box plot and the dashboard's box panel.
    box_style: dict[str, Any] = {
        "yscale": config.plotting.y_scale,
        "only_significant": config.plotting.annotate_only_significant,
        "show_pvalues": config.plotting.annotate_p_values,
    }
    # Facet arrangement, shared by every faceted figure (the dashboard is always stacked).
    facet_style: dict[str, Any] = {
        "layout": config.plotting.facet_layout,
        "share_y": config.plotting.facet_share_y,
    }

    def save(fig: object, stem: str) -> None:
        _emit(progress, f"figure {stem}")
        written.extend(save_figure(fig, figdir / stem, formats=formats, dpi=dpi, close=True))

    if "dashboard" in kinds:
        if facet_col is not None:
            dash = faceted_dashboard(
                kept,
                metric,
                facet_col=facet_col,
                values=facet_values,
                group_col=group_col,
                palette=palette,
                central="median",
                noun=facet_noun,
                annotations_by_facet=box_pairs_by_facet,
                share_y=config.plotting.facet_share_y,
                **box_style,
            )
        else:
            dash = standalone_dashboard(kept, metric, central="median", annotations=box_pairs)
        save(dash, f"dashboard_{metric}")
    if "boxplot" in kinds:
        if facet_col is not None:
            fig = faceted_boxplot(
                kept,
                metric,
                facet_col=facet_col,
                values=facet_values,
                group_col=group_col,
                palette=palette,
                ylabel=ylabel,
                noun=facet_noun,
                annotations_by_facet=box_pairs_by_facet,
                **facet_style,
                **box_style,
            )
        else:
            fig = tilted_boxplot(
                groups, palette=palette, ylabel=ylabel, annotations=box_pairs, **box_style
            ).figure
        save(fig, f"boxplot_{metric}")
    if "cdf" in kinds:
        if facet_col is not None:
            fig = faceted_ccdf(
                kept,
                metric,
                facet_col=facet_col,
                values=facet_values,
                group_col=group_col,
                palette=palette,
                xlabel=ylabel,
                noun=facet_noun,
                **facet_style,
            )
        else:
            fig = ccdf_plot(groups, palette=palette, xlabel=ylabel).figure
        save(fig, f"ccdf_{metric}")
    if "substrate" in kinds and is_two_choice(kept):
        save(substrate_comparison(kept, metric, ylabel=ylabel).figure, f"substrate_{metric}")
    if "timecourse" in kinds and events is not None and not events.empty:
        if facet_col is not None:
            fig = faceted_timecourse(
                events,
                duration,
                facet_col=facet_col,
                values=facet_values,
                group_col=group_col,
                sampling_rate_hz=rate,
                palette=palette,
                noun=facet_noun,
                **facet_style,
            )
        else:
            series = cumulative_timecourse_by_condition(
                events,
                duration,
                group_col=group_col,
                sampling_rate_hz=rate,
            )
            ordered = {k: series[k] for k in order if k in series}
            fig = shaded_lines(ordered, palette=palette, direct_labels=True).figure
        save(fig, "timecourse")
    if "raster" in kinds and events is not None and not events.empty:
        # One full-plate raster per recording, titled and named by its timestamp: a raster
        # only ever shows a single plate, so a combined figure would misrepresent the run.
        ev = events if "file_label" in events.columns else with_file_labels(events)
        channels = list(range(config.hardware.n_channels))
        for fi in sorted(ev["file_index"].unique()):
            one = ev[ev["file_index"] == fi]
            label = str(one["file_label"].iloc[0]) if "file_label" in one.columns else f"file {fi}"
            onsets_by_ch = {int(c): g["onset"].to_numpy() for c, g in one.groupby("channel")}
            rows = [onsets_by_ch.get(c, []) for c in channels]
            pf0 = (
                per_fly[per_fly["file_index"] == fi].drop_duplicates("channel").set_index("channel")
            )
            row_conditions = (
                [str(pf0.loc[c, group_col]) if c in pf0.index else "" for c in channels]
                if group_col in pf0.columns
                else None
            )
            aligned_fills = _arena_fill_markers(data_dir, one, config, duration)
            abs_rows = _absolute_rows(events_absolute, fi, channels)
            if abs_rows is None:
                raster = raster_plot(
                    rows,
                    row_conditions=row_conditions,
                    palette=palette,
                    sampling_rate_hz=rate,
                    fills=aligned_fills,
                )
                raster.set_title(label)
                fig = raster.figure
            else:
                # Side by side: the uncropped recording (with each arena's fill time) next
                # to the arena-aligned window that the analysis actually uses.
                fig = raster_panels(
                    [
                        (
                            "absolute time (uncropped)",
                            abs_rows,
                            _absolute_fill_markers(data_dir, one, config),
                        ),
                        (
                            f"aligned to arena fill ({_minutes(duration, rate)})",
                            rows,
                            aligned_fills,
                        ),
                    ],
                    row_conditions=row_conditions,
                    palette=palette,
                    sampling_rate_hz=rate,
                    suptitle_text=label,
                )
            save(fig, f"raster_{_slug(label)}")

    return written
