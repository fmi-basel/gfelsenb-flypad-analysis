"""Per-fly / per-condition summaries and the tidy event table (design §10).

This is the keystone: detection (M3) + post-processing (M4) + the channel→condition
map (M4) collapse into a long ``per_fly`` table (one row per recorded channel) and a
``per_condition`` aggregate (mean/median/SEM/CI per metric) — the
``GetAnyEvents_ForLab_Excel_Dots`` equivalent. The experiment-level QC removals
(non-eaters, spill) are applied here, on top of the raw detection.

Times are reported in physical units: durations in milliseconds, latencies in
seconds, using ``sampling_rate_hz`` (100 Hz → 10 ms per sample).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats as scipy_stats

from flypad.config.models import NonEaters as NonEatersConfig
from flypad.detect.results import ChannelBouts, ChannelSips
from flypad.postprocess.bursts import ChannelBursts
from flypad.postprocess.quality import flag_non_eaters
from flypad.postprocess.transitions import classify_in_burst

#: Numeric per-fly metrics produced by :func:`channel_metrics`.
METRIC_COLUMNS = (
    "n_sips",
    "n_feeding_bursts",
    "n_sips_in_bursts",
    "n_isolated_sips",
    "n_activity_bouts",
    "total_sip_duration_ms",
    "mean_sip_duration_ms",
    "median_sip_duration_ms",
    "mean_burst_size",
    "mean_burst_duration_ms",
    "total_activity_duration_ms",
    "latency_first_sip_s",
    "latency_first_burst_s",
)

_META_COLUMNS = (
    "file_index",
    "file_name",
    "exp_file",
    "board_position",
    "channel",
    "condition",
    "condition_label",
    "condition_short",
    "sex",
    "substrate",
    "substrate_side",
    "substrate_label",
)


def _mean(arr: npt.NDArray[np.float64]) -> float:
    return float(arr.mean()) if arr.size else float("nan")


def channel_metrics(
    sips: ChannelSips,
    bouts: ChannelBouts,
    bursts: ChannelBursts,
    sampling_rate_hz: int,
) -> dict[str, float]:
    """Compute the per-channel feeding/activity metrics for one fly."""
    ms = 1000.0 / sampling_rate_hz
    sec = 1.0 / sampling_rate_hz
    dur = sips.durations.astype(np.float64)
    n_in_bursts = bursts.total_sips
    return {
        "n_sips": float(len(sips)),
        "n_feeding_bursts": float(len(bursts)),
        "n_sips_in_bursts": float(n_in_bursts),
        "n_isolated_sips": float(len(sips) - n_in_bursts),
        "n_activity_bouts": float(len(bouts)),
        "total_sip_duration_ms": float(dur.sum() * ms),
        "mean_sip_duration_ms": _mean(dur) * ms,
        "median_sip_duration_ms": float(np.median(dur)) * ms if dur.size else float("nan"),
        "mean_burst_size": _mean(bursts.n_sips.astype(np.float64)),
        "mean_burst_duration_ms": _mean(bursts.durations.astype(np.float64)) * ms,
        "total_activity_duration_ms": float(bouts.durations.sum() * ms),
        "latency_first_sip_s": float(sips.onsets[0]) * sec if len(sips) else float("nan"),
        "latency_first_burst_s": float(bursts.onsets[0]) * sec if len(bursts) else float("nan"),
    }


def _iter_channel_rows(
    channel_map: pd.DataFrame,
) -> Iterator[tuple[dict[str, object], int, int]]:
    """Yield ``(row_dict, file_index, channel)`` for each channel-map row."""
    for record in channel_map.to_dict("records"):
        yield record, int(record["file_index"]), int(record["channel"])


def per_fly_summary(
    sips_by_file: list[list[ChannelSips]],
    bouts_by_file: list[list[ChannelBouts]],
    bursts_by_file: list[list[ChannelBursts]],
    channel_map: pd.DataFrame,
    sampling_rate_hz: int,
) -> pd.DataFrame:
    """One row per recorded channel: metadata + per-fly metrics."""
    rows: list[dict[str, object]] = []
    for meta, fi, ch in _iter_channel_rows(channel_map):
        metrics = channel_metrics(
            sips_by_file[fi][ch], bouts_by_file[fi][ch], bursts_by_file[fi][ch], sampling_rate_hz
        )
        rows.append({**meta, **metrics})
    cols = [*(c for c in _META_COLUMNS if c in channel_map.columns), *METRIC_COLUMNS]
    frame = pd.DataFrame(rows, columns=cols)
    count_cols = [
        "n_sips",
        "n_feeding_bursts",
        "n_sips_in_bursts",
        "n_isolated_sips",
        "n_activity_bouts",
    ]
    frame[count_cols] = frame[count_cols].astype("int64")
    return frame


def build_event_table(
    sips_by_file: list[list[ChannelSips]],
    bursts_by_file: list[list[ChannelBursts]],
    channel_map: pd.DataFrame,
    sampling_rate_hz: int,
) -> pd.DataFrame:
    """Tidy long table — one row per sip, with metadata and burst membership."""
    ms = 1000.0 / sampling_rate_hz
    meta_cols = [c for c in _META_COLUMNS if c in channel_map.columns]
    rows: list[dict[str, object]] = []
    for meta, fi, ch in _iter_channel_rows(channel_map):
        sips = sips_by_file[fi][ch]
        n = len(sips)
        if n == 0:
            continue
        bursts = bursts_by_file[fi][ch]
        in_burst = classify_in_burst(n, bursts)
        burst_id = np.full(n, -1, dtype=np.int64)
        for b, (s, e) in enumerate(zip(bursts.sip_start, bursts.sip_end, strict=True)):
            burst_id[int(s) : int(e) + 1] = b
        meta_part = {c: meta[c] for c in meta_cols}
        for i in range(n):
            rows.append(
                {
                    **meta_part,
                    "sip_index": i,
                    "onset": int(sips.onsets[i]),
                    "offset": int(sips.offsets[i]),
                    "duration_ms": float(sips.durations[i]) * ms,
                    "in_burst": bool(in_burst[i]),
                    "burst_id": int(burst_id[i]),
                }
            )
    columns = [*meta_cols, "sip_index", "onset", "offset", "duration_ms", "in_burst", "burst_id"]
    return pd.DataFrame(rows, columns=columns)


def mark_non_eaters(per_fly: pd.DataFrame, non_eaters: NonEatersConfig) -> pd.DataFrame:
    """Add a boolean ``non_eater`` column (per file, by activity-bout count)."""
    out = per_fly.copy()
    out["non_eater"] = False
    for _, grp in out.groupby("file_index"):
        ordered = grp.sort_values("channel")
        mask = flag_non_eaters(
            ordered["n_activity_bouts"].to_numpy(),
            remove_global=non_eaters.remove_global,
            remove_substrate=non_eaters.remove_substrate,
            threshold=non_eaters.threshold_bouts,
        )
        out.loc[ordered.index, "non_eater"] = mask
    return out


def apply_qc_removal(
    per_fly: pd.DataFrame,
    flags: tuple[str, ...] = ("non_eater",),
) -> pd.DataFrame:
    """Drop rows flagged by any of ``flags`` (QC removal), reindexing the result."""
    present = [f for f in flags if f in per_fly.columns]
    if not present:
        return per_fly.reset_index(drop=True)
    drop = per_fly[present].any(axis=1)
    return per_fly[~drop].reset_index(drop=True)


def per_condition_summary(
    per_fly: pd.DataFrame,
    metrics: tuple[str, ...] | None = None,
    *,
    group_cols: tuple[str, ...] = ("condition", "condition_label"),
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Aggregate per-fly metrics by condition: n / mean / median / std / SEM / CI.

    Long format — one row per (group, metric). NaNs are ignored; ``n`` counts the
    finite values. The confidence interval is the Student-t interval on the mean.
    """
    metrics = metrics or METRIC_COLUMNS
    group_cols = tuple(c for c in group_cols if c in per_fly.columns)
    rows: list[dict[str, object]] = []
    grouped = per_fly.groupby(list(group_cols), dropna=False) if group_cols else [((), per_fly)]
    for key, grp in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_map = dict(zip(group_cols, key_tuple, strict=True))
        for metric in metrics:
            vals = grp[metric].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            n = vals.size
            mean = float(vals.mean()) if n else float("nan")
            std = float(vals.std(ddof=1)) if n > 1 else float("nan")
            sem = std / np.sqrt(n) if n > 1 else float("nan")
            if n > 1:
                half = float(scipy_stats.t.ppf(0.5 + ci_level / 2, n - 1)) * sem
            else:
                half = float("nan")
            rows.append(
                {
                    **key_map,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "median": float(np.median(vals)) if n else float("nan"),
                    "std": std,
                    "sem": sem,
                    "ci_low": mean - half,
                    "ci_high": mean + half,
                }
            )
    columns = [
        *group_cols,
        "metric",
        "n",
        "mean",
        "median",
        "std",
        "sem",
        "ci_low",
        "ci_high",
    ]
    return pd.DataFrame(rows, columns=columns)


def dots_table(
    per_fly: pd.DataFrame,
    metric: str,
    *,
    group_col: str = "condition_label",
) -> pd.DataFrame:
    """Wide "dots" table for dot/box plots: one column per group, per-fly values down.

    Columns are padded with NaN to equal length — the layout of MATLAB's
    ``ForLab_Excel_Dots`` export.
    """
    series = {
        str(label): grp[metric].to_numpy(dtype=np.float64)
        for label, grp in per_fly.groupby(group_col, dropna=False)
    }
    width = max((v.size for v in series.values()), default=0)
    padded = {k: np.concatenate([v, np.full(width - v.size, np.nan)]) for k, v in series.items()}
    return pd.DataFrame(padded)


def export_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write ``df`` by extension: ``.csv`` (default), ``.parquet``, or ``.xlsx``."""
    out = Path(path)
    suffix = out.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(out, index=False)
    elif suffix in (".xlsx", ".xls"):
        df.to_excel(out, index=False)  # needs openpyxl
    else:
        df.to_csv(out, index=False)
    return out


@dataclass
class ExperimentSummary:
    """The three keystone tables for one experiment."""

    events: pd.DataFrame
    per_fly: pd.DataFrame
    per_condition: pd.DataFrame


def summarize_experiment(
    sips_by_file: list[list[ChannelSips]],
    bouts_by_file: list[list[ChannelBouts]],
    bursts_by_file: list[list[ChannelBursts]],
    channel_map: pd.DataFrame,
    sampling_rate_hz: int,
    *,
    non_eaters: NonEatersConfig | None = None,
    ci_level: float = 0.95,
) -> ExperimentSummary:
    """Build the event table, per-fly table, and per-condition aggregate in one call.

    When ``non_eaters`` is given, non-eater channels are flagged and removed before
    aggregation (the experiment-level QC step).
    """
    events = build_event_table(sips_by_file, bursts_by_file, channel_map, sampling_rate_hz)
    per_fly = per_fly_summary(
        sips_by_file, bouts_by_file, bursts_by_file, channel_map, sampling_rate_hz
    )
    kept = per_fly
    if non_eaters is not None:
        kept = apply_qc_removal(mark_non_eaters(per_fly, non_eaters))
    per_condition = per_condition_summary(kept, ci_level=ci_level)
    return ExperimentSummary(events=events, per_fly=kept, per_condition=per_condition)
