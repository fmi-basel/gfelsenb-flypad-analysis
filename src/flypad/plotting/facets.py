"""Faceted figures: one axis per facet — substrate or input file (design §10, M6).

When an experiment spans more than one substrate or recording, the per-condition box
plot, CCDF and cumulative time course read far better split into a row of subplots —
one axis per facet — sharing a y-axis and colour palette so conditions stay comparable
across facets. These builders wrap the single-axis plotters (``tilted_boxplot`` /
``ccdf_plot`` / ``shaded_lines``); the facet resolution itself lives in
:mod:`flypad.stats.grouping` so the statistics use exactly the same grouping.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.lines import Line2D

from flypad.plotting.boxplots import Palette, ci_plot, median_iqr_plot, tilted_boxplot
from flypad.plotting.cdf import ccdf_plot
from flypad.plotting.theme import GRAY, condition_palette, suptitle
from flypad.plotting.timecourses import shaded_lines

# Facet resolution is a pure data operation shared with the stats layer.
from flypad.stats.grouping import (
    FacetBy,
    file_facet_label,
    resolve_facets,
    substrate_facets,
    with_file_labels,
)
from flypad.stats.summaries import cumulative_timecourse_by_condition

FloatArray = npt.NDArray[np.float64]

# Re-exported for backwards compatibility (these now live in flypad.stats.grouping).
__all__ = [
    "FacetBy",
    "faceted_boxplot",
    "faceted_ccdf",
    "faceted_dashboard",
    "faceted_timecourse",
    "file_facet_label",
    "resolve_facets",
    "substrate_facets",
    "with_file_labels",
]


def _facet_row(n: int, *, width: float, height: float = 4.0) -> tuple[Any, list[Any]]:
    """A one-row grid of ``n`` subplots sharing the y-axis."""
    fig, axes = plt.subplots(1, n, figsize=(width * n, height), sharey=True, squeeze=False)
    return fig, list(axes[0])


def _condition_groups(per_fly: pd.DataFrame, metric: str, group_col: str) -> dict[str, FloatArray]:
    return {
        str(label): grp[metric].to_numpy(dtype=np.float64)
        for label, grp in per_fly.groupby(group_col, dropna=False)
    }


def _shared_legend(fig: Any, palette: Palette | None, labels: Sequence[str]) -> None:
    """A single figure-level legend mapping each condition to its palette colour."""
    pal = palette or {}
    handles = [Line2D([0], [0], color=pal.get(lbl, GRAY), lw=2.4, label=lbl) for lbl in labels]
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 4),
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, -0.02),
        )


def faceted_boxplot(
    per_fly: pd.DataFrame,
    metric: str,
    *,
    facet_col: str,
    values: Sequence[str],
    group_col: str = "condition_label",
    palette: Palette | None = None,
    ylabel: str | None = None,
    noun: str = "substrate",
) -> Any:
    """Per-condition box plot faceted into one axis per ``facet_col`` value."""
    fig, axes = _facet_row(len(values), width=4.6)
    for i, (ax, val) in enumerate(zip(axes, values, strict=True)):
        subset = per_fly[per_fly[facet_col].astype(str) == val]
        groups = _condition_groups(subset, metric, group_col)
        tilted_boxplot(groups, ax=ax, palette=palette, ylabel=ylabel if i == 0 else None)
        ax.set_title(val)
    suptitle(fig, f"{metric} by {noun}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def faceted_ccdf(
    per_fly: pd.DataFrame,
    metric: str,
    *,
    facet_col: str,
    values: Sequence[str],
    group_col: str = "condition_label",
    palette: Palette | None = None,
    xlabel: str | None = None,
    noun: str = "substrate",
) -> Any:
    """Per-condition CCDF faceted into one axis per ``facet_col`` value (shared legend)."""
    fig, axes = _facet_row(len(values), width=5.0)
    labels = sorted(_condition_groups(per_fly, metric, group_col))
    for ax, val in zip(axes, values, strict=True):
        subset = per_fly[per_fly[facet_col].astype(str) == val]
        groups = _condition_groups(subset, metric, group_col)
        ccdf_plot(groups, ax=ax, palette=palette, xlabel=xlabel or metric, legend=False)
        ax.set_title(val)
    _shared_legend(fig, palette, labels)
    suptitle(fig, f"{metric} CCDF by {noun}")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    return fig


def faceted_timecourse(
    events: pd.DataFrame,
    n_samples: int,
    *,
    facet_col: str,
    values: Sequence[str],
    group_col: str = "condition_label",
    sampling_rate_hz: int = 100,
    palette: Palette | None = None,
    noun: str = "substrate",
) -> Any:
    """Cumulative sip time course faceted into one axis per ``facet_col`` value."""
    fig, axes = _facet_row(len(values), width=6.0, height=3.4)
    labels = sorted(str(x) for x in events[group_col].dropna().unique())
    for ax, val in zip(axes, values, strict=True):
        subset = events[events[facet_col].astype(str) == val]
        series = cumulative_timecourse_by_condition(
            subset, n_samples, group_col=group_col, sampling_rate_hz=sampling_rate_hz
        )
        shaded_lines(series, ax=ax, palette=palette, legend=False)
        ax.set_title(val)
    _shared_legend(fig, palette, labels)
    suptitle(fig, f"cumulative sips by {noun}")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    return fig


def faceted_dashboard(
    per_fly: pd.DataFrame,
    metric: str,
    *,
    facet_col: str,
    values: Sequence[str],
    group_col: str = "condition_label",
    central: str = "median",
    tilt_deg: float = 0.0,
    palette: Palette | None = None,
    noun: str = "substrate",
    title: str | None = None,
) -> Any:
    """Standalone dashboard with one row of panels per facet (top → bottom).

    Each row mirrors :func:`~flypad.plotting.cdf.standalone_dashboard`: the per-fly box
    plot, a central-tendency summary (``central`` = ``"median"`` → median+IQR, ``"mean"``
    → mean+95% CI), and the metric's CCDF. The leftmost panel of each row is titled with
    the facet value; condition colours are shared across every row.
    """
    if palette is None:
        palette = condition_palette(_condition_groups(per_fly, metric, group_col).keys())
    n = len(values)
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 4.0 * n), squeeze=False)
    central_title = "mean ± 95% CI" if central == "mean" else "median ± IQR"
    for r, val in enumerate(values):
        subset = per_fly[per_fly[facet_col].astype(str) == val]
        groups = _condition_groups(subset, metric, group_col)
        tilted_boxplot(groups, ax=axes[r][0], palette=palette, tilt_deg=tilt_deg, ylabel=metric)
        axes[r][0].set_title(val)
        if central == "mean":
            ci_plot(groups, ax=axes[r][1])
        else:
            median_iqr_plot(groups, ax=axes[r][1])
        axes[r][1].set_title(central_title)
        axes[r][1].set_ylabel(metric)
        ccdf_plot(groups, ax=axes[r][2], palette=palette, xlabel=metric)
        axes[r][2].set_title("CCDF")
    suptitle(fig, title or f"{metric} dashboard by {noun}")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / (4.0 * n)))
    return fig
