"""Faceted figures: one axis per facet — substrate or input file (design §10, M6).

When an experiment spans more than one substrate or recording, the per-condition box
plot, CCDF and cumulative time course read far better split into a row of subplots —
one axis per facet — sharing a y-axis and colour palette so conditions stay comparable
across facets. These builders wrap the single-axis plotters (``tilted_boxplot`` /
``ccdf_plot`` / ``shaded_lines``); the facet resolution itself lives in
:mod:`flypad.stats.grouping` so the statistics use exactly the same grouping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.lines import Line2D

from flypad.plotting.boxplots import (
    Palette,
    annotate_significance,
    ci_plot,
    median_iqr_plot,
    tilted_boxplot,
)
from flypad.plotting.cdf import ccdf_plot
from flypad.plotting.labels import metric_label
from flypad.plotting.theme import GRAY, condition_palette, suptitle
from flypad.plotting.timecourses import shaded_lines

# Facet resolution is a pure data operation shared with the stats layer.
from flypad.stats.grouping import (
    FacetBy,
    file_facet_label,
    ordered_condition_labels,
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


def _facet_row(
    n: int, *, width: float, height: float = 4.0, sharex: bool = False
) -> tuple[Any, list[Any]]:
    """A one-row grid of ``n`` subplots sharing the y-axis (and optionally the x-axis)."""
    fig, axes = plt.subplots(
        1, n, figsize=(width * n, height), sharey=True, sharex=sharex, squeeze=False
    )
    return fig, list(axes[0])


def _condition_groups(
    per_fly: pd.DataFrame,
    metric: str,
    group_col: str,
    order: Sequence[str] | None = None,
) -> dict[str, FloatArray]:
    """Metric values per condition, in experiment order (not alphabetical).

    ``order`` pins the sequence — pass the experiment-wide label order so every facet
    lays its categories out the same way, including facets missing some conditions.
    """
    by_label = {
        str(label): np.asarray(grp[metric].to_numpy(dtype=np.float64))
        for label, grp in per_fly.groupby(group_col, dropna=False)
    }
    labels = list(order) if order is not None else ordered_condition_labels(per_fly, group_col)
    empty = np.asarray([], dtype=np.float64)
    return {label: by_label.get(label, empty) for label in labels if label in by_label or order}


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
    annotations_by_facet: Mapping[str, Sequence[tuple[str, str, float]]] | None = None,
    yscale: str = "linear",
    only_significant: bool = False,
    show_pvalues: bool = False,
) -> Any:
    """Per-condition box plot faceted into one axis per ``facet_col`` value.

    ``annotations_by_facet`` maps each facet value to its list of
    ``(label_a, label_b, p_value)`` pairwise comparisons, drawn as significance brackets;
    with a shared y-axis the brackets are placed from the tallest facet so they align.
    """
    fig, axes = _facet_row(len(values), width=4.6)
    order = ordered_condition_labels(per_fly, group_col)
    drawn: list[tuple[Any, str, dict[str, FloatArray]]] = []
    for i, (ax, val) in enumerate(zip(axes, values, strict=True)):
        subset = per_fly[per_fly[facet_col].astype(str) == val]
        groups = _condition_groups(subset, metric, group_col, order)
        tilted_boxplot(
            groups, ax=ax, palette=palette, ylabel=ylabel if i == 0 else None, yscale=yscale
        )
        ax.set_title(val)
        drawn.append((ax, val, groups))
    if annotations_by_facet:
        _annotate_shared(
            drawn,
            annotations_by_facet,
            only_significant=only_significant,
            show_pvalues=show_pvalues,
        )
    suptitle(fig, f"{metric_label(metric)} by {noun}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _annotate_shared(
    drawn: Sequence[tuple[Any, str, dict[str, FloatArray]]],
    annotations_by_facet: Mapping[str, Sequence[tuple[str, str, float]]],
    *,
    only_significant: bool = False,
    show_pvalues: bool = False,
) -> None:
    """Draw brackets on sharey facets, anchored to the tallest facet so they line up."""
    tops = [float(np.nanmax(a)) for _ax, _v, groups in drawn for a in groups.values() if a.size]
    if not tops:
        return
    data_top = max(tops)
    for ax, val, groups in drawn:
        pairs = annotations_by_facet.get(val)
        if pairs:
            annotate_significance(
                ax,
                pairs,
                {label: i for i, label in enumerate(groups)},
                data_top=data_top,
                only_significant=only_significant,
                show_pvalues=show_pvalues,
            )


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
    """Per-condition CCDF faceted into one axis per ``facet_col`` value (shared legend).

    The x-axis is shared across facets: with independent ranges two panels can look alike
    while covering very different value spans.
    """
    fig, axes = _facet_row(len(values), width=5.0, sharex=True)
    order = ordered_condition_labels(per_fly, group_col)
    labels = list(_condition_groups(per_fly, metric, group_col, order))
    for ax, val in zip(axes, values, strict=True):
        subset = per_fly[per_fly[facet_col].astype(str) == val]
        groups = _condition_groups(subset, metric, group_col, order)
        ccdf_plot(groups, ax=ax, palette=palette, xlabel=xlabel or metric, legend=False)
        ax.set_title(val)
    _shared_legend(fig, palette, labels)
    suptitle(fig, f"{metric_label(metric)} — CCDF by {noun}")
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
    labels = ordered_condition_labels(events, group_col)
    for ax, val in zip(axes, values, strict=True):
        subset = events[events[facet_col].astype(str) == val]
        series = cumulative_timecourse_by_condition(
            subset, n_samples, group_col=group_col, sampling_rate_hz=sampling_rate_hz
        )
        ordered = {lbl: series[lbl] for lbl in labels if lbl in series}
        shaded_lines(ordered, ax=ax, palette=palette, legend=False)
        ax.set_title(val)
    _shared_legend(fig, palette, labels)
    suptitle(fig, f"Cumulative sips by {noun}")
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
    annotations_by_facet: Mapping[str, Sequence[tuple[str, str, float]]] | None = None,
    yscale: str = "linear",
    only_significant: bool = False,
    show_pvalues: bool = False,
) -> Any:
    """Standalone dashboard with one row of panels per facet (top → bottom).

    Each row mirrors :func:`~flypad.plotting.cdf.standalone_dashboard`: the per-fly box
    plot, a central-tendency summary (``central`` = ``"median"`` → median+IQR, ``"mean"``
    → mean+95% CI), and the metric's CCDF. The leftmost panel of each row is titled with
    the facet value; condition colours are shared across every row. Each *column* shares a
    y-axis so rows are directly comparable, the legend is drawn once, and
    ``annotations_by_facet`` puts significance brackets on each row's box panel.
    """
    order = ordered_condition_labels(per_fly, group_col)
    if palette is None:
        palette = condition_palette(order, sort=False)
    ann = annotations_by_facet or {}
    n = len(values)
    # sharey="col": rows are stacked for comparison, so each column must share a scale.
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 4.0 * n), squeeze=False, sharey="col")
    central_title = "mean ± 95% CI" if central == "mean" else "median ± IQR"
    label = metric_label(metric)
    box_rows: list[tuple[Any, str, dict[str, FloatArray]]] = []
    for r, val in enumerate(values):
        subset = per_fly[per_fly[facet_col].astype(str) == val]
        groups = _condition_groups(subset, metric, group_col, order)
        tilted_boxplot(
            groups, ax=axes[r][0], palette=palette, tilt_deg=tilt_deg, ylabel=label, yscale=yscale
        )
        axes[r][0].set_title(val)
        box_rows.append((axes[r][0], val, groups))
        if central == "mean":
            ci_plot(groups, ax=axes[r][1])
        else:
            median_iqr_plot(groups, ax=axes[r][1])
        axes[r][1].set_title(central_title)
        axes[r][1].set_ylabel(label)
        # one legend for the whole figure: the CCDF panels all share the palette
        ccdf_plot(groups, ax=axes[r][2], palette=palette, xlabel=label, legend=r == 0)
        axes[r][2].set_title("CCDF")
    if ann:
        _annotate_shared(
            box_rows, ann, only_significant=only_significant, show_pvalues=show_pvalues
        )
    suptitle(fig, title or f"{label} dashboard by {noun}")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / (4.0 * n)))
    return fig
