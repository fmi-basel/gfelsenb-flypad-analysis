"""Faceted figures: one axis per facet — substrate or input file (design §10, M6).

When an experiment spans more than one substrate or recording, the per-condition box
plot, CCDF and cumulative time course read far better split into a strip of subplots —
one axis per facet, stacked top-to-bottom by default (``config.plotting.facet_layout``)
and sharing a colour palette so conditions stay recognisable across facets. Each facet
scales to its own data unless ``share_y`` ties them together: one shared scale makes
magnitudes comparable by eye, but flattens a facet the flies barely fed on. These
builders wrap the single-axis plotters (``tilted_boxplot`` / ``ccdf_plot`` /
``shaded_lines``); the facet resolution itself lives in :mod:`flypad.stats.grouping` so
the statistics use exactly the same grouping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

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

#: How the facets are arranged (see ``config.plotting.facet_layout``).
FacetLayout = Literal["rows", "columns"]

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


def _facet_strip(
    n: int,
    *,
    width: float,
    height: float = 4.0,
    sharex: bool = False,
    sharey: bool = False,
    layout: FacetLayout = "rows",
) -> tuple[Any, list[Any]]:
    """A strip of ``n`` subplots: stacked top→bottom (``rows``) or side by side (``columns``).

    ``sharey`` / ``sharex`` tie the facets to a single scale. Both default to off, so each
    facet autoscales to its own data — a substrate the flies barely touch stays legible
    beside one they feed on heavily, which a shared scale would flatten to a line.
    """
    stacked = layout == "rows"
    nrows, ncols = (n, 1) if stacked else (1, n)
    figsize = (width, height * n) if stacked else (width * n, height)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, sharey=sharey, sharex=sharex, squeeze=False
    )
    return fig, list(axes[:, 0] if stacked else axes[0])


def _label_bottom_only(axes: Sequence[Any]) -> None:
    """Keep the x-label on the bottom axis of a stacked, x-shared strip.

    ``plt.subplots(sharex=True)`` hides the inner tick labels, but the single-axis
    plotters set their x-label afterwards, which would leave every facet but the bottom
    one captioning tick numbers it does not draw.
    """
    for ax in axes[:-1]:
        ax.set_xlabel("")


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
    layout: FacetLayout = "rows",
    share_y: bool = False,
) -> Any:
    """Per-condition box plot faceted into one axis per ``facet_col`` value.

    ``annotations_by_facet`` maps each facet value to its list of
    ``(label_a, label_b, p_value)`` pairwise comparisons, drawn as significance brackets.
    With ``share_y`` the brackets are placed from the tallest facet so they align; with
    independent axes each facet anchors to its own data.
    """
    stacked = layout == "rows"
    # Stacked facets repeat one categorical x-axis, so it is shared and drawn once, under
    # the bottom facet. Side-by-side facets each need their own condition labels.
    fig, axes = _facet_strip(len(values), width=4.6, sharex=stacked, sharey=share_y, layout=layout)
    order = ordered_condition_labels(per_fly, group_col)
    drawn: list[tuple[Any, str, dict[str, FloatArray]]] = []
    for i, (ax, val) in enumerate(zip(axes, values, strict=True)):
        subset = per_fly[per_fly[facet_col].astype(str) == val]
        groups = _condition_groups(subset, metric, group_col, order)
        # Every facet carries the unit unless they all share one y-axis, where the
        # leftmost label speaks for the row.
        show_label = stacked or not share_y or i == 0
        tilted_boxplot(
            groups, ax=ax, palette=palette, ylabel=ylabel if show_label else None, yscale=yscale
        )
        ax.set_title(val)
        drawn.append((ax, val, groups))
    if stacked:
        _label_bottom_only(axes)
    if annotations_by_facet:
        _annotate_shared(
            drawn,
            annotations_by_facet,
            share_y=share_y,
            only_significant=only_significant,
            show_pvalues=show_pvalues,
        )
    suptitle(fig, f"{metric_label(metric)} by {noun}")
    fig.tight_layout(rect=(0, 0, 1, 0.94 if not stacked else 1 - 0.3 / (4.0 * len(values))))
    return fig


def _facet_top(groups: Mapping[str, FloatArray]) -> float | None:
    """Tallest data point across a facet's conditions, or ``None`` when it has no data."""
    tops = [float(np.nanmax(a)) for a in groups.values() if a.size]
    return max(tops) if tops else None


def _annotate_shared(
    drawn: Sequence[tuple[Any, str, dict[str, FloatArray]]],
    annotations_by_facet: Mapping[str, Sequence[tuple[str, str, float]]],
    *,
    share_y: bool = False,
    only_significant: bool = False,
    show_pvalues: bool = False,
) -> None:
    """Draw the significance brackets on each facet's box panel.

    On a shared y-axis every facet anchors to the tallest facet, so the brackets line up
    across the figure. On independent axes each facet anchors to its *own* data —
    anchoring to a taller neighbour would stack the brackets far above the facet's range
    and re-inflate the very axis the independent scaling was meant to keep tight.
    """
    tops = [t for _ax, _v, groups in drawn if (t := _facet_top(groups)) is not None]
    if not tops:
        return
    tallest = max(tops)
    for ax, val, groups in drawn:
        pairs = annotations_by_facet.get(val)
        data_top = tallest if share_y else _facet_top(groups)
        if pairs and data_top is not None:
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
    layout: FacetLayout = "rows",
    share_y: bool = False,
) -> Any:
    """Per-condition CCDF faceted into one axis per ``facet_col`` value (shared legend).

    The CCDF puts the metric on **x**, so ``share_y`` — "share the metric scale" — ties
    the x-axes here. Shared, two panels covering very different value spans can no longer
    look alike; independent, each panel spends its full width on its own range.
    """
    # y is a survival probability here — always the same 0-1 range, so always shared.
    fig, axes = _facet_strip(len(values), width=5.0, sharex=share_y, sharey=True, layout=layout)
    order = ordered_condition_labels(per_fly, group_col)
    labels = list(_condition_groups(per_fly, metric, group_col, order))
    for ax, val in zip(axes, values, strict=True):
        subset = per_fly[per_fly[facet_col].astype(str) == val]
        groups = _condition_groups(subset, metric, group_col, order)
        ccdf_plot(groups, ax=ax, palette=palette, xlabel=xlabel or metric, legend=False)
        ax.set_title(val)
    if layout == "rows" and share_y:  # one shared metric axis, drawn once at the bottom
        _label_bottom_only(axes)
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
    layout: FacetLayout = "rows",
    share_y: bool = False,
) -> Any:
    """Cumulative sip time course faceted into one axis per ``facet_col`` value.

    Time is the same span in every facet, so the x-axis is always shared (drawn once
    under a stacked strip); ``share_y`` ties the cumulative-sip axes.
    """
    fig, axes = _facet_strip(
        len(values), width=6.0, height=3.4, sharex=True, sharey=share_y, layout=layout
    )
    labels = ordered_condition_labels(events, group_col)
    for ax, val in zip(axes, values, strict=True):
        subset = events[events[facet_col].astype(str) == val]
        series = cumulative_timecourse_by_condition(
            subset, n_samples, group_col=group_col, sampling_rate_hz=sampling_rate_hz
        )
        ordered = {lbl: series[lbl] for lbl in labels if lbl in series}
        shaded_lines(ordered, ax=ax, palette=palette, legend=False)
        ax.set_title(val)
    if layout == "rows":  # time is shared, so it is captioned once under the strip
        _label_bottom_only(axes)
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
    share_y: bool = False,
) -> Any:
    """Standalone dashboard with one row of panels per facet (top → bottom).

    Each row mirrors :func:`~flypad.plotting.cdf.standalone_dashboard`: the per-fly box
    plot, a central-tendency summary (``central`` = ``"median"`` → median+IQR, ``"mean"``
    → mean+95% CI), and the metric's CCDF. The leftmost panel of each row is titled with
    the facet value; condition colours are shared across every row, the legend is drawn
    once, and ``annotations_by_facet`` puts significance brackets on each row's box panel.
    ``share_y`` ties each *column* to one scale, making the rows directly comparable at
    the cost of flattening a low-intake facet.
    """
    order = ordered_condition_labels(per_fly, group_col)
    if palette is None:
        palette = condition_palette(order, sort=False)
    ann = annotations_by_facet or {}
    n = len(values)
    # "col": the panels differ across a row, so only same-panel columns can share a scale.
    fig, axes = plt.subplots(
        n, 3, figsize=(13.5, 4.0 * n), squeeze=False, sharey="col" if share_y else False
    )
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
            box_rows,
            ann,
            share_y=share_y,
            only_significant=only_significant,
            show_pvalues=show_pvalues,
        )
    suptitle(fig, title or f"{label} dashboard by {noun}")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / (4.0 * n)))
    return fig
