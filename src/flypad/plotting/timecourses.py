"""Time-course plots with shaded error bands (design §10, M6).

Ports ``jbfill`` (fill between two curves), ``task_14_shaded_plot`` (a mean line with
a shaded ± error band), and the cumulative feeding/activity time courses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from flypad.plotting.labels import time_axis
from flypad.plotting.theme import GRAY, distinguishable_colors


def _palette_colors(
    labels: Sequence[str], colors: Sequence[Any] | None, palette: Mapping[str, Any] | None
) -> list[Any]:
    if palette is not None:
        return [palette.get(label, GRAY) for label in labels]
    if colors is not None:
        return list(colors)
    return distinguishable_colors(len(labels))


def _new_ax(ax: Any | None, figsize: tuple[float, float] = (8.0, 3.2)) -> Any:
    return plt.subplots(figsize=figsize)[1] if ax is None else ax


def jbfill(
    ax: Any,
    x: npt.ArrayLike,
    lower: npt.ArrayLike,
    upper: npt.ArrayLike,
    *,
    color: Any = "#0072B2",
    alpha: float = 0.25,
) -> Any:
    """Fill the band between ``lower`` and ``upper`` over ``x`` (MATLAB ``jbfill``)."""
    return ax.fill_between(
        np.asarray(x, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=1,
    )


def shaded_plot(
    ax: Any,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    err: npt.ArrayLike,
    *,
    color: Any = "#0072B2",
    label: str | None = None,
    alpha: float = 0.25,
) -> Any:
    """Mean line ``y`` with a shaded ``± err`` band (``task_14_shaded_plot``)."""
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    ev = np.asarray(err, dtype=np.float64)
    jbfill(ax, xv, yv - ev, yv + ev, color=color, alpha=alpha)
    ax.plot(xv, yv, color=color, lw=1.8, label=label, zorder=2)
    return ax


def shaded_lines(
    series: Mapping[str, tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]],
    *,
    ax: Any | None = None,
    colors: Sequence[Any] | None = None,
    palette: Mapping[str, Any] | None = None,
    xlabel: str | None = None,
    ylabel: str = "Cumulative sips per fly",
    legend: bool = True,
    direct_labels: bool = False,
) -> Any:
    """Overlay several ``label -> (x, mean, err)`` shaded curves.

    ``xlabel=None`` (the default) treats x as seconds and rescales it to the most readable
    unit (s / min / h). ``direct_labels`` writes each condition at the end of its own curve
    instead of in a legend, removing the colour-matching eye travel; ``legend=False``
    suppresses the per-axis legend (e.g. when several axes share one figure legend).
    """
    ax = _new_ax(ax)
    labels = list(series)
    cols = _palette_colors(labels, colors, palette)
    auto_label = xlabel
    for i, label in enumerate(labels):
        x, y, err = series[label]
        xs = np.asarray(x, dtype=np.float64)
        if xlabel is None:
            xs, auto_label = time_axis(xs)
        shaded_plot(ax, xs, y, err, color=cols[i], label=label)
        if direct_labels and xs.size:
            yv = np.asarray(y, dtype=np.float64)
            ax.annotate(
                label,
                xy=(xs[-1], yv[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=cols[i],
            )
    if labels and legend and not direct_labels:
        ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel(auto_label or "Time (s)")
    ax.set_ylabel(ylabel)
    return ax


def cumulative_timecourse_plot(
    curves: Mapping[str, tuple[npt.ArrayLike, npt.ArrayLike]],
    *,
    ax: Any | None = None,
    colors: Sequence[Any] | None = None,
    palette: Mapping[str, Any] | None = None,
    xlabel: str = "time (s)",
    ylabel: str = "cumulative events",
) -> Any:
    """Plot ``label -> (time, cumulative_count)`` curves (cumulative feeding/activity)."""
    ax = _new_ax(ax)
    labels = list(curves)
    cols = _palette_colors(labels, colors, palette)
    for i, label in enumerate(labels):
        x, y = curves[label]
        ax.plot(x, y, color=cols[i], lw=1.8, label=label)
    if labels:
        ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax
