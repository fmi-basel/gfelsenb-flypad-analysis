"""Box / spread / error-bar plots (design §10, M6).

Ports the MATLAB v2.2 box-plot family: ``TiltedBoxPlot`` (boxes + the individual
per-fly dots that make flyPAD figures legible), ``plotSpread`` (jittered points),
``Median_IQR_Plot`` / ``CI_Plot`` (point ± interval), and ``myErrorbar``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from flypad.plotting.theme import GRAY, INK, distinguishable_colors

FloatArray = npt.NDArray[np.float64]


def _as_groups(groups: Mapping[str, npt.ArrayLike]) -> tuple[list[str], list[FloatArray]]:
    labels = list(groups)
    arrays = []
    for label in labels:
        a = np.asarray(groups[label], dtype=np.float64).ravel()
        arrays.append(a[np.isfinite(a)])
    return labels, arrays


def _colors(n: int, colors: Sequence[Any] | None) -> list[Any]:
    if colors is not None:
        return list(colors)
    return distinguishable_colors(n)


def _new_ax(ax: Any | None) -> Any:
    return plt.subplots(figsize=(1.6 + 0.0, 4.0))[1] if ax is None else ax


def my_errorbar(
    ax: Any,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    yerr: npt.ArrayLike,
    *,
    color: Any = INK,
    **kwargs: Any,
) -> Any:
    """Thin wrapper around :meth:`Axes.errorbar` with the suite's defaults."""
    opts: dict[str, Any] = {
        "fmt": "o",
        "color": color,
        "capsize": 3,
        "elinewidth": 1.3,
        "markersize": 5,
    }
    opts.update(kwargs)
    return ax.errorbar(x, y, yerr=yerr, **opts)


def plot_spread(
    ax: Any,
    groups: Mapping[str, npt.ArrayLike],
    *,
    positions: Sequence[float] | None = None,
    colors: Sequence[Any] | None = None,
    jitter: float = 0.08,
    seed: int = 0,
    size: float = 14,
) -> Any:
    """Jittered scatter of the individual values in each group (``plotSpread``)."""
    labels, arrays = _as_groups(groups)
    pos = list(positions) if positions is not None else list(range(len(labels)))
    cols = _colors(len(labels), colors)
    rng = np.random.default_rng(seed)
    for i, a in enumerate(arrays):
        if a.size == 0:
            continue
        xs = pos[i] + (rng.random(a.size) - 0.5) * 2 * jitter
        ax.scatter(xs, a, s=size, color=cols[i], alpha=0.55, edgecolor="none", zorder=3)
    return ax


def tilted_boxplot(
    groups: Mapping[str, npt.ArrayLike],
    *,
    ax: Any | None = None,
    colors: Sequence[Any] | None = None,
    show_points: bool = True,
    rotation: float = 30.0,
    ylabel: str | None = None,
) -> Any:
    """Box plot per group with overlaid per-fly dots and tilted category labels.

    The flyPAD ``TiltedBoxPlot``: a thin box (median + IQR + whiskers, no fliers)
    behind the raw per-fly points, with rotated x-tick labels.
    """
    ax = _new_ax(ax)
    labels, arrays = _as_groups(groups)
    cols = _colors(len(labels), colors)
    positions = list(range(len(labels)))

    drawable = [(i, a) for i, a in enumerate(arrays) if a.size]
    if drawable:
        bp = ax.boxplot(
            [a for _, a in drawable],
            positions=[positions[i] for i, _ in drawable],
            widths=0.5,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": INK, "linewidth": 1.5},
            whiskerprops={"color": GRAY, "linewidth": 1.0},
            capprops={"color": GRAY, "linewidth": 1.0},
            boxprops={"edgecolor": GRAY, "linewidth": 1.0},
        )
        for (i, _), patch in zip(drawable, bp["boxes"], strict=True):
            patch.set_facecolor(cols[i])
            patch.set_alpha(0.25)

    if show_points:
        plot_spread(ax, groups, positions=positions, colors=cols)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def median_iqr_plot(
    groups: Mapping[str, npt.ArrayLike],
    *,
    ax: Any | None = None,
    color: Any = INK,
    rotation: float = 30.0,
) -> Any:
    """Point at the median with whiskers to the 25th/75th percentile (``Median_IQR_Plot``)."""
    ax = _new_ax(ax)
    labels, arrays = _as_groups(groups)
    pos = list(range(len(labels)))
    for i, a in enumerate(arrays):
        if a.size == 0:
            continue
        med = float(np.median(a))
        lo, hi = np.percentile(a, [25, 75])
        my_errorbar(ax, [pos[i]], [med], [[med - lo], [hi - med]], color=color)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
    return ax


def ci_plot(
    groups: Mapping[str, npt.ArrayLike],
    *,
    ax: Any | None = None,
    color: Any = INK,
    ci_level: float = 0.95,
    rotation: float = 30.0,
) -> Any:
    """Point at the mean with a Student-t confidence interval (``CI_Plot``)."""
    from scipy import stats as scipy_stats

    ax = _new_ax(ax)
    labels, arrays = _as_groups(groups)
    pos = list(range(len(labels)))
    for i, a in enumerate(arrays):
        if a.size < 2:
            continue
        mean = float(a.mean())
        sem = float(a.std(ddof=1)) / np.sqrt(a.size)
        half = float(scipy_stats.t.ppf(0.5 + ci_level / 2, a.size - 1)) * sem
        my_errorbar(ax, [pos[i]], [mean], [half], color=color)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
    return ax
