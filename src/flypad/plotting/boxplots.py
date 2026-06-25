"""Box / spread / error-bar plots (design §10, M6).

Ports the MATLAB v2.2 box-plot family: ``TiltedBoxPlot`` (boxes + the individual
per-fly dots that make flyPAD figures legible, optionally sheared for the classic
tilted look), ``plotSpread`` (jittered points), ``Median_IQR_Plot`` / ``CI_Plot``
(point ± interval), ``myErrorbar``, and a two-choice substrate comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D, blended_transform_factory

from flypad.plotting.theme import GRAY, INK, MATLAB, PYTHON, distinguishable_colors

FloatArray = npt.NDArray[np.float64]
Palette = Mapping[str, Any]


def _as_groups(groups: Mapping[str, npt.ArrayLike]) -> tuple[list[str], list[FloatArray]]:
    labels = list(groups)
    arrays = []
    for label in labels:
        a = np.asarray(groups[label], dtype=np.float64).ravel()
        arrays.append(a[np.isfinite(a)])
    return labels, arrays


def _resolve_colors(
    labels: Sequence[str],
    colors: Sequence[Any] | None,
    palette: Palette | None,
) -> list[Any]:
    if palette is not None:
        return [palette.get(label, GRAY) for label in labels]
    if colors is not None:
        return list(colors)
    return distinguishable_colors(len(labels))


def _new_ax(ax: Any | None) -> Any:
    return plt.subplots(figsize=(4.6, 4.0))[1] if ax is None else ax


def _tilt_box(
    bp: dict[str, list[Any]], index: int, center: tuple[float, float], tilt_deg: float, ax: Any
) -> None:
    """Shear the ``index``-th box (box/whiskers/caps/median) around ``center``."""
    cx, cy = center
    shear = Affine2D().translate(-cx, -cy).skew_deg(tilt_deg, 0).translate(cx, cy) + ax.transData
    artists = [bp["boxes"][index], bp["medians"][index]]
    artists += bp["whiskers"][2 * index : 2 * index + 2]
    artists += bp["caps"][2 * index : 2 * index + 2]
    for artist in artists:
        artist.set_transform(shear)


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
    palette: Palette | None = None,
    jitter: float = 0.08,
    seed: int = 0,
    size: float = 14,
) -> Any:
    """Jittered scatter of the individual values in each group (``plotSpread``)."""
    labels, arrays = _as_groups(groups)
    pos = list(positions) if positions is not None else list(range(len(labels)))
    cols = _resolve_colors(labels, colors, palette)
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
    palette: Palette | None = None,
    show_points: bool = True,
    show_n: bool = True,
    tilt_deg: float = 0.0,
    rotation: float = 30.0,
    ylabel: str | None = None,
) -> Any:
    """Box plot per group with overlaid per-fly dots and tilted category labels.

    ``tilt_deg`` shears the box glyphs for the classic flyPAD tilted look (0 = upright);
    ``show_n`` annotates each group's fly count beneath the axis.
    """
    ax = _new_ax(ax)
    labels, arrays = _as_groups(groups)
    cols = _resolve_colors(labels, colors, palette)
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
        for k, ((i, a), patch) in enumerate(zip(drawable, bp["boxes"], strict=True)):
            patch.set_facecolor(cols[i])
            patch.set_alpha(0.25)
            if tilt_deg:
                _tilt_box(bp, k, (positions[i], float(np.median(a))), tilt_deg, ax)

    if show_points:
        plot_spread(ax, groups, positions=positions, colors=cols)

    if show_n:
        blended = blended_transform_factory(ax.transData, ax.transAxes)
        for i, a in enumerate(arrays):
            ax.text(
                positions[i],
                -0.04,
                f"n={a.size}",
                transform=blended,
                ha="center",
                va="top",
                fontsize=8,
                color=GRAY,
            )

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


def substrate_comparison(
    per_fly: pd.DataFrame,
    metric: str,
    *,
    ax: Any | None = None,
    group_col: str = "condition_label",
    side_col: str = "substrate_side",
    rotation: float = 30.0,
    ylabel: str | None = None,
) -> Any:
    """Two-choice comparison: side-by-side left/right boxes per condition.

    Each condition gets two offset boxes (left vs right substrate), so substrate
    preference is visible at a glance.
    """
    ax = _new_ax(ax)
    conditions = sorted(per_fly[group_col].dropna().unique(), key=str)
    sides = [("left", PYTHON), ("right", MATLAB)]
    width, offset = 0.34, 0.2
    for ci, condition in enumerate(conditions):
        for side, color in sides:
            sign = -1 if side == "left" else 1
            mask = (per_fly[group_col] == condition) & (per_fly[side_col] == side)
            values = per_fly.loc[mask, metric].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            box = ax.boxplot(
                values,
                positions=[ci + sign * offset],
                widths=width,
                showfliers=False,
                patch_artist=True,
                medianprops={"color": INK, "linewidth": 1.4},
                whiskerprops={"color": GRAY},
                capprops={"color": GRAY},
                boxprops={"edgecolor": GRAY},
            )
            box["boxes"][0].set_facecolor(color)
            box["boxes"][0].set_alpha(0.4)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(
        [str(c) for c in conditions], rotation=rotation, ha="right" if rotation else "center"
    )
    handles = [
        Line2D([0], [0], color=PYTHON, lw=6, alpha=0.4, label="left"),
        Line2D([0], [0], color=MATLAB, lw=6, alpha=0.4, label="right"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax
