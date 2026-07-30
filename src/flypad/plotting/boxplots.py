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


def significance_marker(p: float) -> str:
    """Star notation for a p-value: ``***`` <0.001, ``**`` <0.01, ``*`` <0.05, else ``n.s.``."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def format_pvalue(p: float) -> str:
    """Exact p-value for a bracket label (``p<0.001`` below the printable floor)."""
    return "p<0.001" if p < 0.001 else f"p={p:.3g}"


def _axes_fraction_y(ax: Any, y_data: float) -> float:
    """Convert a data-space y to an axes-fraction y (works on any axis scale)."""
    display = ax.transData.transform((0.0, y_data))
    return float(ax.transAxes.inverted().transform(display)[1])


def _expand_top(ax: Any, target_fraction: float) -> None:
    """Grow the y-limit so ``target_fraction`` of the axes height fits inside it."""
    if target_fraction <= 1.0:
        return
    ymin, ymax = ax.get_ylim()
    if ax.get_yscale() == "log" and ymin > 0 and ymax > 0:
        lo, hi = np.log10(ymin), np.log10(ymax)
        ax.set_ylim(ymin, float(10 ** (lo + (hi - lo) * target_fraction)))
    else:
        ax.set_ylim(ymin, ymin + (ymax - ymin) * target_fraction)


def annotate_significance(
    ax: Any,
    pairs: Sequence[tuple[str, str, float]],
    positions: Mapping[str, float],
    *,
    data_top: float,
    base: float = 0.05,
    step: float = 0.085,
    only_significant: bool = False,
    show_pvalues: bool = False,
    alpha: float = 0.05,
) -> None:
    """Draw stacked significance brackets between labelled x-positions.

    ``pairs`` is ``(label_a, label_b, p_value)`` triples and ``positions`` maps each label
    to its x-coordinate; ``data_top`` is the highest data value the brackets must clear.
    Brackets stack bottom-up (narrowest span first) in axes-fraction steps, so the layout
    is identical on linear and log axes, and the y-limit is grown to make room.
    ``only_significant`` drops pairs with ``p >= alpha``; ``show_pvalues`` prints the exact
    p instead of stars. Pairs whose labels are absent are skipped.
    """
    drawable = [(a, b, p) for a, b, p in pairs if a in positions and b in positions]
    if only_significant:
        drawable = [t for t in drawable if t[2] < alpha]
    if not drawable:
        return
    drawable.sort(key=lambda t: abs(positions[t[0]] - positions[t[1]]))

    tick = step * 0.3
    # Reserve the headroom *first*, so the fractions below map to their final positions.
    needed = _axes_fraction_y(ax, data_top) + base + (len(drawable) - 1) * step + tick + 0.06
    _expand_top(ax, needed)

    y0 = _axes_fraction_y(ax, data_top) + base
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    label = format_pvalue if show_pvalues else significance_marker
    for level, (a, b, p) in enumerate(drawable):
        x1, x2 = sorted((positions[a], positions[b]))
        y = y0 + level * step
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + tick, y + tick, y],
            lw=1.0,
            color=INK,
            transform=trans,
            clip_on=False,
        )
        ax.text(
            (x1 + x2) / 2,
            y + tick,
            label(p),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK,
            transform=trans,
        )


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


def swarm_offsets(values: npt.ArrayLike, width: float, n_bins: int = 40) -> FloatArray:
    """Density-aware x-offsets for a beeswarm: points at similar y fan out symmetrically.

    Values are binned along y; within a bin the points are spread evenly about the centre
    (``-width … +width``), so the cloud's silhouette shows the distribution instead of the
    random overlap a uniform jitter produces.
    """
    a = np.asarray(values, dtype=np.float64).ravel()
    out = np.zeros(a.size, dtype=np.float64)
    if a.size == 0:
        return out
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        bins = np.zeros(a.size, dtype=np.int64)
    else:
        bins = np.clip(((a - lo) / (hi - lo) * n_bins).astype(np.int64), 0, n_bins - 1)
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        m = idx.size
        if m == 1:
            continue
        # even spacing about the centre, capped at +/- width
        spread = np.linspace(-1.0, 1.0, m)
        out[idx] = spread * min(width, width * m / 8.0 + width * 0.25)
    return out


def plot_spread(
    ax: Any,
    groups: Mapping[str, npt.ArrayLike],
    *,
    positions: Sequence[float] | None = None,
    colors: Sequence[Any] | None = None,
    palette: Palette | None = None,
    jitter: float = 0.11,
    seed: int = 0,
    size: float = 11,
    mode: str = "swarm",
) -> Any:
    """Scatter of the individual values in each group (``plotSpread``).

    ``mode="swarm"`` (default) lays the points out by density (see :func:`swarm_offsets`);
    ``mode="jitter"`` reproduces the original uniform random offsets.
    """
    labels, arrays = _as_groups(groups)
    pos = list(positions) if positions is not None else list(range(len(labels)))
    cols = _resolve_colors(labels, colors, palette)
    rng = np.random.default_rng(seed)
    for i, a in enumerate(arrays):
        if a.size == 0:
            continue
        if mode == "swarm":
            xs = pos[i] + swarm_offsets(a, jitter)
        else:
            xs = pos[i] + (rng.random(a.size) - 0.5) * 2 * jitter
        ax.scatter(
            xs,
            a,
            s=size,
            color=cols[i],
            alpha=0.75,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
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
    annotations: Sequence[tuple[str, str, float]] | None = None,
    yscale: str = "linear",
    only_significant: bool = False,
    show_pvalues: bool = False,
) -> Any:
    """Box plot per group with overlaid per-fly dots and tilted category labels.

    ``tilt_deg`` shears the box glyphs for the classic flyPAD tilted look (0 = upright);
    ``show_n`` appends each group's fly count as a second line of its tick label.
    ``annotations`` is a list of ``(label_a, label_b, p_value)`` pairwise comparisons drawn
    as stacked significance brackets above the boxes (``only_significant`` / ``show_pvalues``
    tune them). ``yscale`` accepts ``"linear"``, ``"log"`` or ``"symlog"`` — the latter two
    keep a strongly skewed group readable next to a large one.
    """
    ax = _new_ax(ax)
    labels, arrays = _as_groups(groups)
    cols = _resolve_colors(labels, colors, palette)
    positions = list(range(len(labels)))
    if yscale == "log":
        # linthresh-free log needs positive data; zeros are common (flies that never ate)
        ax.set_yscale("symlog" if any((a <= 0).any() for a in arrays if a.size) else "log")
    elif yscale == "symlog":
        ax.set_yscale("symlog")

    drawable = [(i, a) for i, a in enumerate(arrays) if a.size]
    if drawable:
        bp = ax.boxplot(
            [a for _, a in drawable],
            positions=[positions[i] for i, _ in drawable],
            widths=0.5,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": INK, "linewidth": 1.6},
            whiskerprops={"color": GRAY, "linewidth": 1.0},
            capprops={"color": GRAY, "linewidth": 1.0},
            boxprops={"linewidth": 1.1},
            zorder=2,
        )
        for k, ((i, a), patch) in enumerate(zip(drawable, bp["boxes"], strict=True)):
            # Light fill + a saturated edge in the same hue: the box frames the points
            # rather than competing with them.
            patch.set_facecolor(cols[i])
            patch.set_alpha(0.16)
            patch.set_edgecolor(cols[i])
            if tilt_deg:
                _tilt_box(bp, k, (positions[i], float(np.median(a))), tilt_deg, ax)

    if show_points:
        plot_spread(ax, groups, positions=positions, colors=cols)

    # Fold the per-group N into the tick label (a second line) so the count always
    # rides with its category and never collides with rotated / long labels.
    tick_labels = (
        [f"{lbl}\nn={a.size}" for lbl, a in zip(labels, arrays, strict=True)] if show_n else labels
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=rotation, ha="right" if rotation else "center")
    if ylabel:
        ax.set_ylabel(ylabel)

    if annotations:
        finite = [a for a in arrays if a.size]
        if finite:
            annotate_significance(
                ax,
                annotations,
                {label: positions[i] for i, label in enumerate(labels)},
                data_top=float(max(np.nanmax(a) for a in finite)),
                only_significant=only_significant,
                show_pvalues=show_pvalues,
            )
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
