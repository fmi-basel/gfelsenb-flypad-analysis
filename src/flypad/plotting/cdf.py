"""CDF / CCDF figures and the standalone dashboard (design §10, M6).

Ports the ICDF/CCDF figures and the ``PlotFLYPAD_Standalone_MEAN/MEDIAN`` dashboards:
a composite of the per-condition box plot, the metric's CDF/CCDF, and (optionally)
a cumulative time course — the one-glance summary the MATLAB suite produces per run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from flypad.plotting.boxplots import median_iqr_plot, tilted_boxplot
from flypad.plotting.theme import distinguishable_colors, suptitle
from flypad.stats.distributions import ccdf, icdf


def _new_ax(ax: Any | None, figsize: tuple[float, float] = (6.0, 4.0)) -> Any:
    return plt.subplots(figsize=figsize)[1] if ax is None else ax


def cdf_plot(
    groups: Mapping[str, npt.ArrayLike],
    *,
    ax: Any | None = None,
    complementary: bool = False,
    colors: Sequence[Any] | None = None,
    xlabel: str = "value",
) -> Any:
    """Step CDF (or CCDF if ``complementary``) per group."""
    ax = _new_ax(ax)
    labels = list(groups)
    cols = list(colors) if colors is not None else distinguishable_colors(len(labels))
    fn = ccdf if complementary else icdf
    for i, label in enumerate(labels):
        x, y = fn(groups[label])
        if x.size == 0:
            continue
        ax.step(x, y, where="post", color=cols[i], lw=1.8, label=label)
    if labels:
        ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(X ≥ x)" if complementary else "P(X ≤ x)")
    ax.set_ylim(0, 1.02)
    return ax


def ccdf_plot(groups: Mapping[str, npt.ArrayLike], **kwargs: Any) -> Any:
    """Complementary CDF per group (survival curves)."""
    return cdf_plot(groups, complementary=True, **kwargs)


def _grouped_values(
    per_fly: pd.DataFrame,
    metric: str,
    group_col: str,
) -> dict[str, npt.NDArray[np.float64]]:
    return {
        str(label): grp[metric].to_numpy(dtype=np.float64)
        for label, grp in per_fly.groupby(group_col, dropna=False)
    }


def standalone_dashboard(
    per_fly: pd.DataFrame,
    metric: str,
    *,
    group_col: str = "condition_label",
    central: str = "median",
    title: str | None = None,
) -> Any:
    """A three-panel ``PlotFLYPAD_Standalone`` dashboard for one metric.

    Panels: tilted box plot (per-fly distribution), a central-tendency summary
    (median+IQR or mean+CI), and the metric's CCDF. ``central`` selects ``"median"``
    (``PlotFLYPAD_Standalone_MEDIAN``) or ``"mean"`` (``..._MEAN``).
    """
    groups = _grouped_values(per_fly, metric, group_col)
    colors = distinguishable_colors(len(groups))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    tilted_boxplot(groups, ax=axes[0], colors=colors, ylabel=metric)
    axes[0].set_title("per-fly distribution")

    if central == "mean":
        from flypad.plotting.boxplots import ci_plot

        ci_plot(groups, ax=axes[1])
        axes[1].set_title("mean ± 95% CI")
    else:
        median_iqr_plot(groups, ax=axes[1])
        axes[1].set_title("median ± IQR")
    axes[1].set_ylabel(metric)

    ccdf_plot(groups, ax=axes[2], colors=colors, xlabel=metric)
    axes[2].set_title("CCDF")

    fig.tight_layout()
    if title:
        suptitle(fig, title)
        fig.subplots_adjust(top=0.86)
    return fig
