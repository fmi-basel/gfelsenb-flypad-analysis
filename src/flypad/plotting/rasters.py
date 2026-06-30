"""Sip / event raster plots (design §10, M6).

One row per channel, a tick at every event onset — the raw-event overview that sits
at the top of the flyPAD dashboards.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.lines import Line2D

from flypad.plotting.theme import GRAY, INK


def _new_ax(ax: Any | None, n_rows: int) -> Any:
    if ax is not None:
        return ax
    height = max(2.0, min(0.16 * n_rows + 1.0, 14.0))  # cap for full-plate (96-ch) rasters
    return plt.subplots(figsize=(9.0, height))[1]


def raster_plot(
    rows: Sequence[npt.ArrayLike],
    *,
    ax: Any | None = None,
    colors: Sequence[Any] | None = None,
    row_conditions: Sequence[str] | None = None,
    palette: Mapping[str, Any] | None = None,
    row_labels: Sequence[str] | None = None,
    line_length: float = 0.8,
    sampling_rate_hz: int | None = None,
    xlabel: str | None = None,
    legend: bool = True,
) -> Any:
    """Event raster: ``rows[i]`` are the event onsets drawn on raster line ``i``.

    Colour each row by its condition via ``row_conditions`` (one label per row) + a
    ``palette`` (label -> colour); a legend of the conditions present is added. Falls
    back to explicit per-row ``colors`` or a single ink colour. When ``sampling_rate_hz``
    is given the time axis is shown in seconds.
    """
    ax = _new_ax(ax, len(rows))
    scale = 1.0 / sampling_rate_hz if sampling_rate_hz else 1.0
    positions = [np.asarray(r, dtype=np.float64).ravel() * scale for r in rows]
    if xlabel is None:
        xlabel = "time (s)" if sampling_rate_hz else "sample"

    if row_conditions is not None and palette is not None:
        color_arg: Any = [palette.get(str(c), GRAY) for c in row_conditions]
    elif colors is not None:
        color_arg = list(colors)
    else:
        color_arg = INK

    ax.eventplot(
        positions,
        colors=color_arg,
        lineoffsets=np.arange(len(positions)),
        linelengths=line_length,
        linewidths=0.6,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("channel")
    ax.set_ylim(-0.5, len(positions) - 0.5)
    if row_labels is not None:
        ax.set_yticks(np.arange(len(positions)))
        ax.set_yticklabels(list(row_labels))

    if legend and row_conditions is not None and palette is not None:
        present = [c for c in palette if c in set(map(str, row_conditions))]
        handles = [Line2D([0], [0], color=palette[c], lw=3, label=c) for c in present]
        if handles:
            # Outside the axes (top-right) so it never overlaps the raster rows;
            # the suite's tight savefig bbox keeps it in the exported figure.
            ax.legend(
                handles=handles,
                frameon=False,
                fontsize=8,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
            )
    return ax
