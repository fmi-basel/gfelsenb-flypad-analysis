"""Sip / event raster plots (design §10, M6).

One row per channel, a tick at every event onset — the raw-event overview that sits
at the top of the flyPAD dashboards.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from flypad.plotting.theme import INK


def _new_ax(ax: Any | None, n_rows: int) -> Any:
    if ax is not None:
        return ax
    return plt.subplots(figsize=(9.0, max(2.0, 0.18 * n_rows + 1.0)))[1]


def raster_plot(
    rows: Sequence[npt.ArrayLike],
    *,
    ax: Any | None = None,
    colors: Sequence[Any] | None = None,
    row_labels: Sequence[str] | None = None,
    line_length: float = 0.8,
    sampling_rate_hz: int | None = None,
    xlabel: str | None = None,
) -> Any:
    """Event raster: ``rows[i]`` are the event onsets drawn on raster line ``i``.

    Colours may be given per row (e.g. one colour per condition). When
    ``sampling_rate_hz`` is given, the time axis is shown in seconds.
    """
    ax = _new_ax(ax, len(rows))
    scale = 1.0 / sampling_rate_hz if sampling_rate_hz else 1.0
    positions = [np.asarray(r, dtype=np.float64).ravel() * scale for r in rows]
    if xlabel is None:
        xlabel = "time (s)" if sampling_rate_hz else "sample"
    color_arg: Any = INK if colors is None else list(colors)
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
    return ax
