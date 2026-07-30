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

from flypad.plotting.labels import time_axis
from flypad.plotting.theme import GRAY, INK


def _new_ax(ax: Any | None, n_rows: int) -> Any:
    if ax is not None:
        return ax
    height = max(2.0, min(0.16 * n_rows + 1.0, 14.0))  # cap for full-plate (96-ch) rasters
    return plt.subplots(figsize=(9.0, height))[1]


def raster_panels(
    panels: Sequence[
        tuple[str, Sequence[npt.ArrayLike], Sequence[tuple[float, float, float]] | None]
    ],
    *,
    row_conditions: Sequence[str] | None = None,
    palette: Mapping[str, Any] | None = None,
    sampling_rate_hz: int | None = None,
    suptitle_text: str | None = None,
    **kwargs: Any,
) -> Any:
    """Several rasters side by side, sharing the channel axis.

    ``panels`` is a sequence of ``(title, rows, fills)``. Used to show the same recording
    in absolute time next to its arena-aligned version, so the effect of the crop is
    visible. The x-axes are independent — the panels deliberately span different
    durations. Returns the figure.
    """
    from flypad.plotting.theme import suptitle

    n = len(panels)
    n_rows = max((len(rows) for _t, rows, _f in panels), default=1)
    height = max(2.0, min(0.16 * n_rows + 1.0, 14.0))
    fig, axes = plt.subplots(1, n, figsize=(8.0 * n, height), sharey=True, squeeze=False)
    for i, (title, rows, fills) in enumerate(panels):
        ax = axes[0][i]
        raster_plot(
            rows,
            ax=ax,
            row_conditions=row_conditions,
            palette=palette,
            sampling_rate_hz=sampling_rate_hz,
            fills=fills,
            legend=i == n - 1,  # one legend, on the right-hand panel
            **kwargs,
        )
        ax.set_title(title)
        if i:
            ax.set_ylabel("")
    if suptitle_text:
        suptitle(fig, suptitle_text)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if suptitle_text else 1.0))
    return fig


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
    fills: Sequence[tuple[float, float, float]] | None = None,
    fill_color: str = "#D62728",
) -> Any:
    """Event raster: ``rows[i]`` are the event onsets drawn on raster line ``i``.

    Colour each row by its condition via ``row_conditions`` (one label per row) + a
    ``palette`` (label -> colour); a legend of the conditions present is added. Falls
    back to explicit per-row ``colors`` or a single ink colour. When ``sampling_rate_hz``
    is given the time axis is shown in the most readable unit (s / min / h), matching the
    time-course figures.

    ``fills`` overlays ``(sample, row_lo, row_hi)`` markers — the manual arena-fill
    timestamps — as red vertical segments spanning only the rows that arena covers, so
    each arena's line sits against its own channels. Samples are in the same units as
    ``rows`` and are rescaled with them.
    """
    ax = _new_ax(ax, len(rows))
    scale = 1.0 / sampling_rate_hz if sampling_rate_hz else 1.0
    positions = [np.asarray(r, dtype=np.float64).ravel() * scale for r in rows]
    factor = 1.0
    if sampling_rate_hz:
        longest = max((float(p.max()) for p in positions if p.size), default=0.0)
        unit_scale, unit_label = time_axis(np.asarray([longest]))
        factor = float(unit_scale[0] / longest) if longest else 1.0
        positions = [p * factor for p in positions]
        if xlabel is None:
            xlabel = unit_label
    if xlabel is None:
        xlabel = "sample"

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
    if fills:
        for sample, row_lo, row_hi in fills:
            x = float(sample) * scale * factor
            ax.plot(
                [x, x],
                [row_lo - line_length / 2, row_hi + line_length / 2],
                color=fill_color,
                lw=1.4,
                zorder=4,
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
        if fills:
            handles.append(Line2D([0], [0], color=fill_color, lw=1.4, label="arena filled"))
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
