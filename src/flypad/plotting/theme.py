"""Figure styling: palette, rcParams, layout helpers (design §10, M6).

Ports the MATLAB v2.2 figure scaffolding:

* :func:`distinguishable_colors` — Tim Holy's greedy max-distance palette in CIE-Lab,
  for maximally separable per-condition colours.
* :func:`set_theme` / :func:`theme_context` — the clean Tufte/Wilke rcParams used
  across the suite.
* :func:`tight_subplot` — a gridded axes layout with controlled gaps/margins
  (MATLAB ``tight_subplot``); :func:`suptitle` — a styled figure title.

Presentation only: this package may import matplotlib, but the science core
(``signal``/``detect``/``postprocess``/``stats``) never does.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import to_rgb

FloatArray = npt.NDArray[np.float64]

# Colour-blind-safe accents (Wong palette) used throughout the suite.
PYTHON = "#0072B2"
MATLAB = "#E69F00"
ACCENT = "#D55E00"
GRAY = "#9AA0A6"
INK = "#2B2B2B"

THEME: dict[str, Any] = {
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.titleweight": "semibold",
    "axes.titlecolor": INK,
    "axes.labelcolor": "#3c4043",
    "axes.labelsize": 10.5,
    "axes.edgecolor": "#c8ccd0",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#5f6368",
    "ytick.color": "#5f6368",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "lines.solid_capstyle": "round",
    "figure.facecolor": "white",
}


def set_theme() -> None:
    """Apply the flypad figure style to the global matplotlib rcParams."""
    mpl.rcParams.update(cast(Any, THEME))


@contextmanager
def theme_context() -> Iterator[None]:
    """Apply the flypad style for the duration of a ``with`` block, then restore."""
    with mpl.rc_context(cast(Any, THEME)):
        yield


def _srgb_to_lab(rgb: FloatArray) -> FloatArray:
    """Convert an ``(n, 3)`` array of sRGB values in ``[0, 1]`` to CIE-Lab (D65)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    # linear sRGB -> XYZ (D65)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = lin @ m.T
    white = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / white

    def f(t: FloatArray) -> FloatArray:
        delta = 6.0 / 29.0
        return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4.0 / 29.0)

    fx, fy, fz = f(xyz[:, 0]), f(xyz[:, 1]), f(xyz[:, 2])
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=1)


def distinguishable_colors(
    n: int,
    *,
    background: str = "white",
    grid: int = 16,
) -> list[tuple[float, float, float]]:
    """Return ``n`` maximally distinguishable RGB colours (greedy in CIE-Lab).

    Reproduces Tim Holy's ``distinguishable_colors``: from a dense RGB candidate grid,
    greedily pick the colour farthest (in Lab) from the background and all colours
    already chosen.
    """
    if n <= 0:
        return []
    axis = np.linspace(0.0, 1.0, grid)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    candidates = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1)
    cand_lab = _srgb_to_lab(candidates)

    bg_lab = _srgb_to_lab(np.asarray([to_rgb(background)], dtype=np.float64))[0]
    min_dist = np.linalg.norm(cand_lab - bg_lab, axis=1)

    chosen: list[tuple[float, float, float]] = []
    last_lab = bg_lab
    for _ in range(n):
        min_dist = np.minimum(min_dist, np.linalg.norm(cand_lab - last_lab, axis=1))
        idx = int(np.argmax(min_dist))
        rgb = candidates[idx]
        chosen.append((float(rgb[0]), float(rgb[1]), float(rgb[2])))
        last_lab = cand_lab[idx]
    return chosen


def tight_subplot(
    nrows: int,
    ncols: int,
    *,
    figsize: tuple[float, float] | None = None,
    gap: float = 0.06,
    margin: float = 0.08,
) -> tuple[Any, Any]:
    """A grid of axes with uniform gaps and margins (MATLAB ``tight_subplot``).

    Returns ``(figure, axes)`` where ``axes`` is the 2-D array from
    :func:`matplotlib.pyplot.subplots` (squeezed for a single row/column).
    """
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        gridspec_kw={"wspace": gap * ncols, "hspace": gap * nrows},
    )
    fig.subplots_adjust(left=margin, right=1 - margin, bottom=margin, top=1 - margin)
    return fig, axes


def suptitle(fig: Any, text: str, **kwargs: Any) -> Any:
    """A styled figure-level title."""
    opts: dict[str, Any] = {"fontsize": 14, "fontweight": "semibold", "color": INK}
    opts.update(kwargs)
    return fig.suptitle(text, **opts)
