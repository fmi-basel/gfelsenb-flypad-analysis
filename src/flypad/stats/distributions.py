"""Empirical distributions: ICDF / ICDF_Linear / CCDF / CumDifference (design §10).

These mirror the MATLAB v2.2 distribution helpers used for the feeding-statistics
figures:

* :func:`icdf`           — empirical CDF at the observed values, ``F(x) = P(X <= x)``
  (the ``ICDF`` curve plotted value-on-x, probability-on-y).
* :func:`icdf_linear`    — that CDF resampled on a *regular* grid so curves from
  different flies share an x-axis and can be averaged (``ICDF_Linear``).
* :func:`ccdf`           — the complementary CDF / survival function, ``P(X >= x)``.
* :func:`cum_difference` — the signed difference between two CDFs on a shared grid
  (the statistic behind the CCDF permutation comparison).

Plus :func:`quantile` (true inverse CDF) and :func:`cumulative_time_course`
(cumulative event count over the recording, for the time-course plots).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _clean(values: npt.ArrayLike) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def icdf(values: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Empirical CDF. Returns ``(x_sorted, F)`` with ``F = (1..n)/n``."""
    x = np.sort(_clean(values))
    n = x.size
    f = np.arange(1, n + 1, dtype=np.float64) / n if n else np.asarray([], dtype=np.float64)
    return x, f


def ccdf(values: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Complementary CDF (survival). Returns ``(x_sorted, S)`` with ``S = P(X >= x)``."""
    x = np.sort(_clean(values))
    n = x.size
    s = 1.0 - np.arange(0, n, dtype=np.float64) / n if n else np.asarray([], dtype=np.float64)
    return x, s


def icdf_linear(values: npt.ArrayLike, grid: npt.ArrayLike) -> FloatArray:
    """CDF ``P(X <= g)`` evaluated at each point of ``grid`` (a step interpolation)."""
    arr = _clean(values)
    g = np.asarray(grid, dtype=np.float64).ravel()
    if arr.size == 0:
        return np.full(g.shape, np.nan)
    arr_sorted = np.sort(arr)
    return np.searchsorted(arr_sorted, g, side="right").astype(np.float64) / arr.size


def ccdf_linear(values: npt.ArrayLike, grid: npt.ArrayLike) -> FloatArray:
    """Survival ``P(X >= g)`` evaluated at each point of ``grid``."""
    arr = _clean(values)
    g = np.asarray(grid, dtype=np.float64).ravel()
    if arr.size == 0:
        return np.full(g.shape, np.nan)
    arr_sorted = np.sort(arr)
    return 1.0 - np.searchsorted(arr_sorted, g, side="left").astype(np.float64) / arr.size


def cum_difference(
    values_a: npt.ArrayLike,
    values_b: npt.ArrayLike,
    grid: npt.ArrayLike,
) -> FloatArray:
    """Signed difference of two CDFs on a shared ``grid``: ``F_a(grid) - F_b(grid)``."""
    return icdf_linear(values_a, grid) - icdf_linear(values_b, grid)


def quantile(values: npt.ArrayLike, p: npt.ArrayLike) -> FloatArray:
    """True inverse CDF (quantile function) at probability/-ies ``p`` in ``[0, 1]``."""
    arr = _clean(values)
    pf = np.asarray(p, dtype=np.float64)
    if arr.size == 0:
        return np.full(pf.shape, np.nan)
    return np.asarray(np.quantile(arr, pf), dtype=np.float64)


def common_grid(
    *value_sets: npt.ArrayLike,
    n: int = 512,
) -> FloatArray:
    """A regular grid spanning the pooled range of all ``value_sets`` (for averaging)."""
    pooled = np.concatenate([_clean(v) for v in value_sets]) if value_sets else np.asarray([])
    if pooled.size == 0:
        return np.linspace(0.0, 1.0, n)
    lo, hi = float(pooled.min()), float(pooled.max())
    if lo == hi:
        hi = lo + 1.0
    return np.linspace(lo, hi, n)


def cumulative_time_course(
    onsets: npt.ArrayLike,
    n_samples: int,
    n_bins: int = 1000,
) -> tuple[FloatArray, FloatArray]:
    """Cumulative event count over the recording.

    Bins ``onsets`` into ``n_bins`` equal windows over ``[0, n_samples]`` and returns
    ``(bin_right_edges, cumulative_count)`` — the cumulative feeding/activity curve.
    """
    edges = np.linspace(0, n_samples, n_bins + 1)
    counts, _ = np.histogram(np.asarray(onsets, dtype=np.float64).ravel(), bins=edges)
    return edges[1:], np.cumsum(counts).astype(np.float64)
