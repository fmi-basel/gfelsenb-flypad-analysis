"""Significance testing: permutation tests, pairwise comparisons, poly fit (design §10).

Mirrors the MATLAB v2.2 helpers ``pairwise_comparisons`` / ``pairwise_comparisons_CCDF``
(label-shuffling permutation tests) and ``FitPoly_WithRSquare`` (least-squares
polynomial fit reported with its coefficient of determination).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from flypad.stats.distributions import ccdf_linear, common_grid

FloatArray = npt.NDArray[np.float64]
Alternative = Literal["two-sided", "less", "greater"]
StatName = Literal["mean", "median"]
StatFn = Callable[[FloatArray, FloatArray], float]

_EPS = 1e-12


@dataclass
class PermutationResult:
    """Outcome of a two-sample permutation test."""

    statistic: float  # observed test statistic on the real labelling
    pvalue: float
    n_permutations: int
    n_a: int
    n_b: int


def _stat_fn(statistic: StatName) -> StatFn:
    if statistic == "mean":
        return lambda a, b: float(np.mean(a) - np.mean(b))
    if statistic == "median":
        return lambda a, b: float(np.median(a) - np.median(b))
    raise ValueError(f"unknown statistic: {statistic!r}")


def _ks_statistic(a: FloatArray, b: FloatArray) -> float:
    """Max vertical gap between the two survival curves (KS-like distance)."""
    grid = common_grid(a, b, n=256)
    return float(np.nanmax(np.abs(ccdf_linear(a, grid) - ccdf_linear(b, grid))))


def _permutation_pvalue(
    a: FloatArray,
    b: FloatArray,
    stat_fn: StatFn,
    n_permutations: int,
    alternative: Alternative,
    rng: np.random.Generator,
) -> tuple[float, float]:
    obs = stat_fn(a, b)
    pooled = np.concatenate([a, b])
    n, na = pooled.size, a.size
    perm = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        idx = rng.permutation(n)
        perm[i] = stat_fn(pooled[idx[:na]], pooled[idx[na:]])

    if alternative == "two-sided":
        count = int(np.sum(np.abs(perm) >= abs(obs) - _EPS))
    elif alternative == "greater":
        count = int(np.sum(perm >= obs - _EPS))
    else:
        count = int(np.sum(perm <= obs + _EPS))
    pvalue = (count + 1) / (n_permutations + 1)  # add-one: never reports p = 0
    return obs, pvalue


def permutation_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    statistic: StatName = "mean",
    n_permutations: int = 10_000,
    alternative: Alternative = "two-sided",
    seed: int | None = None,
) -> PermutationResult:
    """Two-sample permutation test on a difference-of-location statistic."""
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    av, bv = av[np.isfinite(av)], bv[np.isfinite(bv)]
    if av.size == 0 or bv.size == 0:
        raise ValueError("both samples must be non-empty")
    obs, pvalue = _permutation_pvalue(
        av, bv, _stat_fn(statistic), n_permutations, alternative, np.random.default_rng(seed)
    )
    return PermutationResult(obs, pvalue, n_permutations, av.size, bv.size)


def permutation_test_ccdf(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    n_permutations: int = 10_000,
    seed: int | None = None,
) -> PermutationResult:
    """Permutation test using the max CCDF gap (distribution-shape comparison)."""
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    av, bv = av[np.isfinite(av)], bv[np.isfinite(bv)]
    if av.size == 0 or bv.size == 0:
        raise ValueError("both samples must be non-empty")
    obs, pvalue = _permutation_pvalue(
        av, bv, _ks_statistic, n_permutations, "greater", np.random.default_rng(seed)
    )
    return PermutationResult(obs, pvalue, n_permutations, av.size, bv.size)


def adjust_pvalues(
    pvalues: npt.ArrayLike,
    method: Literal["none", "bonferroni", "holm"] = "holm",
) -> FloatArray:
    """Multiple-comparison adjustment of a set of p-values (clipped to ``<= 1``)."""
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    m = p.size
    if m == 0 or method == "none":
        return p.copy()
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    # Holm step-down
    order = np.argsort(p)
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def pairwise_comparisons(
    groups: Mapping[str, npt.ArrayLike],
    *,
    statistic: StatName = "mean",
    n_permutations: int = 10_000,
    alternative: Alternative = "two-sided",
    seed: int | None = None,
    adjust: Literal["none", "bonferroni", "holm"] = "holm",
    ccdf: bool = False,
) -> pd.DataFrame:
    """All-pairs permutation comparison of the named groups.

    Returns one row per unordered pair with the observed statistic, raw and adjusted
    p-values, and sample sizes. With ``ccdf=True`` the max-CCDF-gap statistic is used
    (``pairwise_comparisons_CCDF``); otherwise a difference-of-``statistic`` test.
    """
    labels = list(groups)
    rows: list[dict[str, object]] = []
    for i, la in enumerate(labels):
        for lb in labels[i + 1 :]:
            if ccdf:
                res = permutation_test_ccdf(
                    groups[la], groups[lb], n_permutations=n_permutations, seed=seed
                )
            else:
                res = permutation_test(
                    groups[la],
                    groups[lb],
                    statistic=statistic,
                    n_permutations=n_permutations,
                    alternative=alternative,
                    seed=seed,
                )
            rows.append(
                {
                    "group_a": la,
                    "group_b": lb,
                    "statistic": res.statistic,
                    "p_value": res.pvalue,
                    "n_a": res.n_a,
                    "n_b": res.n_b,
                }
            )
    frame = pd.DataFrame(rows, columns=["group_a", "group_b", "statistic", "p_value", "n_a", "n_b"])
    if not frame.empty:
        frame["p_adjusted"] = adjust_pvalues(frame["p_value"].to_numpy(), adjust)
    else:
        frame["p_adjusted"] = pd.Series(dtype="float64")
    return frame


@dataclass
class PolyFit:
    """A least-squares polynomial fit with its goodness of fit."""

    coefficients: FloatArray  # highest power first (``numpy.polyfit`` order)
    degree: int
    r_squared: float

    def predict(self, x: npt.ArrayLike) -> FloatArray:
        return np.asarray(np.polyval(self.coefficients, np.asarray(x, dtype=np.float64)))


def fit_poly_with_rsquare(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    degree: int = 1,
) -> PolyFit:
    """Fit a degree-``degree`` polynomial and report its R² (``FitPoly_WithRSquare``)."""
    xv = np.asarray(x, dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    if xv.size != yv.size:
        raise ValueError("x and y must have the same length")
    if xv.size <= degree:
        raise ValueError(f"need more than {degree} points for a degree-{degree} fit")
    coeffs = np.polyfit(xv, yv, degree)
    resid = yv - np.polyval(coeffs, xv)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    # R² = 1 - SS_res/SS_tot; if the data are constant, R² is 1 only for an exact fit.
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else float(ss_res == 0.0)
    return PolyFit(coefficients=coeffs, degree=degree, r_squared=r2)
