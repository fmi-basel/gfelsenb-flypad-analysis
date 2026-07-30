"""All-pairs significance testing across the experiment's comparison groups (design §10).

Builds one tidy ``comparisons`` table of two-sample permutation tests covering every
relevant contrast for a metric:

* **conditions** within each facet (and pooled across facets, ``strata="all"``);
* each **condition across facets** (file/substrate) when faceting is active.

Faceting follows ``config.plotting.facet_by`` (so the stats match the figures) and the
test parameters come from ``config.stats``. p-values are adjusted within each family
(one :func:`pairwise_comparisons` call per ``(contrast, strata)``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from flypad.config.models import Config
from flypad.stats.grouping import resolve_facets, with_file_labels
from flypad.stats.tests import pairwise_comparisons

FloatArray = npt.NDArray[np.float64]

COMPARISON_COLUMNS = (
    "metric",
    "contrast",
    "strata",
    "group_a",
    "group_b",
    "test",
    "statistic",
    "p_value",
    "p_adjusted",
    "n_a",
    "n_b",
)


def _groups(df: pd.DataFrame, key: str, metric: str) -> dict[str, FloatArray]:
    """Finite ``metric`` values per non-empty group of ``key`` (skips empty groups)."""
    out: dict[str, FloatArray] = {}
    for label, grp in df.groupby(key, dropna=False):
        vals = np.asarray(grp[metric].to_numpy(dtype=np.float64))
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[str(label)] = vals
    return out


def build_comparisons(
    per_fly: pd.DataFrame,
    config: Config,
    *,
    metric: str = "n_sips",
    group_col: str = "condition_label",
) -> pd.DataFrame:
    """All pairwise permutation comparisons for ``metric``, as one tidy table.

    Rows describe a two-sample test between ``group_a`` and ``group_b``:

    * ``contrast="condition"`` — two conditions within ``strata`` (a facet value, or
      ``"all"`` for the pooled test across facets);
    * ``contrast=<facet noun>`` — two facets (files/substrates) within ``strata``
      (a condition), present only when faceting is active.

    The ``test`` column names the location statistic (``config.stats.statistic``).
    """
    if group_col not in per_fly.columns:
        group_col = "condition"
    facet_by = config.plotting.facet_by
    work = with_file_labels(per_fly) if facet_by == "file" else per_fly
    facet_col, facet_values, facet_noun = resolve_facets(work, facet_by)

    stat = config.stats.statistic

    def compare(groups: dict[str, FloatArray]) -> pd.DataFrame:
        return pairwise_comparisons(
            groups,
            statistic=stat,
            n_permutations=config.stats.n_permutations,
            alternative=config.stats.alternative,
            seed=config.stats.seed,
            adjust=config.stats.multiple_comparison,
        )

    def tagged(frame: pd.DataFrame, *, contrast: str, strata: str) -> pd.DataFrame:
        out = frame.copy()
        out["metric"] = metric
        out["contrast"] = contrast
        out["strata"] = strata
        out["test"] = stat
        return out.reindex(columns=COMPARISON_COLUMNS)

    blocks: list[pd.DataFrame] = [
        tagged(compare(_groups(work, group_col, metric)), contrast="condition", strata="all"),
    ]
    if facet_col is not None:
        # conditions within each facet
        for val in facet_values:
            subset = work[work[facet_col].astype(str) == val]
            cmp = compare(_groups(subset, group_col, metric))
            blocks.append(tagged(cmp, contrast="condition", strata=val))
        # each condition across facets
        for cond, grp in work.groupby(group_col, dropna=False):
            blocks.append(
                tagged(
                    compare(_groups(grp, facet_col, metric)), contrast=facet_noun, strata=str(cond)
                )
            )

    nonempty = [b for b in blocks if not b.empty]
    if not nonempty:
        return pd.DataFrame(columns=list(COMPARISON_COLUMNS))
    return pd.concat(nonempty, ignore_index=True)
