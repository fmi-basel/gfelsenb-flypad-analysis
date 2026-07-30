"""Facet grouping: split an experiment by substrate or input file (design §10).

These helpers decide *how* the tidy tables are split into comparison groups — one
group per substrate, or one per input recording (titled by timestamp). They are pure
data operations (pandas + filename parsing, no matplotlib), so both the figures
(:mod:`flypad.plotting.facets`) and the statistics (:mod:`flypad.stats.comparisons`)
share exactly the same faceting via :func:`resolve_facets`.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from flypad.io.discovery import parse_filename

#: How data are split into one group per facet (see ``config.plotting.facet_by``).
FacetBy = Literal["substrate", "file", "none"]


def substrate_facets(df: pd.DataFrame) -> tuple[str | None, list[str]]:
    """Choose the column to facet substrates by, and its ordered distinct values.

    Prefers the named substrate identity (``substrate_label``) when two or more
    distinct non-empty labels are present; otherwise falls back to the physical
    ``substrate_side`` for an unlabeled two-choice assay. Returns ``(None, [])``
    when there is only a single substrate, so callers draw a single-axis plot.
    """
    if "substrate_label" in df.columns:
        labels = df["substrate_label"].dropna().astype(str).str.strip()
        labeled = labels[labels != ""]
        if labeled.size:
            distinct = sorted(labeled.unique())
            return ("substrate_label", distinct) if len(distinct) >= 2 else (None, [])
    if "substrate_side" in df.columns:
        sides = sorted(df["substrate_side"].dropna().astype(str).unique())
        if len(sides) >= 2:
            return "substrate_side", sides
    return None, []


def file_facet_label(file_name: object, file_index: object) -> str:
    """A short per-file facet title: the recording's ``date time`` from its filename.

    Falls back to the filename, then to ``file <index>`` when no timestamp parses.
    """
    name = str(file_name) if file_name else ""
    meta = parse_filename(name) if name else None
    if meta is not None and meta.date and meta.time:
        return f"{meta.date} {meta.time}"
    return name or f"file {file_index}"


def _file_label_map(df: pd.DataFrame) -> dict[int, str]:
    """Map each ``file_index`` to a unique timestamp label (disambiguated on collision)."""
    order = df.sort_values("file_index").drop_duplicates("file_index")
    idxs = [int(i) for i in order["file_index"]]
    names = list(order["file_name"]) if "file_name" in order.columns else [""] * len(idxs)
    labels = [file_facet_label(n, i) for n, i in zip(names, idxs, strict=True)]
    if len(set(labels)) < len(labels):  # identical timestamps -> append the file index
        labels = [f"{lbl} [{i}]" for lbl, i in zip(labels, idxs, strict=True)]
    return dict(zip(idxs, labels, strict=True))


def with_file_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``file_label`` column (timestamp per file).

    A no-op (returns ``df`` unchanged) when there is no ``file_index`` column.
    """
    if "file_index" not in df.columns:
        return df
    out = df.copy()
    out["file_label"] = out["file_index"].map(_file_label_map(df))
    return out


def resolve_facets(df: pd.DataFrame, facet_by: FacetBy) -> tuple[str | None, list[str], str]:
    """Resolve ``facet_by`` to ``(facet_col, ordered_values, noun)``.

    ``"file"`` splits by input file (titled by timestamp, chronological); ``"substrate"``
    uses :func:`substrate_facets`; ``"none"`` disables faceting. ``facet_col`` is ``None``
    (single group) when fewer than two facets are available. For ``"file"`` the caller
    must have added the ``file_label`` column via :func:`with_file_labels`.
    """
    if facet_by == "none":
        return None, [], ""
    if facet_by == "file":
        if "file_index" not in df.columns:
            return None, [], "file"
        label_map = _file_label_map(df)
        values = [label_map[idx] for idx in sorted(label_map)]
        return ("file_label", values, "file") if len(values) >= 2 else (None, [], "file")
    col, values = substrate_facets(df)
    return col, values, "substrate"
