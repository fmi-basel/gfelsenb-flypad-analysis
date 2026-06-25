"""Figure export to PNG / EPS / SVG (design §10, M6)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_FORMATS: tuple[str, ...] = ("png", "eps", "svg")


def save_figure(
    fig: Any,
    path: str | Path,
    *,
    formats: Sequence[str] | None = None,
    dpi: int = 300,
    transparent: bool = False,
    close: bool = False,
    **savefig_kwargs: Any,
) -> list[Path]:
    """Save ``fig`` to one or more formats.

    If ``path`` already has a suffix, that single file is written. Otherwise ``path``
    is treated as a stem and one file per entry in ``formats`` (default PNG/EPS/SVG) is
    written. Returns the paths written.
    """
    out = Path(path)
    targets = (
        [out]
        if out.suffix
        else [out.with_suffix(f".{fmt}") for fmt in (formats or DEFAULT_FORMATS)]
    )
    written: list[Path] = []
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=dpi, transparent=transparent, **savefig_kwargs)
        written.append(target)
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return written
