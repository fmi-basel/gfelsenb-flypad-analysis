"""Declarative stage registry and runner (design §5).

A *stage* is a pure callable ``(Context, Config) -> Context``. Stages register
themselves by name; the execution order comes from config. This tiny module is
the entire pipeline abstraction — adding a stage is one decorated function,
changing behaviour is one strategy + one config field.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from flypad.config.models import Config
from flypad.pipeline.context import Context

StageFn = Callable[[Context, Config], Context]

REGISTRY: dict[str, StageFn] = {}


def register(name: str) -> Callable[[StageFn], StageFn]:
    """Decorator: register a stage function under ``name``."""

    def deco(fn: StageFn) -> StageFn:
        if name in REGISTRY:
            raise ValueError(f"stage already registered: {name!r}")
        REGISTRY[name] = fn
        return fn

    return deco


def run(
    context: Context,
    config: Config,
    order: Sequence[str],
    on_progress: Callable[[str], None] | None = None,
) -> Context:
    """Execute the named stages in order, threading the context through.

    ``on_progress`` (if given) is called with each stage name after it runs —
    a plain callback, so the core never imports Qt.
    """
    for name in order:
        try:
            stage = REGISTRY[name]
        except KeyError as exc:
            raise KeyError(f"unknown stage: {name!r} (registered: {sorted(REGISTRY)})") from exc
        context = stage(context, config)
        if on_progress is not None:
            on_progress(name)
    return context


def registered_stages() -> list[str]:
    """Sorted names of all registered stages."""
    return sorted(REGISTRY)
