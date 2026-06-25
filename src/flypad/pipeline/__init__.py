"""Pipeline orchestration: stage registry + runner (design §5)."""

from flypad.pipeline.context import Context
from flypad.pipeline.stages import REGISTRY, register, registered_stages, run

__all__ = ["REGISTRY", "Context", "register", "registered_stages", "run"]
