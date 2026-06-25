"""flypad — Python analysis suite for flyPAD capacitance recordings.

See ``flypad_new_software_design.html`` for the architecture and the decisions log.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flypad")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
