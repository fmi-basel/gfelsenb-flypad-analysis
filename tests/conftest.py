"""Shared test configuration: force a headless matplotlib backend."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
