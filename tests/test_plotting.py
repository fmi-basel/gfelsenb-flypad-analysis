"""M6: plotting — theme/palette, box plots, time courses, CDFs, rasters, export."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from flypad.plotting import (
    ccdf_plot,
    cdf_plot,
    cumulative_timecourse_plot,
    distinguishable_colors,
    jbfill,
    raster_plot,
    save_figure,
    shaded_plot,
    standalone_dashboard,
    theme_context,
    tight_subplot,
    tilted_boxplot,
)
from flypad.plotting.theme import _srgb_to_lab


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _groups() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "fed": rng.normal(10, 2, 20),
        "starved": rng.normal(18, 3, 20),
        "refed": rng.normal(12, 2, 20),
    }


# --------------------------------------------------------------------------- #
# theme / palette
# --------------------------------------------------------------------------- #
def test_distinguishable_colors_count_and_range() -> None:
    cols = distinguishable_colors(8)
    assert len(cols) == 8
    arr = np.asarray(cols)
    assert arr.shape == (8, 3)
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_distinguishable_colors_are_distinct() -> None:
    cols = distinguishable_colors(6)
    # every pair differs noticeably in Lab space
    lab = _srgb_to_lab(np.asarray(cols))
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            assert np.linalg.norm(lab[i] - lab[j]) > 10.0


def test_distinguishable_colors_empty() -> None:
    assert distinguishable_colors(0) == []


def test_srgb_to_lab_white_and_black() -> None:
    lab = _srgb_to_lab(np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
    assert lab[0, 0] == pytest.approx(100.0, abs=0.5)  # white L* ≈ 100
    assert lab[1, 0] == pytest.approx(0.0, abs=0.5)  # black L* ≈ 0


def test_tight_subplot_grid_shape() -> None:
    _fig, axes = tight_subplot(2, 3, figsize=(9, 5))
    assert np.asarray(axes).shape == (2, 3)


def test_theme_context_restores_rcparams() -> None:
    import matplotlib as mpl

    before = mpl.rcParams["axes.spines.top"]
    with theme_context():
        assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.top"] == before


# --------------------------------------------------------------------------- #
# box plots
# --------------------------------------------------------------------------- #
def test_tilted_boxplot_draws_boxes_and_points() -> None:
    ax = tilted_boxplot(_groups(), ylabel="n_sips")
    assert len(ax.patches) == 3  # one box per group
    assert len(ax.collections) >= 3  # jittered point clouds
    assert [t.get_text() for t in ax.get_xticklabels()] == ["fed", "starved", "refed"]


def test_tilted_boxplot_skips_empty_groups() -> None:
    groups = {"a": [1.0, 2, 3, 4], "b": []}
    ax = tilted_boxplot(groups, show_points=False)
    assert len(ax.patches) == 1  # only the non-empty group gets a box


def test_median_iqr_plot_one_errorbar_per_group() -> None:
    from flypad.plotting import median_iqr_plot

    ax = median_iqr_plot(_groups())
    assert len(ax.containers) == 3  # three errorbar containers


# --------------------------------------------------------------------------- #
# time courses
# --------------------------------------------------------------------------- #
def test_jbfill_adds_polycollection() -> None:
    _fig, ax = plt.subplots()
    x = np.linspace(0, 1, 10)
    before = len(ax.collections)
    jbfill(ax, x, x - 0.1, x + 0.1)
    assert len(ax.collections) == before + 1


def test_shaded_plot_line_plus_band() -> None:
    _fig, ax = plt.subplots()
    x = np.arange(10.0)
    shaded_plot(ax, x, x, np.ones_like(x), label="cond")
    assert len(ax.lines) == 1
    assert len(ax.collections) == 1


def test_cumulative_timecourse_plot_one_line_per_curve() -> None:
    curves = {
        "fed": (np.arange(5), np.array([0, 1, 2, 3, 4])),
        "starved": (np.arange(5), np.array([0, 2, 4, 6, 8])),
    }
    ax = cumulative_timecourse_plot(curves)
    assert len(ax.lines) == 2


# --------------------------------------------------------------------------- #
# CDF / CCDF + dashboard
# --------------------------------------------------------------------------- #
def test_cdf_plot_monotone_in_bounds() -> None:
    ax = cdf_plot(_groups())
    assert len(ax.lines) == 3
    for line in ax.lines:
        y = line.get_ydata()
        assert np.all(np.diff(y) >= -1e-9)  # non-decreasing CDF
        assert y.max() <= 1.0 + 1e-9


def test_ccdf_plot_decreasing() -> None:
    ax = ccdf_plot(_groups())
    for line in ax.lines:
        y = line.get_ydata()
        assert np.all(np.diff(y) <= 1e-9)  # non-increasing survival


def _per_fly() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for cond in ("fed", "starved"):
        for _ in range(15):
            rows.append({"condition_label": cond, "n_sips": float(rng.integers(0, 200))})
    return pd.DataFrame(rows)


def test_standalone_dashboard_has_three_panels() -> None:
    fig = standalone_dashboard(_per_fly(), "n_sips", central="median", title="20240215")
    assert len(fig.axes) == 3


def test_standalone_dashboard_mean_variant() -> None:
    fig = standalone_dashboard(_per_fly(), "n_sips", central="mean")
    assert len(fig.axes) == 3


# --------------------------------------------------------------------------- #
# rasters
# --------------------------------------------------------------------------- #
def test_raster_plot_rows() -> None:
    rows = [np.array([10, 50, 90]), np.array([20, 40]), np.array([])]
    ax = raster_plot(rows, row_labels=["c0", "c1", "c2"])
    assert len(ax.collections) == 3  # one eventplot collection per row
    assert ax.get_ylim()[1] == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def test_save_figure_multiple_formats(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = save_figure(fig, tmp_path / "fig", formats=("png", "svg"))
    assert [p.suffix for p in paths] == [".png", ".svg"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_save_figure_single_suffix(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    paths = save_figure(fig, tmp_path / "one.png")
    assert len(paths) == 1 and paths[0].name == "one.png"
