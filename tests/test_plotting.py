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
    condition_palette,
    cumulative_timecourse_plot,
    distinguishable_colors,
    faceted_boxplot,
    faceted_ccdf,
    faceted_dashboard,
    faceted_timecourse,
    jbfill,
    raster_plot,
    resolve_facets,
    save_figure,
    shaded_lines,
    shaded_plot,
    standalone_dashboard,
    substrate_comparison,
    substrate_facets,
    theme_context,
    tight_subplot,
    tilted_boxplot,
    with_file_labels,
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
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert [lbl.split("\n")[0] for lbl in labels] == ["fed", "starved", "refed"]
    assert all(lbl.endswith("n=20") for lbl in labels)  # count folded into the tick label


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


def test_standalone_dashboard_annotates_box_panel() -> None:
    fig = standalone_dashboard(_per_fly(), "n_sips", annotations=[("fed", "starved", 0.0001)])
    assert "***" in [t.get_text() for t in fig.axes[0].texts]  # bracket on the box panel


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


def test_save_figure_default_formats_are_png_and_pdf(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = save_figure(fig, tmp_path / "fig")
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_plotting_config_defaults_to_pdf() -> None:
    from flypad.config.models import Plotting

    assert Plotting().vector_format == "pdf"


# --------------------------------------------------------------------------- #
# M6 improvements: palette, substrate, tilt/N, units, time-course bands
# --------------------------------------------------------------------------- #
def test_condition_palette_is_stable_by_label() -> None:
    p1 = condition_palette(["starved", "fed", "refed"])
    p2 = condition_palette(["fed", "refed", "starved"])  # different order
    assert p1 == p2  # keyed by sorted label, order-independent
    assert set(p1) == {"fed", "refed", "starved"}


def test_tilted_boxplot_tilt_and_n_annotation() -> None:
    groups = {"a": [1.0, 2, 3, 4], "b": [2.0, 3, 4, 5]}
    ax = tilted_boxplot(groups, tilt_deg=12.0, show_n=True)
    assert len(ax.patches) == 2  # boxes still drawn when tilted
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert any("n=4" in lbl for lbl in labels)  # N folded into the tick label


def test_tilted_boxplot_show_n_false_omits_count() -> None:
    ax = tilted_boxplot({"a": [1.0, 2, 3], "b": [4.0, 5]}, show_n=False)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["a", "b"]  # no "n=" appended


def test_metric_label_known_and_fallback() -> None:
    from flypad.plotting import metric_label

    assert metric_label("n_sips") == "Sips per fly"
    assert metric_label("some_new_metric") == "Some new metric"  # prettified fallback


def test_time_axis_picks_unit() -> None:
    from flypad.plotting import time_axis

    _v, lbl = time_axis(np.array([0.0, 60.0]))
    assert lbl == "Time (s)"
    v, lbl = time_axis(np.array([0.0, 600.0]))
    assert lbl == "Time (min)" and v[-1] == pytest.approx(10.0)
    v, lbl = time_axis(np.array([0.0, 10800.0]))
    assert lbl == "Time (h)" and v[-1] == pytest.approx(3.0)


def test_swarm_offsets_spread_and_bounds() -> None:
    from flypad.plotting.boxplots import swarm_offsets

    off = swarm_offsets(np.zeros(9), 0.1)  # all identical -> maximal spread
    assert np.abs(off).max() <= 0.1 + 1e-9
    assert off.min() < 0 < off.max()  # symmetric about the centre
    assert swarm_offsets(np.array([1.0]), 0.1)[0] == 0.0  # a lone point stays centred


def test_tilted_boxplot_symlog_scale() -> None:
    ax = tilted_boxplot({"a": [0.0, 1, 10], "b": [100.0, 1000, 5000]}, yscale="symlog")
    assert ax.get_yscale() == "symlog"


def test_tilted_boxplot_log_falls_back_to_symlog_with_zeros() -> None:
    ax = tilted_boxplot({"a": [0.0, 1, 10]}, yscale="log")
    assert ax.get_yscale() == "symlog"  # zeros are unrepresentable on a pure log axis


def test_annotations_only_significant_and_pvalues() -> None:
    groups = {"a": [1.0, 2, 3], "b": [8.0, 9, 10]}
    ax = tilted_boxplot(groups, annotations=[("a", "b", 0.4)], only_significant=True)
    assert "n.s." not in [t.get_text() for t in ax.texts]  # dropped
    ax2 = tilted_boxplot(groups, annotations=[("a", "b", 0.012)], show_pvalues=True)
    assert "p=0.012" in [t.get_text() for t in ax2.texts]


def test_significance_marker_thresholds() -> None:
    from flypad.plotting import significance_marker

    assert significance_marker(0.0005) == "***"
    assert significance_marker(0.005) == "**"
    assert significance_marker(0.03) == "*"
    assert significance_marker(0.2) == "n.s."


def test_tilted_boxplot_annotations_draw_brackets() -> None:
    ax = tilted_boxplot({"a": [1.0, 2, 3], "b": [8.0, 9, 10]}, annotations=[("a", "b", 0.0001)])
    assert "***" in [t.get_text() for t in ax.texts]  # significance marker drawn


def test_tilted_boxplot_palette_colors_boxes() -> None:
    from matplotlib.colors import to_rgb

    palette = condition_palette(["a", "b"])
    ax = tilted_boxplot({"a": [1.0, 2, 3], "b": [4.0, 5, 6]}, palette=palette, show_points=False)
    facecolor = ax.patches[0].get_facecolor()[:3]
    assert np.allclose(facecolor, to_rgb(palette["a"]), atol=1e-6)


def test_condition_palette_uses_wong_colors() -> None:
    from flypad.plotting.theme import WONG

    palette = condition_palette(["fed", "starved"])
    assert set(palette.values()) <= set(WONG)  # colour-blind-safe, not the neon RGB grid


def test_condition_palette_preserves_order_when_unsorted() -> None:
    palette = condition_palette(["starved", "fed"], sort=False)
    assert list(palette) == ["starved", "fed"]


def test_substrate_comparison_two_boxes_per_condition() -> None:
    per_fly = pd.DataFrame(
        {
            "condition_label": ["a", "a", "b", "b"],
            "substrate_side": ["left", "right", "left", "right"],
            "n_sips": [10.0, 4.0, 8.0, 6.0],
        }
    )
    ax = substrate_comparison(per_fly, "n_sips", ylabel="n_sips")
    assert len(ax.patches) == 4  # 2 conditions x (left, right)
    assert ax.get_legend() is not None


def test_raster_seconds_axis() -> None:
    ax = raster_plot([np.array([100, 200])], sampling_rate_hz=100)  # 1-2 s
    assert ax.get_xlabel() == "Time (s)"


def test_raster_switches_to_minutes_for_long_recordings() -> None:
    ax = raster_plot([np.array([0, 259_209])], sampling_rate_hz=100)  # ~43 min
    assert ax.get_xlabel() == "Time (min)"
    assert ax.collections[0].get_positions()[-1] == pytest.approx(43.2, abs=0.1)


def test_raster_sample_axis_without_rate() -> None:
    ax = raster_plot([np.array([100, 200])])
    assert ax.get_xlabel() == "sample"


def test_shaded_lines_band_per_series() -> None:
    series = {
        "a": (np.arange(5.0), np.arange(5.0), np.ones(5)),
        "b": (np.arange(5.0), 2 * np.arange(5.0), np.ones(5)),
    }
    ax = shaded_lines(series, palette=condition_palette(["a", "b"]))
    assert len(ax.lines) == 2
    assert len(ax.collections) == 2  # one shaded band per series


def test_raster_colored_by_condition() -> None:
    palette = condition_palette(["fed", "starved"])
    rows = [np.array([10, 50]), np.array([20]), np.array([30, 70])]
    ax = raster_plot(rows, row_conditions=["fed", "starved", "fed"], palette=palette)
    assert ax.get_legend() is not None  # one entry per condition present
    assert len(ax.get_legend().get_texts()) == 2


# --------------------------------------------------------------------------- #
# substrate faceting: one axis per substrate for box plot / CCDF / time course
# --------------------------------------------------------------------------- #
def _per_fly_two_substrates() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    for label in ("sucrose", "yeast"):
        for cond in ("fed", "starved"):
            for _ in range(12):
                rows.append(
                    {
                        "condition_label": cond,
                        "substrate_label": label,
                        "substrate_side": "left" if label == "sucrose" else "right",
                        "n_sips": float(rng.integers(0, 200)),
                    }
                )
    return pd.DataFrame(rows)


def _events_two_substrates() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for fi, label in enumerate(("sucrose", "yeast")):
        for ch, cond in enumerate(("fed", "starved")):
            for onset in rng.integers(0, 1000, size=10):
                rows.append(
                    {
                        "file_index": fi,
                        "channel": ch,
                        "condition_label": cond,
                        "substrate_label": label,
                        "onset": int(onset),
                    }
                )
    return pd.DataFrame(rows)


def test_substrate_facets_prefers_labels() -> None:
    col, values = substrate_facets(_per_fly_two_substrates())
    assert col == "substrate_label"
    assert values == ["sucrose", "yeast"]


def test_substrate_facets_falls_back_to_side_when_unlabeled() -> None:
    per_fly = pd.DataFrame(
        {
            "condition_label": ["a", "a"],
            "substrate_label": ["", ""],
            "substrate_side": ["left", "right"],
            "n_sips": [1.0, 2.0],
        }
    )
    col, values = substrate_facets(per_fly)
    assert col == "substrate_side"
    assert values == ["left", "right"]


def test_substrate_facets_none_for_single_substrate() -> None:
    per_fly = pd.DataFrame(
        {
            "condition_label": ["a", "b"],
            "substrate_label": ["sucrose", "sucrose"],
            "substrate_side": ["left", "left"],
            "n_sips": [1.0, 2.0],
        }
    )
    assert substrate_facets(per_fly) == (None, [])


def test_faceted_boxplot_one_axis_per_substrate() -> None:
    per_fly = _per_fly_two_substrates()
    palette = condition_palette(per_fly["condition_label"])
    fig = faceted_boxplot(
        per_fly,
        "n_sips",
        facet_col="substrate_label",
        values=["sucrose", "yeast"],
        palette=palette,
        ylabel="n_sips",
    )
    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ["sucrose", "yeast"]
    # 2 conditions -> 2 boxes on each substrate axis
    assert all(len(ax.patches) == 2 for ax in fig.axes)


def test_faceted_ccdf_shares_one_legend() -> None:
    per_fly = _per_fly_two_substrates()
    palette = condition_palette(per_fly["condition_label"])
    fig = faceted_ccdf(
        per_fly,
        "n_sips",
        facet_col="substrate_label",
        values=["sucrose", "yeast"],
        palette=palette,
    )
    assert len(fig.axes) == 2
    assert fig.legends and len(fig.legends[0].get_texts()) == 2  # one shared legend, 2 conditions
    assert all(ax.get_legend() is None for ax in fig.axes)  # no per-axis legends


def test_faceted_timecourse_one_axis_per_substrate() -> None:
    events = _events_two_substrates()
    palette = condition_palette(events["condition_label"])
    fig = faceted_timecourse(
        events,
        1000,
        facet_col="substrate_label",
        values=["sucrose", "yeast"],
        sampling_rate_hz=100,
        palette=palette,
    )
    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ["sucrose", "yeast"]
    assert fig.legends and len(fig.legends[0].get_texts()) == 2


# --------------------------------------------------------------------------- #
# file / timestamp faceting: one axis per input recording
# --------------------------------------------------------------------------- #
def _per_fly_two_files() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    names = {
        0: "CapacitanceData_C01_01_96_2026-07-09T10_01_52.333+02_00",
        1: "CapacitanceData_C01_01_96_2026-07-09T15_43_50.831+02_00",
    }
    rows = []
    for fi in (0, 1):
        for cond in ("fed", "starved"):
            for _ in range(10):
                rows.append(
                    {
                        "file_index": fi,
                        "file_name": names[fi],
                        "condition_label": cond,
                        "n_sips": float(rng.integers(0, 200)),
                    }
                )
    return pd.DataFrame(rows)


def test_with_file_labels_uses_timestamp() -> None:
    out = with_file_labels(_per_fly_two_files())
    assert set(out["file_label"]) == {"2026-07-09 10:01:52", "2026-07-09 15:43:50"}


def test_with_file_labels_noop_without_file_index() -> None:
    df = pd.DataFrame({"condition_label": ["a"], "n_sips": [1.0]})
    assert with_file_labels(df) is df


def test_resolve_facets_file_orders_by_index() -> None:
    col, values, noun = resolve_facets(_per_fly_two_files(), "file")
    assert col == "file_label" and noun == "file"
    assert values == ["2026-07-09 10:01:52", "2026-07-09 15:43:50"]  # chronological


def test_resolve_facets_none_disables() -> None:
    assert resolve_facets(_per_fly_two_substrates(), "none") == (None, [], "")


def test_resolve_facets_single_file_is_single_axis() -> None:
    one = _per_fly_two_files()
    one = one[one["file_index"] == 0]
    assert resolve_facets(one, "file")[0] is None  # only one file -> no faceting


def test_faceted_boxplot_by_file_titled_with_timestamp() -> None:
    per_fly = with_file_labels(_per_fly_two_files())
    col, values, noun = resolve_facets(per_fly, "file")
    palette = condition_palette(per_fly["condition_label"])
    fig = faceted_boxplot(
        per_fly, "n_sips", facet_col=col, values=values, palette=palette, noun=noun
    )
    assert [ax.get_title() for ax in fig.axes] == ["2026-07-09 10:01:52", "2026-07-09 15:43:50"]
    assert fig._suptitle.get_text() == "Sips per fly by file"


def test_faceted_dashboard_row_per_facet() -> None:
    per_fly = _per_fly_two_substrates()
    palette = condition_palette(per_fly["condition_label"])
    fig = faceted_dashboard(
        per_fly,
        "n_sips",
        facet_col="substrate_label",
        values=["sucrose", "yeast"],
        palette=palette,
        central="median",
    )
    assert len(fig.axes) == 6  # 2 facet rows x 3 panels
    # leftmost panel of each row is titled with the facet value; right column is the CCDF
    assert fig.axes[0].get_title() == "sucrose"
    assert fig.axes[3].get_title() == "yeast"
    assert fig.axes[2].get_title() == "CCDF" and fig.axes[5].get_title() == "CCDF"
    assert fig._suptitle.get_text() == "Sips per fly dashboard by substrate"


def test_faceted_boxplot_annotations_per_facet() -> None:
    per_fly = _per_fly_two_substrates()
    palette = condition_palette(per_fly["condition_label"])
    fig = faceted_boxplot(
        per_fly,
        "n_sips",
        facet_col="substrate_label",
        values=["sucrose", "yeast"],
        palette=palette,
        annotations_by_facet={
            "sucrose": [("fed", "starved", 0.0001)],
            "yeast": [("fed", "starved", 0.9)],
        },
    )
    assert "***" in [t.get_text() for t in fig.axes[0].texts]  # sucrose facet significant
    assert "n.s." in [t.get_text() for t in fig.axes[1].texts]  # yeast facet not


def test_faceted_dashboard_by_file_mean_variant() -> None:
    per_fly = with_file_labels(_per_fly_two_files())
    col, values, noun = resolve_facets(per_fly, "file")
    fig = faceted_dashboard(
        per_fly, "n_sips", facet_col=col, values=values, central="mean", noun=noun
    )
    assert len(fig.axes) == 6
    assert fig.axes[0].get_title() == "2026-07-09 10:01:52"
    assert fig.axes[1].get_title() == "mean ± 95% CI"
    assert fig._suptitle.get_text() == "Sips per fly dashboard by file"


def test_faceted_dashboard_annotates_box_panels() -> None:
    per_fly = _per_fly_two_substrates()
    palette = condition_palette(per_fly["condition_label"])
    fig = faceted_dashboard(
        per_fly,
        "n_sips",
        facet_col="substrate_label",
        values=["sucrose", "yeast"],
        palette=palette,
        annotations_by_facet={
            "sucrose": [("fed", "starved", 0.0001)],
            "yeast": [("fed", "starved", 0.9)],
        },
    )
    # box panels are the first column of each row: axes[0] (row 0) and axes[3] (row 1)
    assert "***" in [t.get_text() for t in fig.axes[0].texts]
    assert "n.s." in [t.get_text() for t in fig.axes[3].texts]
