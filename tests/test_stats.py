"""M5: stats — distributions, significance tests, and per-fly/condition summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flypad.config.models import NonEaters
from flypad.detect.results import ChannelBouts, ChannelSips
from flypad.postprocess.bursts import group_feeding_bursts
from flypad.stats import (
    METRIC_COLUMNS,
    apply_qc_removal,
    build_event_table,
    ccdf,
    channel_metrics,
    cum_difference,
    cumulative_time_course,
    dots_table,
    fit_poly_with_rsquare,
    icdf,
    icdf_linear,
    mark_non_eaters,
    pairwise_comparisons,
    per_condition_summary,
    per_fly_summary,
    permutation_test,
    permutation_test_ccdf,
    quantile,
    summarize_experiment,
)
from flypad.stats.tests import adjust_pvalues

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample" / "20240215"
SAMPLE_MAT = SAMPLE_DIR / "20240215.mat"
RAW_FILES = sorted(SAMPLE_DIR.glob("CapacitanceData_*"))
RATE = 100


def _sips(onsets: list[int], offsets: list[int]) -> ChannelSips:
    return ChannelSips(
        onsets=np.asarray(onsets, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
    )


def _bouts(n: int) -> ChannelBouts:
    on = np.arange(n, dtype=np.int64) * 100
    return ChannelBouts(onsets=on, offsets=on + 50)


# --------------------------------------------------------------------------- #
# distributions
# --------------------------------------------------------------------------- #
def test_icdf_and_ccdf_are_complementary() -> None:
    x, f = icdf([4, 1, 3, 2])
    assert x.tolist() == [1, 2, 3, 4]
    np.testing.assert_allclose(f, [0.25, 0.5, 0.75, 1.0])
    xc, s = ccdf([4, 1, 3, 2])
    assert xc.tolist() == [1, 2, 3, 4]
    np.testing.assert_allclose(s, [1.0, 0.75, 0.5, 0.25])  # P(X >= x)


def test_icdf_linear_matches_uniform_cdf() -> None:
    rng = np.random.default_rng(0)
    vals = rng.uniform(0, 1, 20_000)
    grid = np.linspace(0, 1, 11)
    np.testing.assert_allclose(icdf_linear(vals, grid), grid, atol=0.02)


def test_cum_difference_zero_for_same_sample() -> None:
    vals = [1.0, 2.0, 3.0, 4.0]
    grid = np.linspace(0, 5, 7)
    np.testing.assert_allclose(cum_difference(vals, vals, grid), 0.0)


def test_quantile_inverts_cdf() -> None:
    vals = np.arange(0, 101)
    np.testing.assert_allclose(quantile(vals, [0.0, 0.5, 1.0]), [0, 50, 100])


def test_cumulative_time_course_is_monotone_and_totals() -> None:
    onsets = [10, 20, 30, 90, 95]
    _edges, cum = cumulative_time_course(onsets, n_samples=100, n_bins=10)
    assert cum[-1] == len(onsets)
    assert np.all(np.diff(cum) >= 0)


def test_distributions_handle_empty() -> None:
    x, f = icdf([])
    assert x.size == 0 and f.size == 0
    assert np.isnan(icdf_linear([], [0.0, 1.0])).all()


# --------------------------------------------------------------------------- #
# significance tests
# --------------------------------------------------------------------------- #
def test_permutation_test_separates_distinct_groups() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 40)
    b = rng.normal(4, 1, 40)
    res = permutation_test(a, b, n_permutations=2000, seed=7)
    assert res.pvalue < 0.01
    assert res.n_a == 40 and res.n_b == 40


def test_permutation_test_identical_groups_not_significant() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 60)
    res = permutation_test(base[:30], base[30:], n_permutations=2000, seed=3)
    assert res.pvalue > 0.05


def test_permutation_test_is_deterministic_with_seed() -> None:
    a, b = [1.0, 2, 3, 4, 5], [3.0, 4, 5, 6, 7]
    r1 = permutation_test(a, b, n_permutations=500, seed=42)
    r2 = permutation_test(a, b, n_permutations=500, seed=42)
    assert r1.pvalue == r2.pvalue


def test_permutation_test_ccdf_detects_shape_difference() -> None:
    rng = np.random.default_rng(4)
    a = rng.exponential(1.0, 80)
    b = rng.exponential(4.0, 80)
    res = permutation_test_ccdf(a, b, n_permutations=1000, seed=5)
    assert 0.0 <= res.statistic <= 1.0
    assert res.pvalue < 0.05


def test_pairwise_comparisons_shape_and_adjust() -> None:
    rng = np.random.default_rng(6)
    groups = {
        "fed": rng.normal(0, 1, 25),
        "starved": rng.normal(3, 1, 25),
        "refed": rng.normal(0.2, 1, 25),
    }
    table = pairwise_comparisons(groups, n_permutations=1000, seed=1)
    assert len(table) == 3  # 3 choose 2
    assert {"group_a", "group_b", "statistic", "p_value", "p_adjusted"} <= set(table.columns)
    assert (table["p_adjusted"] >= table["p_value"] - 1e-9).all()


def test_adjust_pvalues_methods() -> None:
    p = [0.01, 0.02, 0.03]
    np.testing.assert_allclose(adjust_pvalues(p, "bonferroni"), [0.03, 0.06, 0.09])
    np.testing.assert_allclose(adjust_pvalues(p, "none"), p)
    holm = adjust_pvalues(p, "holm")
    assert holm[0] == pytest.approx(0.03)  # 3 * 0.01


def test_fit_poly_with_rsquare_perfect_line() -> None:
    x = np.arange(10, dtype=float)
    y = 2 * x + 1
    fit = fit_poly_with_rsquare(x, y, degree=1)
    np.testing.assert_allclose(fit.coefficients, [2.0, 1.0])
    assert fit.r_squared == pytest.approx(1.0)
    np.testing.assert_allclose(fit.predict([100.0]), [201.0])


def test_fit_poly_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="degree"):
        fit_poly_with_rsquare([1.0], [1.0], degree=1)


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
def test_channel_metrics_values() -> None:
    sips = _sips([0, 30, 60, 90, 120], [20, 50, 80, 110, 140])
    bursts = group_feeding_bursts(sips, ifi_threshold=20.0, min_sips=3)
    m = channel_metrics(sips, _bouts(3), bursts, RATE)
    assert m["n_sips"] == 5
    assert m["total_sip_duration_ms"] == pytest.approx(1000.0)  # 100 samples * 10 ms
    assert m["mean_sip_duration_ms"] == pytest.approx(200.0)
    assert m["n_feeding_bursts"] == 1
    assert m["n_sips_in_bursts"] == 5 and m["n_isolated_sips"] == 0
    assert m["mean_burst_size"] == pytest.approx(5.0)
    assert m["n_activity_bouts"] == 3
    assert m["latency_first_sip_s"] == pytest.approx(0.0)


def test_channel_metrics_empty_channel() -> None:
    empty = _sips([], [])
    bursts = group_feeding_bursts(empty, 20.0, 3)
    m = channel_metrics(empty, _bouts(0), bursts, RATE)
    assert m["n_sips"] == 0 and m["total_sip_duration_ms"] == 0.0
    assert np.isnan(m["mean_sip_duration_ms"])
    assert np.isnan(m["latency_first_sip_s"])


def _toy_experiment() -> tuple[list, list, list, pd.DataFrame]:
    """One file, four channels = two arenas; arena (2,3) is a non-eater pair."""
    sip_specs = {
        0: ([0, 30, 60, 90, 120], [20, 50, 80, 110, 140]),  # eater, 1 burst
        1: ([0, 40, 600], [20, 60, 620]),  # few sips
        2: ([], []),  # silent
        3: ([10], [25]),  # one sip
    }
    bout_counts = {0: 10, 1: 5, 2: 0, 3: 1}
    sips = [_sips(*sip_specs[c]) for c in range(4)]
    bouts = [_bouts(bout_counts[c]) for c in range(4)]
    bursts = [group_feeding_bursts(s, 20.0, 3) for s in sips]
    channel_map = pd.DataFrame(
        {
            "file_index": [0, 0, 0, 0],
            "file_name": ["f0"] * 4,
            "exp_file": ["exp_0.txt"] * 4,
            "board_position": [1, 1, 1, 1],
            "channel": [0, 1, 2, 3],
            "condition": [1, 1, 2, 2],
            "condition_label": ["fed", "fed", "starved", "starved"],
            "condition_short": ["fed", "fed", "stv", "stv"],
            "sex": ["F", "F", "F", "F"],
            "substrate": [1, 2, 1, 2],
            "substrate_side": ["left", "right", "left", "right"],
            "substrate_label": ["suc"] * 4,
        }
    )
    return [sips], [bouts], [bursts], channel_map


def test_per_fly_summary_columns_and_counts() -> None:
    sips, bouts, bursts, cmap = _toy_experiment()
    pf = per_fly_summary(sips, bouts, bursts, cmap, RATE)
    assert len(pf) == 4
    assert set(METRIC_COLUMNS) <= set(pf.columns)
    assert pf.loc[pf.channel == 0, "n_sips"].iloc[0] == 5
    assert pf.loc[pf.channel == 2, "n_sips"].iloc[0] == 0
    assert pf["n_sips"].dtype == np.int64


def test_build_event_table_one_row_per_sip() -> None:
    sips, _bts, bursts, cmap = _toy_experiment()
    table = build_event_table(sips, bursts, cmap, RATE)
    assert len(table) == 5 + 3 + 0 + 1  # sip counts per channel
    ch0 = table[table.channel == 0]
    assert ch0["in_burst"].all()  # all 5 form one burst
    assert (ch0["burst_id"] == 0).all()
    ch3 = table[table.channel == 3]
    assert not ch3["in_burst"].any()  # single sip is isolated
    assert (table["duration_ms"] > 0).all()


def test_mark_and_remove_non_eaters_global() -> None:
    sips, bouts, bursts, cmap = _toy_experiment()
    pf = per_fly_summary(sips, bouts, bursts, cmap, RATE)
    marked = mark_non_eaters(
        pf, NonEaters(remove_global=True, remove_substrate=False, threshold_bouts=2)
    )
    # arena (2,3): bouts 0 + 1 = 1 <= 2 -> both removed; arena (0,1): 10 + 5 kept
    assert marked.loc[marked.channel == 2, "non_eater"].iloc[0]
    assert marked.loc[marked.channel == 3, "non_eater"].iloc[0]
    assert not marked.loc[marked.channel == 0, "non_eater"].iloc[0]
    kept = apply_qc_removal(marked)
    assert sorted(kept["channel"]) == [0, 1]


def test_per_condition_summary_aggregates() -> None:
    sips, bouts, bursts, cmap = _toy_experiment()
    pf = per_fly_summary(sips, bouts, bursts, cmap, RATE)
    pc = per_condition_summary(pf, metrics=("n_sips",))
    fed = pc[(pc.condition == 1) & (pc.metric == "n_sips")].iloc[0]
    assert fed["n"] == 2
    assert fed["mean"] == pytest.approx((5 + 3) / 2)
    assert {"mean", "median", "std", "sem", "ci_low", "ci_high"} <= set(pc.columns)


def test_dots_table_padding() -> None:
    sips, bouts, bursts, cmap = _toy_experiment()
    pf = per_fly_summary(sips, bouts, bursts, cmap, RATE)
    dots = dots_table(pf, "n_sips", group_col="condition_label")
    assert set(dots.columns) == {"fed", "starved"}
    assert len(dots) == 2  # two flies per condition


def test_summarize_experiment_applies_qc() -> None:
    sips, bouts, bursts, cmap = _toy_experiment()
    summary = summarize_experiment(
        sips, bouts, bursts, cmap, RATE, non_eaters=NonEaters(threshold_bouts=2)
    )
    assert sorted(summary.per_fly["channel"]) == [0, 1]  # non-eater arena dropped
    assert not summary.events.empty
    assert "starved" not in set(summary.per_condition["condition_label"])


def test_export_table_csv_roundtrip(tmp_path: Path) -> None:
    sips, bouts, bursts, cmap = _toy_experiment()
    pf = per_fly_summary(sips, bouts, bursts, cmap, RATE)
    out = pf.pipe(lambda d: d)
    from flypad.stats import export_table

    path = export_table(out, tmp_path / "per_fly.csv")
    assert path.exists()
    reloaded = pd.read_csv(path)
    assert len(reloaded) == len(pf)


# --------------------------------------------------------------------------- #
# parity: aggregate metrics from Python detection vs the MATLAB .mat events
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (RAW_FILES and SAMPLE_MAT.exists()),
    reason="raw sample binaries / .mat not present",
)
def test_aggregate_metrics_match_matlab_events() -> None:
    """Total sips & total feeding duration agree with the MATLAB ground truth."""
    from flypad.config import load_config
    from flypad.datamodel import load_recording
    from flypad.detect.run import detect_recording
    from flypad.io import read_events_mat

    cfg = load_config(REPO_ROOT / "configs" / "example_experiment.yaml", preset="matlab_compat")
    mat = read_events_mat(SAMPLE_MAT)

    py_sips = py_dur = mat_sips = mat_dur = 0
    for fi, path in enumerate(RAW_FILES):
        res = detect_recording(load_recording(path, cfg).capacitance, cfg)
        for ch in range(mat.n_channels):
            py_sips += len(res.sips[ch])
            py_dur += int(res.sips[ch].durations.sum())
            mat_sips += int(mat.ons[fi][ch].size)
            mat_dur += int(np.asarray(mat.durations[fi][ch]).sum())

    assert abs(py_sips - mat_sips) / mat_sips < 0.02  # totals within ~2%
    assert abs(py_dur - mat_dur) / mat_dur < 0.05  # total duration within ~5%


# --------------------------------------------------------------------------- #
# two-choice preference + cumulative time course
# --------------------------------------------------------------------------- #
def test_preference_index_pairs_arenas() -> None:
    from flypad.stats import preference_index

    per_fly = pd.DataFrame(
        {
            "file_index": [0, 0, 0, 0],
            "channel": [0, 1, 2, 3],
            "condition_label": ["a", "a", "b", "b"],
            "n_sips": [10, 5, 0, 0],
        }
    )
    pi = preference_index(per_fly).sort_values("arena").reset_index(drop=True)
    assert len(pi) == 2
    assert pi.loc[0, "preference"] == pytest.approx((10 - 5) / 15)  # arena 0
    assert np.isnan(pi.loc[1, "preference"])  # arena 1 ate nothing
    assert list(pi["condition_label"]) == ["a", "b"]


def test_cumulative_timecourse_by_condition() -> None:
    from flypad.stats import cumulative_timecourse_by_condition

    events = pd.DataFrame(
        {
            "file_index": [0, 0, 0, 0, 0],
            "channel": [0, 0, 2, 2, 2],
            "condition_label": ["a", "a", "b", "b", "b"],
            "onset": [10, 90, 10, 50, 90],
        }
    )
    out = cumulative_timecourse_by_condition(events, n_samples=100, n_bins=10, sampling_rate_hz=100)
    assert set(out) == {"a", "b"}
    time_s, mean_a, _sem = out["a"]
    assert len(time_s) == 10
    assert time_s[-1] == pytest.approx(1.0)  # 100 samples @ 100 Hz = 1 s
    assert mean_a[-1] == pytest.approx(2.0)  # one fly, 2 sips total
    assert out["b"][1][-1] == pytest.approx(3.0)
    assert np.all(np.diff(mean_a) >= 0)  # cumulative is non-decreasing
