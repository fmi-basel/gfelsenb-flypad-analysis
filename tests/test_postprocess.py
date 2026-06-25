"""M4: post-processing — feeding bursts, QC removal, transitions, metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flypad.config.models import (
    FeedingBursts,
    NonEaters,
    QualityControl,
)
from flypad.detect.results import ChannelSips
from flypad.postprocess import (
    ArenaTransitions,
    assess_quality,
    build_channel_condition_map,
    channel_condition_map_for_dir,
    channel_transitions,
    classify_in_burst,
    detect_feeding_bursts,
    feeding_ifi_threshold,
    flag_non_eaters,
    group_feeding_bursts,
    mode_int,
    parse_exp_file,
    parse_log_file,
    saturation_fraction,
    zero_fraction,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample" / "20240215"
SAMPLE_CSV = SAMPLE_DIR / "channel_condition_map.csv"
SAMPLE_MAT = SAMPLE_DIR / "20240215.mat"
RAW_FILES = sorted(SAMPLE_DIR.glob("CapacitanceData_*"))
DUR = 425_391  # MATLAB Events.Dur for this fixture (min length across files)


def _sips(onsets: list[int], offsets: list[int]) -> ChannelSips:
    return ChannelSips(
        onsets=np.asarray(onsets, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
    )


# --------------------------------------------------------------------------- #
# metadata parser (primary regression target: reproduce the 192-row CSV)
# --------------------------------------------------------------------------- #
def test_parse_exp_file_sections() -> None:
    exp = parse_exp_file(SAMPLE_DIR / "exp_0.txt")
    assert exp.conditions[1] == "fully fed"
    assert exp.conditions[5] == "44h wet starved + 30 min refed"
    assert exp.substrate_labels["left"].startswith("100 mM sucrose")
    assert len(exp.board) == 12
    # quirky "8 : 44h refed, F" line parses to position 8
    by_pos = exp.board_by_position()
    assert by_pos[8].condition_short == "44h refed"
    assert by_pos[8].sex == "F"
    # condition number assigned by first-appearance order of the short label
    order = exp.condition_order()
    assert order == {"fed": 1, "24h": 2, "44h": 3, "24h refed": 4, "44h refed": 5}


def test_parse_log_file_substrate_labels() -> None:
    log = parse_log_file(SAMPLE_DIR / "LOG.txt")
    assert log.substrate_labels == {1: "100 mM sucrose", 2: "100 mM sucrose"}
    assert log.condition_labels == {1: "fly"}


def test_channel_condition_map_reproduces_fixture() -> None:
    """The metadata parser must reproduce channel_condition_map.csv byte-for-byte."""
    expected = pd.read_csv(SAMPLE_CSV)
    order = expected.drop_duplicates("file_index").sort_values("file_index")
    specs = [
        (name, SAMPLE_DIR / exp) for name, exp in zip(order.file_name, order.exp_file, strict=True)
    ]
    got = build_channel_condition_map(specs, log_file=SAMPLE_DIR / "LOG.txt", n_channels=96)
    pd.testing.assert_frame_equal(got, expected)


def test_channel_map_files_differ_only_at_positions_11_12() -> None:
    expected = pd.read_csv(SAMPLE_CSV)
    f0 = expected[expected.file_index == 0].reset_index(drop=True)
    f1 = expected[expected.file_index == 1].reset_index(drop=True)
    differ = f0["condition"].to_numpy() != f1["condition"].to_numpy()
    # channels 80..95 = board positions 11,12
    assert set(np.flatnonzero(differ)) == set(range(80, 96))
    assert set(f0.loc[differ, "condition"]) == {5}
    assert set(f1.loc[differ, "condition"]) == {4}


def test_substrate_side_even_odd() -> None:
    expected = pd.read_csv(SAMPLE_CSV)
    even = expected[expected.channel % 2 == 0]
    odd = expected[expected.channel % 2 == 1]
    assert set(even.substrate) == {1} and set(even.substrate_side) == {"left"}
    assert set(odd.substrate) == {2} and set(odd.substrate_side) == {"right"}


@pytest.mark.skipif(not RAW_FILES, reason="raw sample binaries not present (git-ignored)")
def test_channel_condition_map_for_dir_matches_fixture() -> None:
    expected = pd.read_csv(SAMPLE_CSV)
    got = channel_condition_map_for_dir(SAMPLE_DIR)
    pd.testing.assert_frame_equal(got, expected)


# --------------------------------------------------------------------------- #
# feeding bursts (GET_FEEDING_BURSTS)
# --------------------------------------------------------------------------- #
def test_mode_int_smallest_on_ties() -> None:
    assert mode_int([5, 5, 3, 3, 9]) == 3  # tie -> smallest
    assert mode_int([7, 7, 7, 2]) == 7
    assert mode_int([]) == 0


def test_feeding_ifi_threshold_is_two_times_mode() -> None:
    assert feeding_ifi_threshold([10, 10, 10, 40]) == 20.0
    with pytest.raises(ValueError, match="ifi_criterion"):
        feeding_ifi_threshold([1, 2], criterion="bogus")


def test_group_feeding_bursts_splits_on_large_gap() -> None:
    # IFIs (onset[i+1]-offset[i]): all 10 except a 200-sample gap after the 4th sip
    onsets = [0, 30, 60, 90, 320, 350, 380]
    offsets = [20, 50, 80, 110, 340, 370, 400]
    sips = _sips(onsets, offsets)
    bursts = group_feeding_bursts(sips, ifi_threshold=20.0, min_sips=3)
    assert len(bursts) == 2
    assert bursts.n_sips.tolist() == [4, 3]
    assert bursts.onsets.tolist() == [0, 320]
    assert bursts.offsets.tolist() == [110, 400]
    assert bursts.sip_start.tolist() == [0, 4]
    assert bursts.sip_end.tolist() == [3, 6]


def test_group_feeding_bursts_drops_short_runs() -> None:
    sips = _sips([0, 30, 500], [20, 50, 520])  # two close, one isolated
    bursts = group_feeding_bursts(sips, ifi_threshold=20.0, min_sips=3)
    assert len(bursts) == 0


def test_group_feeding_bursts_too_few_sips() -> None:
    sips = _sips([0, 10], [5, 15])
    assert len(group_feeding_bursts(sips, 20.0, min_sips=3)) == 0


def test_detect_feeding_bursts_pooled_vs_per_channel() -> None:
    cfg = FeedingBursts(min_sips=3)
    quiet = _sips([0, 10, 20, 30, 40], [5, 15, 25, 35, 45])  # IFIs of 5
    bursty = _sips([0, 8, 16, 24], [4, 12, 20, 28])  # IFIs of 4
    thr, bursts = detect_feeding_bursts([quiet, bursty], cfg, scope="pooled")
    assert thr == feeding_ifi_threshold(np.concatenate([quiet.ifi, bursty.ifi]), cfg.ifi_criterion)
    assert len(bursts) == 2
    assert all(len(b) >= 1 for b in bursts)


# --------------------------------------------------------------------------- #
# quality control: spill / unconnected / non-eater
# --------------------------------------------------------------------------- #
def test_saturation_and_zero_fraction() -> None:
    raw = np.array([[4095, 0, 7], [4095, 0, 8], [3, 0, 4095], [4095, 5, 9]])
    np.testing.assert_allclose(saturation_fraction(raw), [0.75, 0.0, 0.25])
    np.testing.assert_allclose(zero_fraction(raw), [0.0, 0.75, 0.0])


def test_flag_non_eaters_global_arena() -> None:
    # arenas (0,1),(2,3),(4,5); threshold 2 on combined arena count
    counts = [0, 0, 5, 0, 1, 0]
    mask = flag_non_eaters(counts, remove_global=True, remove_substrate=False, threshold=2)
    # arena0 total 0 -> remove both; arena1 total 5 -> keep; arena2 total 1 -> remove both
    assert mask.tolist() == [True, True, False, False, True, True]


def test_flag_non_eaters_substrate_only() -> None:
    counts = [0, 5, 1, 3]
    mask = flag_non_eaters(counts, remove_global=False, remove_substrate=True, threshold=2)
    assert mask.tolist() == [True, False, True, False]


def test_flag_non_eaters_odd_trailing_channel() -> None:
    counts = [5, 5, 0]  # lone channel 2 judged alone
    mask = flag_non_eaters(counts, remove_global=True, remove_substrate=False, threshold=2)
    assert mask.tolist() == [False, False, True]


def test_flag_non_eaters_threshold_is_inclusive() -> None:
    # MATLAB drops flies with "<=2" bouts; count == threshold must be removed.
    counts = [2, 3, 1, 4]  # substrate rule: 2<=2 remove, 3 keep, 1 remove, 4 keep
    mask = flag_non_eaters(counts, remove_global=False, remove_substrate=True, threshold=2)
    assert mask.tolist() == [True, False, True, False]


def test_assess_quality_no_removal_on_clean_signal() -> None:
    rng = np.random.default_rng(0)
    raw = rng.integers(100, 3000, size=(1000, 8))  # no saturation, no zeros
    counts = np.full(8, 50, dtype=np.int64)  # everyone ate plenty
    result = assess_quality(raw, counts, QualityControl(), NonEaters())
    assert not result.remove_mask.any()
    assert result.kept_channels.tolist() == list(range(8))


# --------------------------------------------------------------------------- #
# transitions
# --------------------------------------------------------------------------- #
def test_channel_transitions_basic() -> None:
    left = _sips([0, 100], [10, 110])  # left at t=0, t=100
    right = _sips([50, 150], [60, 160])  # right at t=50, t=150
    tr = channel_transitions(left, right)
    # timeline: L(0) R(50) L(100) R(150) -> 3 transitions
    assert tr.n_transitions == 3
    assert tr.times.tolist() == [50, 100, 150]
    assert tr.from_side.tolist() == [0, 1, 0]
    assert tr.to_side.tolist() == [1, 0, 1]
    assert tr.ibi.tolist() == [40, 40, 40]  # 50-10, 100-60, 150-110
    assert tr.latency_left == 0 and tr.latency_right == 50


def test_channel_transitions_no_switch() -> None:
    left = _sips([0, 10, 20], [5, 15, 25])
    right = _sips([], [])
    tr = channel_transitions(left, right)
    assert isinstance(tr, ArenaTransitions)
    assert tr.n_transitions == 0
    assert tr.latency_left == 0 and tr.latency_right == -1


def test_classify_in_burst() -> None:
    sips = _sips([0, 30, 60, 90, 320, 350, 380], [20, 50, 80, 110, 340, 370, 400])
    bursts = group_feeding_bursts(sips, ifi_threshold=20.0, min_sips=3)
    mask = classify_in_burst(len(sips), bursts)
    assert mask.tolist() == [True, True, True, True, True, True, True]
    # if we keep only the first burst, the last 3 sips become isolated
    sips2 = _sips([0, 30, 60, 500, 700], [20, 50, 80, 520, 720])
    bursts2 = group_feeding_bursts(sips2, ifi_threshold=20.0, min_sips=3)
    mask2 = classify_in_burst(len(sips2), bursts2)
    assert mask2.tolist() == [True, True, True, False, False]


# --------------------------------------------------------------------------- #
# QC regression against the MATLAB ground truth (raw binaries required)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (RAW_FILES and SAMPLE_MAT.exists()),
    reason="raw sample binaries / .mat not present",
)
def test_spill_unconnected_match_matlab_ground_truth() -> None:
    """SpillQuality / Unconnected fractions must reproduce Events.* from the .mat."""
    import h5py

    from flypad.io.raw import load_raw

    with h5py.File(SAMPLE_MAT, "r") as f:
        ev = f["Events"]
        spill_refs = np.asarray(ev["SpillQuality"])
        unconn_refs = np.asarray(ev["Unconnected"])

        def deref(ref: object) -> float:
            obj = f[ref]
            if obj.attrs.get("MATLAB_empty", 0):
                return float("nan")
            arr = np.asarray(obj).ravel()
            return float(arr[0]) if arr.size else float("nan")

        for file_index, raw_path in enumerate(RAW_FILES):
            raw = np.asarray(load_raw(raw_path, n_channels=96, duration=DUR).values)
            spill = saturation_fraction(raw, 4095)
            zeros = zero_fraction(raw)
            mat_spill = np.array([deref(spill_refs[c, file_index]) for c in range(96)])
            mat_unconn = np.array([deref(unconn_refs[c, file_index]) for c in range(96)])
            assert np.nanmax(np.abs(spill - mat_spill)) < 1e-3
            assert np.nanmax(np.abs(zeros - mat_unconn)) < 1e-3
