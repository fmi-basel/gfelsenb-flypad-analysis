"""M3: detection — bouts + greedy/adjacent sip pairing on synthetic traces."""

from __future__ import annotations

import numpy as np

from flypad.config import load_config
from flypad.config.models import Pairing, SipThreshold, ThresholdMethod
from flypad.detect import detect_recording, detect_sips, extract_bouts
from flypad.detect._synth import make_trace
from flypad.signal.filters import derivative
from flypad.signal.thresholds import compute_thresholds


def _two_channel_signal(onsets: list[int], n: int = 2000) -> np.ndarray:
    a = make_trace(n, onsets, seed=1)
    b = make_trace(n, [o + 3 for o in onsets], seed=2)
    return np.stack([a, b], axis=1)


def test_extract_bouts_finds_runs() -> None:
    mask = np.zeros((100, 1), dtype=bool)
    mask[20:30, 0] = True
    mask[60:75, 0] = True
    bouts = extract_bouts(mask)
    assert len(bouts[0]) == 2
    np.testing.assert_array_equal(bouts[0].onsets, [20, 60])
    np.testing.assert_array_equal(bouts[0].offsets, [30, 75])


def _detect(onsets: list[int], pairing: Pairing) -> list:
    sig = _two_channel_signal(onsets)
    deriv = derivative(sig)
    thr = SipThreshold(method=ThresholdMethod.fixed_height, fixed_height=50.0)
    t_pos, t_neg = compute_thresholds(sig, deriv, np.zeros_like(sig), thr)
    cfg = load_config(preset="corrected").sip_detection
    cfg = cfg.model_copy(update={"pairing": pairing})
    return detect_sips(deriv, t_pos, t_neg, cfg)


def test_greedy_recovers_injected_sips() -> None:
    onsets = [200, 500, 900, 1400]
    sips = _detect(onsets, Pairing.greedy)
    # channel 0: an onset near each injected attachment (within a few samples)
    got = sips[0].onsets
    assert len(got) == len(onsets)
    for true_on in onsets:
        assert np.min(np.abs(got - true_on)) <= 3


def test_adjacent_recovers_injected_sips() -> None:
    onsets = [200, 500, 900, 1400]
    sips = _detect(onsets, Pairing.adjacent)
    got = sips[0].onsets
    assert len(got) == len(onsets)
    for true_on in onsets:
        assert np.min(np.abs(got - true_on)) <= 3


def test_durations_and_ifi_consistent() -> None:
    sips = _detect([300, 800], Pairing.greedy)[0]
    assert np.all(sips.durations == sips.offsets - sips.onsets)
    if len(sips) > 1:
        assert sips.ifi.size == len(sips) - 1


def test_detect_recording_smoke() -> None:
    import xarray as xr

    sig = _two_channel_signal([300, 700, 1100])
    da = xr.DataArray(sig, dims=("time", "channel"))
    cfg = load_config(preset="matlab_compat")
    cfg = cfg.model_copy(update={"hardware": cfg.hardware.model_copy(update={"n_channels": 2})})
    result = detect_recording(da, cfg)
    assert result.n_channels == 2
    assert result.crop_offset == cfg.preprocessing.baseline_span  # crop mode
