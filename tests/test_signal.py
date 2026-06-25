"""M2: signal processing — RMS, median/baseline/derivative, threshold strategies."""

from __future__ import annotations

import numpy as np
import pytest

from flypad.config.models import SipThreshold, ThresholdMethod
from flypad.signal import (
    compute_thresholds,
    derivative,
    detrend,
    fastrms,
    median_filter_1d,
)
from flypad.signal.thresholds import adaptive_mad_thresholds

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- fastrms
def test_fastrms_matches_reference_conv() -> None:
    """fastrms must equal sqrt(conv(x**2, ones, 'same') / window) in the interior."""
    x = RNG.standard_normal(500)
    window = 50
    ref = np.sqrt(np.convolve(x**2, np.ones(window), "same") / window)
    got = fastrms(x, window)
    # interior (away from edges) must match the MATLAB-style reference
    np.testing.assert_allclose(got[window:-window], ref[window:-window], rtol=1e-9, atol=1e-9)


def test_fastrms_vectorised_equals_per_channel() -> None:
    x = RNG.standard_normal((300, 8))
    got = fastrms(x, 20, axis=0)
    for c in range(x.shape[1]):
        np.testing.assert_allclose(got[:, c], fastrms(x[:, c], 20))


def test_fastrms_constant_signal() -> None:
    x = np.full(200, 3.0)
    rms = fastrms(x, 10)
    np.testing.assert_allclose(rms[50:-50], 3.0, rtol=1e-9)


def test_fastrms_amplitude_correction() -> None:
    x = RNG.standard_normal(200)
    np.testing.assert_allclose(fastrms(x, 10, ampl=True), fastrms(x, 10) * np.sqrt(2.0))


def test_fastrms_nonnegative() -> None:
    assert np.all(fastrms(RNG.standard_normal((100, 4)), 7) >= 0.0)


# ---------------------------------------------------------------- median filter
def test_median_filter_odd_matches_scipy() -> None:
    from scipy.ndimage import median_filter as ndi

    x = RNG.standard_normal((200, 3))
    np.testing.assert_array_equal(median_filter_1d(x, 7), ndi(x, size=(7, 1), mode="nearest"))


def test_median_filter_even_averages_two_middle() -> None:
    # window of 4 over a clean ramp: median = mean of the two central samples.
    x = np.arange(10, dtype=float)
    out = median_filter_1d(x, 4)
    # interior point k: window {k-2,k-1,k,k+1} -> mean of {k-1,k} = k-0.5
    assert out[5] == pytest.approx(4.5)


def test_median_filter_removes_single_spike() -> None:
    x = np.zeros(21)
    x[10] = 1000.0
    assert median_filter_1d(x, 5)[10] == pytest.approx(0.0)


# ---------------------------------------------------------------- detrend
def test_detrend_crop_shortens_by_2_span() -> None:
    x = RNG.standard_normal((400, 5))
    from flypad.config.models import EdgeHandling

    delta, baseline = detrend(x, kernel=6, span=50, edge_handling=EdgeHandling.crop)
    assert delta.shape == (400 - 2 * 50, 5)
    assert baseline.shape == delta.shape


def test_detrend_removes_slow_baseline() -> None:
    from flypad.config.models import EdgeHandling

    t = np.arange(600)
    baseline = 0.01 * t  # slow linear drift
    sip = np.zeros(600)
    sip[300:305] = 80.0  # a fast transient
    x = (baseline + sip)[:, None]
    delta, _ = detrend(x, kernel=6, span=50, edge_handling=EdgeHandling.reflect)
    # the transient survives; the slow drift is largely removed in quiet regions
    assert delta[300:305, 0].max() > 50.0
    assert abs(delta[100, 0]) < 5.0


# ---------------------------------------------------------------- derivative
def test_derivative_length_preserved_and_values() -> None:
    x = np.array([[0.0], [1.0], [3.0], [6.0]])
    d = derivative(x)
    assert d.shape == x.shape
    np.testing.assert_array_equal(d[:, 0], [0.0, 1.0, 2.0, 3.0])


# ---------------------------------------------------------------- thresholds
def test_fixed_height_thresholds() -> None:
    deriv = RNG.standard_normal((100, 4))
    t_pos, t_neg = compute_thresholds(
        deriv,
        deriv,
        np.zeros((100, 4)),
        SipThreshold(method=ThresholdMethod.fixed_height, fixed_height=50.0),
    )
    assert t_pos.item() == 50.0 and t_neg.item() == -50.0


def test_ibis_noise_thresholds_per_channel_constant() -> None:
    delta = RNG.standard_normal((200, 3))
    mask = np.zeros((200, 3))  # all "quiet"
    t_pos, t_neg = compute_thresholds(
        delta,
        derivative(delta),
        mask,
        SipThreshold(method=ThresholdMethod.ibis_noise),
    )
    assert t_pos.shape == (1, 3)
    assert np.all(t_pos > 0) and np.all(t_neg < 0)


def test_adaptive_mad_floor_and_sign() -> None:
    deriv = RNG.standard_normal((400, 2)) * 2.0
    t_pos, t_neg = adaptive_mad_thresholds(deriv, window=50, mad_scale=5.926, offset=50.0)
    assert t_pos.shape == deriv.shape
    assert np.all(t_pos >= 50.0)  # floored
    assert np.all(t_neg <= -50.0)
