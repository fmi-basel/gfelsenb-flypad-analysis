"""Sip detection: greedy (MATLAB-compat) and adjacent (corrected) pairing (design §5.7).

Both consume the length-preserving derivative plus per-channel thresholds. The
greedy strategy reproduces ``ProcessDataDamPAD``'s forward level-crossing while-loop;
the adjacent strategy is the corrected peak-to-peak pairing with a *real* amplitude
gate (the old Python port's ``ratios < 0.5`` was inert).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

from flypad.config.models import Pairing, SipDetection
from flypad.detect.results import ChannelSips

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _per_channel(t: FloatArray, ch: int) -> FloatArray | float:
    """Extract a channel's threshold: a scalar (constant) or a length-T array."""
    if t.shape[0] == 1:
        return float(t[0, ch if t.shape[1] > 1 else 0])
    return np.asarray(t[:, ch], dtype=np.float64)


def _peaks(x: FloatArray, distance: int) -> IntArray:
    locs, _ = find_peaks(x, distance=distance)
    return np.asarray(locs, dtype=np.int64)


def _greedy_pair(
    trace: FloatArray,
    pos_events: IntArray,
    window: int,
    min_window: int,
    equality_factor: float,
) -> tuple[IntArray, IntArray]:
    """MATLAB while-loop: from each attachment, take the first detachment that drops
    to <= -equality_factor x the attachment peak within ``window`` samples."""
    n = trace.shape[0]
    onsets: list[int] = []
    offsets: list[int] = []
    if pos_events.size < 2:
        return np.asarray(onsets, dtype=np.int64), np.asarray(offsets, dtype=np.int64)
    cur = int(pos_events[0])
    while cur < n:
        target = trace[cur] * -equality_factor
        seg = trace[cur : min(cur + window + 1, n)]
        below = np.flatnonzero(seg <= target)
        accepted = False
        if below.size:
            f0 = int(below[0])
            if f0 + 1 >= min_window:  # MATLAB find(...) is 1-based: first >= MinWindow
                off = cur + f0
                onsets.append(cur)
                offsets.append(off)
                j = int(np.searchsorted(pos_events, off, side="right"))
                cur = int(pos_events[j]) if j < pos_events.size else n
                accepted = True
        if not accepted:
            j = int(np.searchsorted(pos_events, cur, side="right"))
            cur = int(pos_events[j]) if j < pos_events.size else n
    return np.asarray(onsets, dtype=np.int64), np.asarray(offsets, dtype=np.int64)


def _greedy_channel(
    fderiv: FloatArray,
    pos_thr: FloatArray | float,
    neg_thr: FloatArray | float,
    cfg: SipDetection,
) -> tuple[IntArray, IntArray]:
    n = fderiv.shape[0]
    pos_mask = fderiv > pos_thr
    neg_mask = fderiv < neg_thr
    chosen = np.zeros(n, dtype=bool)
    chosen[_peaks(fderiv, cfg.peak_min_distance)] = True
    chosen[_peaks(-fderiv, cfg.peak_min_distance)] = True
    ppos = pos_mask & chosen
    nneg = neg_mask & chosen
    trace = np.zeros(n, dtype=np.float64)
    trace[ppos] = fderiv[ppos]
    trace[nneg] = fderiv[nneg]
    pos_events = np.flatnonzero(ppos).astype(np.int64)
    return _greedy_pair(
        trace, pos_events, cfg.max_duration_samples, cfg.min_duration_samples, cfg.equality_factor
    )


def _adjacent_channel(
    fderiv: FloatArray,
    pos_thr: FloatArray | float,
    neg_thr: FloatArray | float,
    cfg: SipDetection,
) -> tuple[IntArray, IntArray]:
    pos_locs = np.asarray(
        find_peaks(np.clip(fderiv, 0, None), distance=cfg.peak_min_distance, height=pos_thr)[0],
        dtype=np.int64,
    )
    neg_height = -np.asarray(neg_thr)
    neg_locs = np.asarray(
        find_peaks(np.clip(-fderiv, 0, None), distance=cfg.peak_min_distance, height=neg_height)[0],
        dtype=np.int64,
    )
    if pos_locs.size == 0 or neg_locs.size == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    events = np.concatenate([pos_locs, neg_locs])
    labels = np.concatenate([np.ones(pos_locs.size), np.zeros(neg_locs.size)])
    order = np.argsort(events, kind="stable")
    ev = events[order]
    lb = labels[order]
    trans = np.diff(lb) == -1  # an attachment (1) immediately followed by a detachment (0)
    attach = ev[:-1][trans]
    detach = ev[1:][trans]
    dur = detach - attach
    amp_ratio = -fderiv[detach] / fderiv[attach]  # both magnitudes positive
    keep = (
        (dur > cfg.min_duration_samples)
        & (dur < cfg.max_duration_samples)
        & (amp_ratio >= cfg.equality_factor)
    )
    return attach[keep].astype(np.int64), detach[keep].astype(np.int64)


def detect_sips(
    derivative: FloatArray,
    t_pos: FloatArray,
    t_neg: FloatArray,
    cfg: SipDetection,
) -> list[ChannelSips]:
    """Detect sips per channel using the configured pairing strategy."""
    n_ch = derivative.shape[1]
    out: list[ChannelSips] = []
    for ch in range(n_ch):
        fderiv = np.asarray(derivative[:, ch], dtype=np.float64)
        pos_thr = _per_channel(t_pos, ch)
        neg_thr = _per_channel(t_neg, ch)
        if cfg.pairing is Pairing.greedy:
            on, off = _greedy_channel(fderiv, pos_thr, neg_thr, cfg)
        else:
            on, off = _adjacent_channel(fderiv, pos_thr, neg_thr, cfg)
        out.append(ChannelSips(onsets=on, offsets=off))
    return out
