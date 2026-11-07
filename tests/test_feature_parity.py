"""
tests/test_feature_parity.py — Mathematical & Numerical Parity for Feature Pipeline

Validates numerical parity between Python reference FeaturePipeline and
C++ mm::FeaturePipeline specifications across all 8 features:
1. OFI L1
2. OFI L5
3. Inventory skew
4. Inventory urgency
5. Volatility regime
6. Microprice offset
7. Spread ticks
8. Session progress

Requirement: abs(feature_python[i] - feature_cpp[i]) < 1e-5 across multiple market regimes.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from src.python.feature_pipeline import BookSnapshot, FeaturePipeline, PriceLevel


def make_snapshot(
    best_bid: int,
    best_ask: int,
    bid_qtys: list[int],
    ask_qtys: list[int],
    timestamp_ns: int = 1_000_000_000,
) -> BookSnapshot:
    """Construct a BookSnapshot from price levels and depth volumes."""
    snap = BookSnapshot(timestamp_ns=timestamp_ns, best_bid=best_bid, best_ask=best_ask)
    for i in range(10):
        b_qty = bid_qtys[i] if i < len(bid_qtys) else 0
        a_qty = ask_qtys[i] if i < len(ask_qtys) else 0
        snap.bids[i] = PriceLevel(price=best_bid - i * 5, total_qty=b_qty, order_count=max(1, b_qty // 10))
        snap.asks[i] = PriceLevel(price=best_ask + i * 5, total_qty=a_qty, order_count=max(1, a_qty // 10))
    return snap


def cpp_reference_compute(snap: BookSnapshot, net_position: int, max_position: int, session_start_ns: int = 0, session_duration_s: int = 23400) -> np.ndarray:
    """Direct line-by-line transcription of mm::FeaturePipeline::update (C++17)."""
    # 8. Session progress
    start_ns = snap.timestamp_ns if session_start_ns == 0 else session_start_ns
    elapsed = max(0, snap.timestamp_ns - start_ns)
    progress = min(1.0, float(elapsed) / float(session_duration_s * 1_000_000_000))

    # 1. OFI L1
    b_l1 = float(snap.bids[0].total_qty)
    a_l1 = float(snap.asks[0].total_qty)
    denom_l1 = b_l1 + a_l1
    ofi_l1 = (b_l1 - a_l1) / denom_l1 if denom_l1 > 1e-10 else 0.0

    # 2. OFI L5
    b_l5 = sum(float(snap.bids[i].total_qty) for i in range(5))
    a_l5 = sum(float(snap.asks[i].total_qty) for i in range(5))
    denom_l5 = b_l5 + a_l5
    ofi_l5 = (b_l5 - a_l5) / denom_l5 if denom_l5 > 1e-10 else 0.0

    # 3 & 4. Inventory
    pos_norm = float(net_position) / float(max_position)
    skew = max(-1.0, min(1.0, pos_norm))
    urgency = max(0.0, min(1.0, abs(pos_norm)))

    # 5. Vol regime (default for single snapshot without history = 0.0)
    vol_regime = 0.0

    # 6. Microprice offset
    micro_offset = 0.0
    if denom_l1 > 1e-10 and snap.best_bid > 0 and snap.best_ask > 0:
        bid_p = float(snap.best_bid)
        ask_p = float(snap.best_ask)
        micro_price = (ask_p * b_l1 + bid_p * a_l1) / denom_l1
        mid_p = (bid_p + ask_p) * 0.5
        raw_offset = micro_price - mid_p
        half_spread = (ask_p - bid_p) * 0.5
        micro_offset = raw_offset / half_spread if half_spread > 1e-10 else raw_offset

    # 7. Spread in ticks (log1p)
    spread_ticks = 0.0
    if snap.best_ask > 0 and snap.best_bid > 0:
        raw_spread = snap.best_ask - snap.best_bid
        if raw_spread > 0:
            spread_ticks = math.log1p(float(raw_spread) - 1.0)

    # Sanitize helper
    def san(v: float) -> float:
        if not math.isfinite(v):
            return 0.0
        return float(np.clip(v, -5.0, 5.0))

    return np.array(
        [
            san(ofi_l1),
            san(ofi_l5),
            san(skew),
            san(urgency),
            san(vol_regime),
            san(micro_offset),
            san(spread_ticks),
            san(progress),
        ],
        dtype=np.float32,
    )


def test_regime_1_balanced_liquid_book():
    """Regime 1: Perfectly balanced liquid book (neutral OFI, zero skew)."""
    snap = make_snapshot(
        best_bid=20000,
        best_ask=20005,
        bid_qtys=[100, 80, 60, 40, 20],
        ask_qtys=[100, 80, 60, 40, 20],
        timestamp_ns=1_000_000_000,
    )
    pipeline = FeaturePipeline(max_position=500)
    features = pipeline.update(snap, net_position=0)
    expected = cpp_reference_compute(snap, net_position=0, max_position=500)

    assert np.all(np.isfinite(features))
    assert np.allclose(features, expected, atol=1e-5)
    assert features[0] == 0.0  # OFI L1 neutral
    assert features[1] == 0.0  # OFI L5 neutral
    assert features[2] == 0.0  # skew neutral
    assert features[3] == 0.0  # urgency zero
    assert features[5] == 0.0  # microprice equals mid


def test_regime_2_strong_buy_pressure():
    """Regime 2: Heavy buy pressure (bids 5x asks, net short position)."""
    snap = make_snapshot(
        best_bid=20000,
        best_ask=20005,
        bid_qtys=[500, 400, 300, 200, 100],
        ask_qtys=[100, 80, 60, 40, 20],
        timestamp_ns=2_000_000_000,
    )
    pipeline = FeaturePipeline(max_position=500)
    features = pipeline.update(snap, net_position=-250)
    expected = cpp_reference_compute(snap, net_position=-250, max_position=500)

    assert np.allclose(features, expected, atol=1e-5)
    assert features[0] > 0.6  # OFI L1 strongly positive
    assert features[1] > 0.6  # OFI L5 strongly positive
    assert features[2] == -0.5  # Skew is -0.5
    assert features[3] == 0.5   # Urgency is 0.5
    assert features[5] > 0.6   # Microprice shifted toward ask


def test_regime_3_strong_sell_pressure():
    """Regime 3: Heavy sell pressure (asks 4x bids, net long position)."""
    snap = make_snapshot(
        best_bid=19995,
        best_ask=20000,
        bid_qtys=[50, 40, 30, 20, 10],
        ask_qtys=[200, 160, 120, 80, 40],
        timestamp_ns=3_000_000_000,
    )
    pipeline = FeaturePipeline(max_position=500)
    features = pipeline.update(snap, net_position=350)
    expected = cpp_reference_compute(snap, net_position=350, max_position=500)

    assert np.allclose(features, expected, atol=1e-5)
    assert features[0] < -0.5  # OFI L1 negative
    assert features[1] < -0.5  # OFI L5 negative
    assert np.isclose(features[2], 0.7, atol=1e-5)   # Skew is +0.7
    assert np.isclose(features[3], 0.7, atol=1e-5)   # Urgency is 0.7
    assert features[5] < -0.5  # Microprice shifted toward bid


def test_regime_4_wide_illiquid_spread():
    """Regime 4: Wide spread (spread = 25 ticks)."""
    snap = make_snapshot(
        best_bid=19975,
        best_ask=20000,  # 25 ticks wide
        bid_qtys=[10, 10, 10, 10, 10],
        ask_qtys=[10, 10, 10, 10, 10],
        timestamp_ns=4_000_000_000,
    )
    pipeline = FeaturePipeline(max_position=500)
    features = pipeline.update(snap, net_position=0)
    expected = cpp_reference_compute(snap, net_position=0, max_position=500)

    assert np.allclose(features, expected, atol=1e-5)
    # log1p(25 - 1) = log(25) = ~3.2188758
    assert math.isclose(features[6], math.log(25.0), abs_tol=1e-5)


def test_regime_5_empty_book_division_by_zero():
    """Regime 5: Boundary conditions with zero liquidity."""
    snap = BookSnapshot(timestamp_ns=5_000_000_000, best_bid=0, best_ask=0)
    pipeline = FeaturePipeline(max_position=500)
    features = pipeline.update(snap, net_position=0)

    assert np.all(np.isfinite(features))
    assert features[0] == 0.0
    assert features[1] == 0.0
    assert features[5] == 0.0
    assert features[6] == 0.0


def test_regime_6_inventory_limit_clamping():
    """Regime 6: Extreme inventory beyond max_position (should clamp to [-1, 1] and [0, 1])."""
    snap = make_snapshot(
        best_bid=20000, best_ask=20005, bid_qtys=[100]*5, ask_qtys=[100]*5
    )
    pipeline = FeaturePipeline(max_position=500)

    # Positive breach (+800 / 500 = +1.6 -> clamp to 1.0)
    f_pos = pipeline.update(snap, net_position=800)
    assert f_pos[2] == 1.0
    assert f_pos[3] == 1.0

    # Negative breach (-750 / 500 = -1.5 -> clamp to -1.0)
    f_neg = pipeline.update(snap, net_position=-750)
    assert f_neg[2] == -1.0
    assert f_neg[3] == 1.0


def test_regime_7_rolling_volatility_sequence():
    """Regime 7: Sequential price updates verifying rolling volatility RMS accumulator."""
    pipeline = FeaturePipeline(max_position=500, vol_window=10)
    prices = [20000, 20010, 19995, 20025, 20015, 20030, 20020, 20040, 20035, 20050, 20045, 20060]

    for i, p in enumerate(prices):
        snap = make_snapshot(best_bid=p - 5, best_ask=p + 5, bid_qtys=[100]*5, ask_qtys=[100]*5, timestamp_ns=1_000_000_000 + i * 1_000_000_000)
        feats = pipeline.update(snap, net_position=0)
        assert np.all(np.isfinite(feats))

    # After 12 steps with window 10, vol_count == 10
    assert pipeline.vol_count == 10
    assert pipeline.vol_sum_sq > 0.0
    assert math.isfinite(feats[4])
