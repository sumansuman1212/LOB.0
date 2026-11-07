"""
feature_pipeline.py — Python Reference Implementation of C++ FeaturePipeline

Mirrors mm::FeaturePipeline (src/cpp/feature_pipeline.cpp) exactly for
numerical parity testing and cross-stack verification.

Requirements: 2.1-2.13, 10.4, 11.4
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np


SIGMA_CLAMP: float = 5.0
EPSILON: float = 1e-10
MAX_LOB_LEVELS: int = 10
OBS_DIM: int = 8


@dataclass
class PriceLevel:
    price: int = 0
    total_qty: int = 0
    order_count: int = 0


@dataclass
class BookSnapshot:
    timestamp_ns: int = 0
    best_bid: int = 0
    best_ask: int = 0
    bids: list[PriceLevel] = field(default_factory=lambda: [PriceLevel() for _ in range(MAX_LOB_LEVELS)])
    asks: list[PriceLevel] = field(default_factory=lambda: [PriceLevel() for _ in range(MAX_LOB_LEVELS)])


class FeaturePipeline:
    """Python reference implementation of C++ mm::FeaturePipeline."""

    def __init__(
        self,
        max_position: int = 500,
        vol_window: int = 60,
        ofi_levels: int = 5,
        session_duration_s: int = 23400,  # 6.5 hours
    ) -> None:
        self.max_position = max(1, max_position)
        self.vol_window = max(2, vol_window)
        self.ofi_levels = min(max(1, ofi_levels), MAX_LOB_LEVELS)
        self.session_duration_ns = session_duration_s * 1_000_000_000

        self.vol_buf: list[float] = [0.0] * self.vol_window
        self.vol_head: int = 0
        self.vol_count: int = 0
        self.vol_sum: float = 0.0
        self.vol_sum_sq: float = 0.0
        self.vol_mean_baseline: float = 0.0
        self.vol_std_baseline: float = 1.0

        self.last_mid_price: int = 0
        self.session_start_ns: int = 0
        self.last_update_ns: int = -2**63
        self.current_features: np.ndarray = np.zeros(OBS_DIM, dtype=np.float32)

    def reset(self) -> None:
        """Reset pipeline to initial state."""
        self.vol_buf = [0.0] * self.vol_window
        self.vol_head = 0
        self.vol_count = 0
        self.vol_sum = 0.0
        self.vol_sum_sq = 0.0
        self.vol_mean_baseline = 0.0
        self.vol_std_baseline = 1.0
        self.last_mid_price = 0
        self.session_start_ns = 0
        self.last_update_ns = -2**63
        self.current_features = np.zeros(OBS_DIM, dtype=np.float32)

    @staticmethod
    def sanitize(v: float) -> float:
        """NaN/Inf guard and 5-sigma clamp (Req 2.10, 11.4)."""
        if not math.isfinite(v):
            return 0.0
        if v > SIGMA_CLAMP:
            return SIGMA_CLAMP
        if v < -SIGMA_CLAMP:
            return -SIGMA_CLAMP
        return float(v)

    @staticmethod
    def online_variance(total_sum: float, sum_sq: float, n: int) -> float:
        """Sample variance computed from accumulated sum / sum-of-squares."""
        if n < 2:
            return 0.0
        mean = total_sum / float(n)
        var = (sum_sq / float(n)) - (mean * mean)
        return float(var if var > 0.0 else 0.0)

    def push_return(self, r: float) -> None:
        """Push a new log-return into circular buffer."""
        r_d = float(r)
        if self.vol_count == self.vol_window:
            old_r = float(self.vol_buf[self.vol_head])
            self.vol_sum -= old_r
            self.vol_sum_sq -= old_r * old_r

        self.vol_buf[self.vol_head] = float(r)
        self.vol_sum += r_d
        self.vol_sum_sq += r_d * r_d

        self.vol_head = (self.vol_head + 1) % self.vol_window
        if self.vol_count < self.vol_window:
            self.vol_count += 1

        if self.vol_count == self.vol_window:
            var = self.online_variance(self.vol_sum, self.vol_sum_sq, self.vol_count)
            self.vol_std_baseline = math.sqrt(var) if var > EPSILON else 1.0
            self.vol_mean_baseline = self.vol_sum / float(self.vol_count)

    def compute_vol_regime(self) -> float:
        """Compute rolling realized volatility z-score."""
        if self.vol_count < 2:
            return 0.0
        mean_sq = self.vol_sum_sq / float(self.vol_count)
        if mean_sq < 0.0:
            mean_sq = 0.0
        realized_vol = math.sqrt(mean_sq)
        std_b = self.vol_std_baseline if self.vol_std_baseline > EPSILON else 1.0
        z = (realized_vol - self.vol_mean_baseline) / std_b
        return self.sanitize(z)

    def update(self, snap: BookSnapshot, net_position: int) -> np.ndarray:
        """Compute all 8 features and store in current_features."""
        ts = snap.timestamp_ns
        if self.session_start_ns == 0:
            self.session_start_ns = ts
        self.last_update_ns = ts

        # 8. Session progress
        session_progress = 0.0
        if self.session_duration_ns > 0:
            elapsed = ts - self.session_start_ns
            if elapsed < 0:
                elapsed = 0
            progress = float(elapsed) / float(self.session_duration_ns)
            session_progress = min(1.0, progress)

        # 1. OFI L1
        bid_l1 = float(snap.bids[0].total_qty)
        ask_l1 = float(snap.asks[0].total_qty)
        ofi_l1_denom = bid_l1 + ask_l1
        ofi_l1 = (bid_l1 - ask_l1) / ofi_l1_denom if ofi_l1_denom > EPSILON else 0.0

        # 2. OFI L5
        bid_l5_sum = sum(float(snap.bids[i].total_qty) for i in range(self.ofi_levels))
        ask_l5_sum = sum(float(snap.asks[i].total_qty) for i in range(self.ofi_levels))
        ofi_l5_denom = bid_l5_sum + ask_l5_sum
        ofi_l5 = (bid_l5_sum - ask_l5_sum) / ofi_l5_denom if ofi_l5_denom > EPSILON else 0.0

        # 3. Inventory skew & 4. Urgency
        pos_norm = float(net_position) / float(self.max_position)
        inventory_skew = max(-1.0, min(1.0, pos_norm))
        inventory_urgency = max(0.0, min(1.0, abs(pos_norm)))

        # 5. Volatility regime
        mid_price_ticks = (snap.best_bid + snap.best_ask) // 2 if (snap.best_bid > 0 and snap.best_ask > 0) else 0
        if self.last_mid_price > 0 and mid_price_ticks > 0:
            log_ret = math.log(float(mid_price_ticks) / float(self.last_mid_price))
            if math.isfinite(log_ret):
                self.push_return(log_ret)
        if mid_price_ticks > 0:
            self.last_mid_price = mid_price_ticks

        vol_regime = self.compute_vol_regime()

        # 6. Micro-price offset
        micro_price_offset = 0.0
        if ofi_l1_denom > EPSILON and snap.best_bid > 0 and snap.best_ask > 0:
            bid_p = float(snap.best_bid)
            ask_p = float(snap.best_ask)
            micro_price = (ask_p * bid_l1 + bid_p * ask_l1) / ofi_l1_denom
            mid_p = (bid_p + ask_p) * 0.5
            raw_offset = micro_price - mid_p
            half_spread = (ask_p - bid_p) * 0.5
            normalized = (raw_offset / half_spread) if half_spread > EPSILON else raw_offset
            micro_price_offset = normalized

        # 7. Spread in ticks (log1p scaling)
        spread_ticks = 0.0
        if snap.best_ask > 0 and snap.best_bid > 0:
            raw_spread = snap.best_ask - snap.best_bid
            if raw_spread > 0:
                spread_ticks = math.log1p(float(raw_spread) - 1.0)

        # Assemble and clamp
        self.current_features = np.array(
            [
                self.sanitize(ofi_l1),
                self.sanitize(ofi_l5),
                self.sanitize(inventory_skew),
                self.sanitize(inventory_urgency),
                self.sanitize(vol_regime),
                self.sanitize(micro_price_offset),
                self.sanitize(spread_ticks),
                self.sanitize(session_progress),
            ],
            dtype=np.float32,
        )
        return self.current_features

    def observe(self) -> np.ndarray:
        """Return current feature vector."""
        return self.current_features.copy()

