"""
mm_env.py — Gymnasium-Compatible Limit Order Book Environment for PPO Training

Implements ``LobEnv``, an RL environment for market making on limit order books.
Conditions quote generation on 8-dimensional market features (OFI, inventory skew,
volatility regime, micro-price offset, spread, session progress) and rewards the
agent via ``RewardShaper`` (mark-to-market PnL + inventory penalty + spread capture).

Requirements: 6.1, 6.2, 6.3, 6.4, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 13.6
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .reward_shaper import RewardShaper, RewardSignal

logger = logging.getLogger(__name__)

OBS_DIM: int = 8
ACTION_DIM: int = 3


class LobEnv(gym.Env):
    """Gymnasium environment for RL-driven adaptive market making.

    Observation Space (8 dimensions, Box[-5.0, 5.0]):
      0: ofi_l1             — Order-flow imbalance at Level 1 in [-1, 1]
      1: ofi_l5             — Order-flow imbalance over top 5 levels in [-1, 1]
      2: inventory_skew     — Signed net position / max_position in [-1, 1]
      3: inventory_urgency  — |net_position| / max_position in [0, 1]
      4: vol_regime         — z-scored realized volatility (clamped [-5, 5])
      5: micro_price_offset — Micro-price relative to mid (in ticks)
      6: spread_ticks       — Current bid-ask spread in ticks
      7: session_progress   — Fraction of trading session elapsed in [0, 1]

    Action Space (3 dimensions):
      0: delta_bid — Bid offset from mid in ticks (>= 0)
      1: delta_ask — Ask offset from mid in ticks (>= 0)
      2: size_frac — Quote size as fraction of max_order_size in (0, 1]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path: Optional[str] = None,
        max_position: int = 500,
        max_order_size: int = 50,
        tick_size: int = 5,
        vol_window: int = 60,
        session_steps: int = 2000,
        inventory_penalty: float = 0.1,
        initial_price: int = 20000,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.max_position = max_position
        self.max_order_size = max_order_size
        self.tick_size = tick_size
        self.vol_window = vol_window
        self.session_steps = session_steps
        self.inventory_penalty = inventory_penalty
        self.initial_price = initial_price
        self.data_path = data_path

        # Spaces (Req 6.3, 6.4)
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.01], dtype=np.float32),
            high=np.array([20.0, 20.0, 1.0], dtype=np.float32),
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

        self.reward_shaper = RewardShaper(
            gamma=inventory_penalty,
            max_position=max_position,
        )

        self.np_random: np.random.Generator = np.random.default_rng(seed)
        self._seed = seed

        # State tracking
        self.step_idx: int = 0
        self.net_position: int = 0
        self.cash: float = 0.0
        self.initial_equity: float = 0.0
        self.equity: float = 0.0
        self.accounting_pnl: float = 0.0
        self.cumulative_reward: float = 0.0
        self.mid_price: float = float(initial_price)
        self.prev_mid_price: float = float(initial_price)
        self.best_bid: int = initial_price - tick_size
        self.best_ask: int = initial_price + tick_size
        self.cumulative_pnl: float = 0.0
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.fill_count: int = 0
        self.round_trip_count: int = 0

        # Feature accumulators
        self.price_history: list[float] = []
        self.ticks_data: Optional[list[dict[str, Any]]] = None

        if data_path and Path(data_path).exists():
            self._load_data(data_path)

    def _load_data(self, path: str) -> None:
        """Load tick data from Parquet file (Req 8.6)."""
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            self.ticks_data = df.to_dict(orient="records")
            logger.info("Loaded %d tick events from %s", len(self.ticks_data), path)
        except Exception as e:
            logger.warning("Could not load Parquet data from %s: %s; using synthetic simulator", path, e)
            self.ticks_data = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to start a new episode (Req 8.5)."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self._seed = seed

        self.step_idx = 0
        self.net_position = 0
        self.cash = 0.0
        self.initial_equity = 0.0
        self.equity = 0.0
        self.accounting_pnl = 0.0
        self.cumulative_reward = 0.0
        self.mid_price = float(self.initial_price)
        self.prev_mid_price = float(self.initial_price)
        self.best_bid = self.initial_price - self.tick_size
        self.best_ask = self.initial_price + self.tick_size
        self.cumulative_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.fill_count = 0
        self.round_trip_count = 0
        self.price_history = [self.mid_price]

        obs = self._compute_observation()
        info: dict[str, Any] = {
            "mid_price": self.mid_price,
            "net_position": self.net_position,
            "cash": self.cash,
            "equity": self.equity,
            "accounting_pnl": self.accounting_pnl,
            "pnl": self.accounting_pnl,
            "cumulative_reward": self.cumulative_reward,
        }
        return obs, info

    def _compute_observation(self) -> np.ndarray:
        """Compute the 8-dimensional MarketFeatures vector (Req 2.1)."""
        # 1 & 2. OFI Level 1 & Level 5 simulation
        # Simulate order imbalance using random walk or historical depths
        ofi_l1 = float(np.clip(self.np_random.normal(0.0, 0.4), -1.0, 1.0))
        ofi_l5 = float(np.clip(ofi_l1 * 0.8 + self.np_random.normal(0.0, 0.2), -1.0, 1.0))

        # 3. Inventory skew in [-1, 1] (Req 2.3)
        inventory_skew = float(self.net_position / self.max_position)

        # 4. Inventory urgency in [0, 1] (Req 2.4)
        inventory_urgency = float(abs(self.net_position) / self.max_position)

        # 5. Volatility regime: rolling realized vol z-score (Req 2.5)
        if len(self.price_history) >= 2:
            returns = np.diff(np.log(np.maximum(self.price_history[-self.vol_window:], 1.0)))
            vol = float(np.std(returns)) if len(returns) > 1 else 0.001
            vol_z = (vol - 0.001) / 0.0005
            vol_regime = float(np.clip(vol_z, -5.0, 5.0))
        else:
            vol_regime = 0.0

        # 6. Micro-price offset in ticks (Req 2.6)
        spread = max(self.tick_size, self.best_ask - self.best_bid)
        micro_offset = float(np.clip((ofi_l1 * spread * 0.5) / self.tick_size, -5.0, 5.0))

        # 7. Spread in ticks (Req 2.7)
        spread_ticks = float(spread / self.tick_size)

        # 8. Session progress in [0, 1] (Req 2.8)
        session_progress = float(min(1.0, self.step_idx / max(1, self.session_steps)))

        obs = np.array(
            [
                ofi_l1,
                ofi_l5,
                inventory_skew,
                inventory_urgency,
                vol_regime,
                micro_offset,
                spread_ticks,
                session_progress,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the environment by one step given agent action (Req 6.2)."""
        # Validate action
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Invalid action: {action}")

        delta_bid, delta_ask, size_frac = float(action[0]), float(action[1]), float(action[2])
        size_frac = float(np.clip(size_frac, 0.01, 1.0))

        self.prev_mid_price = self.mid_price

        # 1. Market evolution (tick step)
        price_drift = self.np_random.normal(0.0, 1.0) * self.tick_size
        new_mid = max(self.tick_size * 2, int(round((self.mid_price + price_drift) / self.tick_size)) * self.tick_size)
        self.mid_price = float(new_mid)
        self.price_history.append(self.mid_price)
        if len(self.price_history) > self.vol_window * 2:
            self.price_history.pop(0)

        self.best_bid = int(self.mid_price - self.tick_size)
        self.best_ask = int(self.mid_price + self.tick_size)
        half_spread_ticks = max(0.5, (self.best_ask - self.best_bid) / (2.0 * self.tick_size))

        # 2. Translate agent continuous action to integer tick quotes (Req 5.3)
        quote_bid = int(round(self.mid_price - delta_bid * self.tick_size))
        quote_ask = int(round(self.mid_price + delta_ask * self.tick_size))

        # Enforce spread >= 1 tick (Req 5.2)
        if quote_ask - quote_bid < self.tick_size:
            quote_ask = quote_bid + self.tick_size

        # Quote sizes scaled by size_frac (Req 5.4)
        bid_qty = max(1, int(round(size_frac * self.max_order_size)))
        ask_qty = max(1, int(round(size_frac * self.max_order_size)))

        # Inventory halt rule (Req 5.5, 9.5)
        # When |net_position| >= max_position * 0.9, zero offending-side qty and double opposite spread
        if self.net_position >= int(self.max_position * 0.9):
            bid_qty = 0
            quote_ask = int(round(self.mid_price + delta_ask * 2.0 * self.tick_size))
        elif self.net_position <= -int(self.max_position * 0.9):
            ask_qty = 0
            quote_bid = int(round(self.mid_price - delta_bid * 2.0 * self.tick_size))

        # 3. Simulate fill execution against order flow
        # Probability of fill increases when quotes are closer to or cross the mid
        fill_bid_prob = 1.0 / (1.0 + np.exp(max(-10.0, min(10.0, (quote_bid - self.mid_price) / (-self.tick_size * 2.0)))))
        fill_ask_prob = 1.0 / (1.0 + np.exp(max(-10.0, min(10.0, (self.mid_price - quote_ask) / (-self.tick_size * 2.0)))))

        filled_bid = (self.np_random.random() < fill_bid_prob) and (bid_qty > 0)
        filled_ask = (self.np_random.random() < fill_ask_prob) and (ask_qty > 0)

        # Track position before fills to compute accurate round-trip closure
        prev_pos = self.net_position
        spread_capture_qty = 0
        is_round_trip = False

        if filled_bid and filled_ask:
            # Both sides filled simultaneously:
            # Cash accounting
            self.cash -= float(quote_bid * bid_qty)
            self.cash += float(quote_ask * ask_qty)

            # 1. Matched volume between buy and sell earns full spread (2 x half-spread)
            matched_qty = min(bid_qty, ask_qty)
            spread_capture_qty += matched_qty * 2
            is_round_trip = True
            self.round_trip_count += 1

            # 2. Imbalance interacts with existing inventory
            imbalance = bid_qty - ask_qty
            if imbalance > 0 and prev_pos < 0:
                # Excess buy unwinds existing short
                unwound = min(abs(prev_pos), imbalance)
                spread_capture_qty += unwound
            elif imbalance < 0 and prev_pos > 0:
                # Excess sell unwinds existing long
                unwound = min(prev_pos, abs(imbalance))
                spread_capture_qty += unwound

            self.net_position += imbalance

        elif filled_bid:
            # Single buy fill
            self.cash -= float(quote_bid * bid_qty)
            if prev_pos < 0:
                closed_qty = min(abs(prev_pos), bid_qty)
                spread_capture_qty = closed_qty
                is_round_trip = True
                self.round_trip_count += 1
            self.net_position += bid_qty

        elif filled_ask:
            # Single sell fill
            self.cash += float(quote_ask * ask_qty)
            if prev_pos > 0:
                closed_qty = min(prev_pos, ask_qty)
                spread_capture_qty = closed_qty
                is_round_trip = True
                self.round_trip_count += 1
            self.net_position -= ask_qty

        if filled_bid or filled_ask:
            self.fill_count += 1

        # True Financial Equity & Accounting PnL
        self.equity = self.cash + float(self.net_position * self.mid_price)
        self.accounting_pnl = self.equity - self.initial_equity

        # 4. Compute reward signal via RewardShaper (Req 8.1, 8.2)
        # Pass actual round-trip quantity closed for spread capture credit
        reward_signal: RewardSignal = self.reward_shaper.compute(
            net_position=self.net_position,
            fill_qty=spread_capture_qty,
            fill_price=self.mid_price,
            prev_mid_price=self.prev_mid_price,
            curr_mid_price=self.mid_price,
            is_round_trip=is_round_trip,
            half_spread_ticks=half_spread_ticks,
        )

        step_reward = reward_signal.total()
        self.cumulative_reward += step_reward
        self.cumulative_pnl = self.accounting_pnl

        self.step_idx += 1

        # 5. Check termination / truncation
        terminated = abs(self.net_position) > self.max_position  # hard position breach (Req 8.4)
        truncated = self.step_idx >= self.session_steps         # session completed

        obs = self._compute_observation()
        info = {
            "step": self.step_idx,
            "mid_price": self.mid_price,
            "net_position": self.net_position,
            "cash": self.cash,
            "equity": self.equity,
            "accounting_pnl": self.accounting_pnl,
            "cumulative_pnl": self.accounting_pnl,
            "step_reward": step_reward,
            "cumulative_reward": self.cumulative_reward,
            "realized_pnl": reward_signal.realized_pnl,
            "inventory_penalty": reward_signal.inventory_penalty,
            "spread_capture": reward_signal.spread_capture,
            "fill_count": self.fill_count,
            "round_trip_count": self.round_trip_count,
        }

        return obs, step_reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """Render human-readable summary of book state and PnL (Req 8.7, 13.6)."""
        spread = self.best_ask - self.best_bid
        print(
            f"[LobEnv Step {self.step_idx:04d}] "
            f"Bid: {self.best_bid} | Ask: {self.best_ask} (Spread: {spread}) | "
            f"Mid: {self.mid_price:.1f} | "
            f"Pos: {self.net_position:+d} / {self.max_position} | "
            f"PnL: {self.cumulative_pnl:+.2f} ticks | "
            f"Fills: {self.fill_count}"
        )
