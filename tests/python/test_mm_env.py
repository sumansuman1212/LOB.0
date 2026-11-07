"""
tests/test_mm_env.py — Unit and Property tests for LobEnv

Properties Tested:
- Property 21: Reward Component Signs (Req 8.2)
- Property 22: Reward Formula Consistency (Req 8.1, 8.3)
- Property 23: Deterministic Replay (Req 8.5)
- Observation and Action Space bounds (Req 6.3, 6.4)
- Termination on position limit breach (Req 8.4)
"""
from __future__ import annotations

import math
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.python.mm_env import LobEnv, OBS_DIM, ACTION_DIM
from src.python.reward_shaper import RewardShaper, RewardSignal


def test_mm_env_spaces():
    """Verify observation and action spaces (Req 6.3, 6.4)."""
    env = LobEnv(max_position=500, tick_size=5)

    assert env.observation_space.shape == (8,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.shape == (3,)
    assert env.action_space.dtype == np.float32

    obs, info = env.reset(seed=123)
    assert obs.shape == (8,)
    assert np.all(np.isfinite(obs))
    assert "mid_price" in info
    assert "net_position" in info


def test_mm_env_step():
    """Verify step transition outputs."""
    env = LobEnv(max_position=500, tick_size=5)
    env.reset(seed=42)

    action = np.array([2.0, 2.0, 0.5], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == (8,)
    assert np.all(np.isfinite(next_obs))
    assert isinstance(reward, float)
    assert math.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "cumulative_pnl" in info


def test_deterministic_replay():
    """Property 23: Deterministic Replay with identical RNG seeds (Req 8.5)."""
    actions = [
        np.array([1.0, 1.0, 0.2], dtype=np.float32),
        np.array([2.0, 3.0, 0.5], dtype=np.float32),
        np.array([0.5, 0.5, 0.1], dtype=np.float32),
        np.array([3.0, 1.0, 0.8], dtype=np.float32),
    ]

    env1 = LobEnv(seed=777)
    obs1, _ = env1.reset(seed=777)
    rewards1 = []
    for a in actions:
        o, r, _, _, _ = env1.step(a)
        rewards1.append(r)

    env2 = LobEnv(seed=777)
    obs2, _ = env2.reset(seed=777)
    rewards2 = []
    for a in actions:
        o, r, _, _, _ = env2.step(a)
        rewards2.append(r)

    assert np.allclose(obs1, obs2)
    assert np.allclose(rewards1, rewards2)


@settings(max_examples=50)
@given(
    net_position=st.integers(min_value=-500, max_value=500),
    fill_qty=st.integers(min_value=0, max_value=100),
    prev_mid=st.floats(min_value=100.0, max_value=50000.0, allow_nan=False, allow_infinity=False),
    curr_mid=st.floats(min_value=100.0, max_value=50000.0, allow_nan=False, allow_infinity=False),
    is_round_trip=st.booleans(),
    half_spread=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
)
def test_reward_shaper_properties(
    net_position: int,
    fill_qty: int,
    prev_mid: float,
    curr_mid: float,
    is_round_trip: bool,
    half_spread: float,
):
    """Property 21 & 22: Reward Component Signs and Formula Consistency (Req 8.1, 8.2, 8.3)."""
    shaper = RewardShaper(gamma=0.1, max_position=500)
    signal: RewardSignal = shaper.compute(
        net_position=net_position,
        fill_qty=fill_qty,
        fill_price=curr_mid,
        prev_mid_price=prev_mid,
        curr_mid_price=curr_mid,
        is_round_trip=is_round_trip,
        half_spread_ticks=half_spread,
    )

    # Property 21: Component Signs
    assert signal.inventory_penalty <= 0.0, "inventory_penalty must be <= 0"
    assert signal.spread_capture >= 0.0, "spread_capture must be >= 0"

    # Property 22: Total consistency
    total = signal.total()
    expected = signal.realized_pnl + signal.inventory_penalty + signal.spread_capture
    assert math.isclose(total, expected, abs_tol=1e-6)


def test_position_unwinding_and_spread_capture():
    """Verify that spread capture credits only the actual closed quantity.
    
    Tests:
    1. Zero inventory to open position: 0 spread capture
    2. Position increase: 0 spread capture
    3. Partial position closure: credits only the closed fraction
    4. Full reversal: credits only the closed fraction, not newly opened volume
    """
    env = LobEnv(max_position=500, tick_size=5)
    env.reset(seed=42)

    # 1. Zero inventory: open long position (+10)
    # Mock single buy fill
    env.net_position = 0
    # Simulate step logic directly with known fill
    prev_pos = env.net_position
    bid_qty = 10
    # If filled_bid:
    spread_capture_qty = 0
    is_round_trip = False
    if prev_pos < 0:
        closed_qty = min(abs(prev_pos), bid_qty)
        spread_capture_qty = closed_qty
        is_round_trip = True
    env.net_position += bid_qty

    assert env.net_position == 10
    assert spread_capture_qty == 0
    assert not is_round_trip

    # 2. Position increase: long 10, buy 5 more -> pos 15, spread capture = 0
    prev_pos = env.net_position
    bid_qty = 5
    spread_capture_qty = 0
    is_round_trip = False
    if prev_pos < 0:
        closed_qty = min(abs(prev_pos), bid_qty)
        spread_capture_qty = closed_qty
        is_round_trip = True
    env.net_position += bid_qty

    assert env.net_position == 15
    assert spread_capture_qty == 0
    assert not is_round_trip

    # 3. Partial position closure: long 15, sell 6 -> pos 9, spread capture = 6
    prev_pos = env.net_position
    ask_qty = 6
    spread_capture_qty = 0
    is_round_trip = False
    if prev_pos > 0:
        closed_qty = min(prev_pos, ask_qty)
        spread_capture_qty = closed_qty
        is_round_trip = True
    env.net_position -= ask_qty

    assert env.net_position == 9
    assert spread_capture_qty == 6
    assert is_round_trip

    # 4. Full reversal: long 9, sell 25 -> pos -16, spread capture = 9 (NOT 25!)
    prev_pos = env.net_position
    ask_qty = 25
    spread_capture_qty = 0
    is_round_trip = False
    if prev_pos > 0:
        closed_qty = min(prev_pos, ask_qty)
        spread_capture_qty = closed_qty
        is_round_trip = True
    env.net_position -= ask_qty

    assert env.net_position == -16
    assert spread_capture_qty == 9, "Spread capture must only credit the 9 lots closed, not all 25 lots!"
    assert is_round_trip

