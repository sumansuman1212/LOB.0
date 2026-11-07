"""
reward_shaper.py — Reward Signal Computation for PPO Market-Making Training

Implements ``RewardSignal`` (a decomposed per-step reward) and ``RewardShaper``
(the stateless computation class that produces it) for use by ``LobEnv`` during
PPO rollout collection.

Reward decomposition (Requirements 8.1, 8.2, 8.3):

    total = realized_pnl + inventory_penalty + spread_capture

Components
----------
realized_pnl     — mark-to-market change in ticks; can be positive or negative.
inventory_penalty — −γ × |net_position| / max_position; always ≤ 0.
spread_capture   — half-spread × fill_qty earned on completed round-trip fills;
                   always ≥ 0.

Sign invariants are enforced by assertions in ``RewardSignal.total()``.

Requirements: 8.1, 8.2, 8.3
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# RewardSignal
# ---------------------------------------------------------------------------

@dataclass
class RewardSignal:
    """Per-step reward decomposition for diagnostic logging and PPO training.

    Attributes
    ----------
    realized_pnl:
        Mark-to-market change in ticks for the current net position.
        Defined as ``(curr_mid_price - prev_mid_price) * net_position``.
        Unconstrained — can be negative when the position moves against mid.
    inventory_penalty:
        ``-γ × |net_position| / max_position``.  Always ≤ 0; discourages
        accumulation of large directional exposure (Req 8.2, 8.3).
    spread_capture:
        Half-spread in ticks earned on a completed round-trip fill.
        ``fill_qty × half_spread_ticks`` when ``is_round_trip=True``,
        else 0.  Always ≥ 0 (Req 8.2).

    Requirements: 8.1, 8.2, 8.3
    """

    realized_pnl: float       # mark-to-market change in ticks (unconstrained)
    inventory_penalty: float  # -γ · |net_position| / max_position  (≤ 0)
    spread_capture: float     # half-spread earned on round-trip fills  (≥ 0)

    def total(self) -> float:
        """Return the sum of all reward components.

        Asserts sign constraints at runtime:
        - ``inventory_penalty ≤ 0`` (always a cost, not a reward)
        - ``spread_capture ≥ 0`` (only earned, never subtracted)

        Returns
        -------
        float
            ``realized_pnl + inventory_penalty + spread_capture``

        Requirements: 8.1, 8.2
        """
        assert self.inventory_penalty <= 0.0, (
            f"inventory_penalty must be non-positive, got {self.inventory_penalty}"
        )
        assert self.spread_capture >= 0.0, (
            f"spread_capture must be non-negative, got {self.spread_capture}"
        )
        return self.realized_pnl + self.inventory_penalty + self.spread_capture


# ---------------------------------------------------------------------------
# RewardShaper
# ---------------------------------------------------------------------------

class RewardShaper:
    """Stateless reward computation for the LobEnv training environment.

    Translates a fill event and current market state into a ``RewardSignal``
    with three independently interpretable components.

    Parameters
    ----------
    gamma:
        Inventory penalty coefficient γ ≥ 0.  Controls how strongly the
        agent is penalised for holding a large net position.  Corresponds
        to ``StrategyConfig.inventory_penalty`` (Req 8.3).
    max_position:
        Maximum allowed absolute net position in lots.  Used to normalise
        the inventory penalty to ``[-1, 0]`` regardless of position scale.

    Requirements: 8.1, 8.2, 8.3
    """

    def __init__(self, gamma: float, max_position: int) -> None:
        if gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if max_position <= 0:
            raise ValueError(
                f"max_position must be a positive integer, got {max_position}"
            )

        self.gamma = gamma
        self.max_position = max_position

    def compute(
        self,
        net_position: int,
        fill_qty: int,
        fill_price: float,  # noqa: ARG002  — reserved for future mtm extensions
        prev_mid_price: float,
        curr_mid_price: float,
        is_round_trip: bool,
        half_spread_ticks: float,
    ) -> RewardSignal:
        """Compute the reward signal for one environment step.

        Parameters
        ----------
        net_position:
            Signed net position *after* the fill event (positive = long,
            negative = short).
        fill_qty:
            Number of lots filled in this event.  Must be ≥ 0.
        fill_price:
            Price at which the fill occurred (in ticks).  Currently reserved
            for future mark-to-market extensions; not used in the primary
            formula.
        prev_mid_price:
            Mid-price *before* the current tick event (in ticks).
        curr_mid_price:
            Mid-price *after* the current tick event (in ticks).
        is_round_trip:
            ``True`` if this fill completes a round-trip (a matched buy and
            sell pair), in which case ``spread_capture`` is credited.
        half_spread_ticks:
            Half of the prevailing bid-ask spread in ticks at the time of the
            fill.  Must be ≥ 0.

        Returns
        -------
        RewardSignal
            Decomposed reward with sign invariants validated by
            ``RewardSignal.total()``.

        Raises
        ------
        ValueError
            If ``fill_qty < 0`` or ``half_spread_ticks < 0``.

        Requirements: 8.1, 8.2, 8.3
        """
        if fill_qty < 0:
            raise ValueError(f"fill_qty must be non-negative, got {fill_qty}")
        if half_spread_ticks < 0.0:
            raise ValueError(
                f"half_spread_ticks must be non-negative, got {half_spread_ticks}"
            )

        # ------------------------------------------------------------------
        # 1. Realized PnL — mark-to-market change in ticks (Req 8.1)
        #
        #    Δ_mtm = (curr_mid - prev_mid) × net_position
        #
        #    Positive when the mid moves in favour of the position (price up
        #    for a long, price down for a short).  Can be negative.
        # ------------------------------------------------------------------
        realized_pnl: float = (curr_mid_price - prev_mid_price) * net_position

        # ------------------------------------------------------------------
        # 2. Inventory penalty — discourages large positions (Req 8.2, 8.3)
        #
        #    penalty = −γ × |net_position| / max_position
        #
        #    Always ≤ 0.  When net_position = 0 the penalty is zero.
        #    When |net_position| = max_position the penalty is −γ.
        # ------------------------------------------------------------------
        inventory_penalty: float = (
            -self.gamma * abs(net_position) / self.max_position
        )

        # ------------------------------------------------------------------
        # 3. Spread capture — earned on completed round-trip fills (Req 8.2)
        #
        #    capture = fill_qty × half_spread_ticks  if is_round_trip else 0
        #
        #    Always ≥ 0.  Represents the half-spread income realized when a
        #    market-making cycle (buy + sell) closes.
        # ------------------------------------------------------------------
        spread_capture: float = (
            fill_qty * half_spread_ticks if is_round_trip else 0.0
        )

        return RewardSignal(
            realized_pnl=realized_pnl,
            inventory_penalty=inventory_penalty,
            spread_capture=spread_capture,
        )
