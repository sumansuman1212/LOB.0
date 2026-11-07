"""
trajectory_buffer.py — PPO Trajectory Buffer with GAE-λ Advantage Estimation

Stores rollout transitions in pre-allocated contiguous NumPy arrays and
computes Generalized Advantage Estimation (GAE-λ) for PPO training.

Requirements: 4.1–4.10, 11.5
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch

# Number of observation dimensions — must match MarketFeatures::DIM.
OBS_DIM: int = 8
# Number of action dimensions — (delta_bid, delta_ask, size_frac).
ACTION_DIM: int = 3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transition dataclass
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """A single environment step transition stored in the trajectory buffer.

    Fields
    ------
    obs       : float array of shape (OBS_DIM,)  — observation at step t
    action    : float array of shape (ACTION_DIM,) — action taken at step t
    log_prob  : float — log π_θ(a|s); must be finite (Req 4.10, 11.5)
    value     : float — V(s) estimate from the critic
    reward    : float — r_t received from the environment
    done      : bool  — episode terminal flag
    """
    obs: np.ndarray      # shape (OBS_DIM,)
    action: np.ndarray   # shape (ACTION_DIM,)
    log_prob: float
    value: float
    reward: float
    done: bool


# ---------------------------------------------------------------------------
# TrajectoryBuffer
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    """Pre-allocated ring-style buffer for PPO rollout transitions.

    Capacity must be a power of two (Req 4.2). All internal storage is
    pre-allocated at construction time and reused via ``clear()`` to avoid
    heap churn on the training hot path (Req 4.9).

    Usage
    -----
    >>> buf = TrajectoryBuffer(capacity=2048)
    >>> buf.push(transition)           # add a step
    >>> buf.compute_advantages(0.99, 0.95, last_value=v)
    >>> obs_t   = buf.obs_tensor()     # zero-copy torch view
    >>> adv_t   = buf.advantage_tensor()
    >>> buf.clear()                    # reset without releasing memory
    """

    def __init__(self, capacity: int) -> None:
        """Initialise the buffer and pre-allocate all storage arrays.

        Parameters
        ----------
        capacity:
            Maximum number of transitions the buffer can hold before a
            ``compute_advantages`` / ``clear`` cycle is required.
            **Must be a power of two** (Req 4.2).

        Raises
        ------
        ValueError
            If ``capacity`` is not a positive power of two.
        """
        if capacity <= 0 or (capacity & (capacity - 1)) != 0:
            raise ValueError(
                f"TrajectoryBuffer capacity must be a positive power of two, "
                f"got {capacity}"
            )

        self._capacity: int = capacity
        self._idx: int = 0          # next-write position
        self._advantages_ready: bool = False

        # ------------------------------------------------------------------
        # Pre-allocate contiguous NumPy arrays (Req 4.1)
        # ------------------------------------------------------------------
        self._obs      = np.zeros((capacity, OBS_DIM),    dtype=np.float32)
        self._actions  = np.zeros((capacity, ACTION_DIM), dtype=np.float32)
        self._log_probs = np.zeros(capacity,              dtype=np.float32)
        self._values   = np.zeros(capacity,               dtype=np.float32)
        self._rewards  = np.zeros(capacity,               dtype=np.float32)
        self._dones    = np.zeros(capacity,               dtype=np.bool_)

        # Computed by compute_advantages(); not valid until that call returns.
        self._advantages = np.zeros(capacity, dtype=np.float32)
        self._returns    = np.zeros(capacity, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum number of transitions this buffer can hold."""
        return self._capacity

    @property
    def size(self) -> int:
        """Number of transitions currently stored."""
        return self._idx

    def full(self) -> bool:
        """Return True when the buffer has reached capacity."""
        return self._idx >= self._capacity

    def push(self, transition: Transition) -> bool:
        """Store one transition at the current write index.

        Rejects transitions whose ``log_prob`` is non-finite (NaN or ±Inf)
        and logs an error without mutating any buffer state (Req 4.10, 11.5).

        Parameters
        ----------
        transition:
            The ``Transition`` to store.

        Returns
        -------
        bool
            ``True`` if the transition was accepted, ``False`` if rejected.

        Raises
        ------
        RuntimeError
            If the buffer is already full (capacity reached without ``clear``).
        """
        # Guard: reject non-finite log_prob (Req 4.10, 11.5)
        if not math.isfinite(transition.log_prob):
            logger.error(
                "TrajectoryBuffer.push: rejected transition with non-finite "
                "log_prob=%s at buffer index %d. Transition not stored.",
                transition.log_prob,
                self._idx,
            )
            return False

        if self.full():
            raise RuntimeError(
                "TrajectoryBuffer is full (capacity=%d). Call clear() before "
                "pushing more transitions." % self._capacity
            )

        i = self._idx

        # Copy transition data into pre-allocated arrays
        np.copyto(self._obs[i], transition.obs)
        np.copyto(self._actions[i], transition.action)
        self._log_probs[i] = transition.log_prob
        self._values[i]    = transition.value
        self._rewards[i]   = transition.reward
        self._dones[i]     = transition.done

        self._idx += 1
        self._advantages_ready = False
        return True

    def compute_advantages(
        self,
        gamma: float,
        lam: float,
        last_value: float = 0.0,
    ) -> None:
        """Compute GAE-λ advantages and returns for the stored rollout.

        Iterates over stored transitions in **reverse chronological order**
        (Req 4.3). Terminal steps (`done=True`) bootstrap with
        ``next_value = 0`` (Req 4.4); the non-terminal final step uses the
        provided ``last_value`` from the critic (Req 4.5). Advantages are
        then normalized to mean ≈ 0 and std ≈ 1 (Req 4.6) and asserted
        finite (Req 4.7).

        Parameters
        ----------
        gamma:
            Discount factor γ ∈ (0, 1).
        lam:
            GAE smoothing parameter λ ∈ (0, 1).
        last_value:
            Critic estimate V(s_{T}) for the state following the final
            stored transition. Pass 0.0 if the episode ended terminally
            (Req 4.4). Defaults to 0.0 so callers that explicitly reached a
            terminal state need not supply it.

        Raises
        ------
        RuntimeError
            If the buffer is empty.
        AssertionError
            If any computed advantage or return value is non-finite after
            normalization (Req 4.7).
        """
        T = self._idx
        if T == 0:
            raise RuntimeError("compute_advantages() called on empty buffer.")

        rewards  = self._rewards[:T]
        values   = self._values[:T]
        dones    = self._dones[:T]

        advantages = self._advantages  # write directly into pre-alloc array
        gae = 0.0

        # ------------------------------------------------------------------
        # Reverse-order GAE-λ sweep (Req 4.3)
        # ------------------------------------------------------------------
        for t in range(T - 1, -1, -1):
            if dones[t]:
                # Terminal step: bootstrap value is 0 (Req 4.4)
                next_value = 0.0
                gae = 0.0  # reset GAE accumulator at episode boundary
            elif t == T - 1:
                # Non-terminal final step: use critic's last_value (Req 4.5)
                next_value = last_value
            else:
                next_value = float(values[t + 1])

            delta = float(rewards[t]) + gamma * next_value - float(values[t])
            gae   = delta + gamma * lam * (1.0 - float(dones[t])) * gae
            advantages[t] = gae

        # Returns = advantages + values (pre-normalization)
        self._returns[:T] = advantages[:T] + values

        # ------------------------------------------------------------------
        # Normalize advantages: (adv - mean) / (std + ε) (Req 4.6)
        # Compute mean and std in float64 to avoid float32 catastrophic cancellation
        # ------------------------------------------------------------------
        adv_slice = advantages[:T]
        adv_mean  = float(np.mean(adv_slice, dtype=np.float64))
        adv_std   = float(np.std(adv_slice, dtype=np.float64))
        if adv_std > 1e-6:
            adv_slice[:] = (adv_slice - adv_mean) / adv_std
        else:
            adv_slice[:] = 0.0

        # ------------------------------------------------------------------
        # Assert all values finite (Req 4.7)
        # ------------------------------------------------------------------
        assert np.all(np.isfinite(advantages[:T])), (
            "compute_advantages: non-finite values detected in advantages after "
            "normalization. Check reward/value inputs for NaN or overflow."
        )
        assert np.all(np.isfinite(self._returns[:T])), (
            "compute_advantages: non-finite values detected in returns. "
            "Check reward/value inputs for NaN or overflow."
        )

        self._advantages_ready = True

    # ------------------------------------------------------------------
    # Zero-copy tensor views (Req 4.8)
    # ------------------------------------------------------------------

    def obs_tensor(self) -> torch.Tensor:
        """Zero-copy view of the stored observations as a float32 tensor.

        Returns shape ``(size, OBS_DIM)``.

        Notes
        -----
        The returned tensor shares memory with the internal NumPy array.
        Do **not** modify the tensor after calling ``clear()``.
        """
        return torch.from_numpy(self._obs[: self._idx])

    def action_tensor(self) -> torch.Tensor:
        """Zero-copy view of stored actions as a float32 tensor.

        Returns shape ``(size, ACTION_DIM)``.
        """
        return torch.from_numpy(self._actions[: self._idx])

    def log_prob_tensor(self) -> torch.Tensor:
        """Zero-copy view of stored log-probabilities as a float32 tensor.

        Returns shape ``(size,)``.
        """
        return torch.from_numpy(self._log_probs[: self._idx])

    def value_tensor(self) -> torch.Tensor:
        """Zero-copy view of stored critic value estimates as a float32 tensor.

        Returns shape ``(size,)``.
        """
        return torch.from_numpy(self._values[: self._idx])

    def advantage_tensor(self) -> torch.Tensor:
        """Zero-copy view of computed advantages as a float32 tensor.

        Returns shape ``(size,)``.

        Raises
        ------
        RuntimeError
            If ``compute_advantages()`` has not been called since the last
            ``push()`` or ``clear()``.
        """
        self._require_advantages()
        return torch.from_numpy(self._advantages[: self._idx])

    def return_tensor(self) -> torch.Tensor:
        """Zero-copy view of computed returns (targets for value loss).

        Returns shape ``(size,)``.

        Raises
        ------
        RuntimeError
            If ``compute_advantages()`` has not been called since the last
            ``push()`` or ``clear()``.
        """
        self._require_advantages()
        return torch.from_numpy(self._returns[: self._idx])

    # ------------------------------------------------------------------
    # Reset (Req 4.9)
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset the write index without releasing allocated memory (Req 4.9).

        All pre-allocated NumPy arrays are retained; only the logical size and
        the advantages-ready flag are reset. This is safe to call after every
        rollout to reuse the same buffer across training episodes.
        """
        self._idx = 0
        self._advantages_ready = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_advantages(self) -> None:
        if not self._advantages_ready:
            raise RuntimeError(
                "Advantage/return tensors are not yet available. "
                "Call compute_advantages() before accessing advantage_tensor() "
                "or return_tensor()."
            )

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self._idx

    @property
    def size(self) -> int:
        """Return the number of stored transitions."""
        return self._idx

    def __repr__(self) -> str:
        return (
            f"TrajectoryBuffer(capacity={self._capacity}, "
            f"size={self._idx}, "
            f"advantages_ready={self._advantages_ready})"
        )
