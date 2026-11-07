"""
ppo_agent.py — PPO Actor-Critic Network and Agent

Implements the ``ActorCritic`` neural network (shared MLP trunk + Gaussian
actor head + scalar critic head) and the ``PPOAgent`` wrapper that holds the
network and its optimizer.

Architecture
------------
- Shared trunk : Linear(8, 256) → Tanh → Linear(256, 256) → Tanh
- Actor mean   : Linear(256, 3)          — mean of Gaussian action distribution
- Actor log_std: nn.Parameter(zeros(3))  — learnable, clamped to [-4, 2]
- Critic       : Linear(256, 1)          — scalar state-value V(s)

Requirements: 3.1, 3.2, 3.8
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ---------------------------------------------------------------------------
# Constants (must match MarketFeatures::DIM and QuoteAction dims)
# ---------------------------------------------------------------------------
OBS_DIM: int = 8
ACTION_DIM: int = 3
HIDDEN_SIZE: int = 256

# log_std clamp range (Req 3.1)
LOG_STD_MIN: float = -4.0
LOG_STD_MAX: float = 2.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ActorCritic network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Shared-trunk actor-critic network for PPO.

    The trunk is a 3-layer MLP (input → 256 → 256) with ``nn.Tanh``
    activations shared by both the actor and critic heads.

    Actor head
    ----------
    - ``mean_head``: Linear(256, 3) → action means for (delta_bid, delta_ask, size_frac)
    - ``log_std``  : ``nn.Parameter`` of shape (3,), clamped to ``[-4, 2]`` in
      every forward pass to keep the Gaussian well-conditioned.

    Critic head
    -----------
    - ``value_head``: Linear(256, 1) → squeezed to a scalar / batch vector V(s).

    Parameters
    ----------
    obs_dim:
        Observation space dimension. Must be 8 to match ``MarketFeatures::DIM``.
    action_dim:
        Action space dimension. Must be 3 for (delta_bid, delta_ask, size_frac).
    hidden_size:
        Width of each hidden layer in the shared trunk.

    Requirements: 3.1, 3.2, 3.8
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_size: int = HIDDEN_SIZE,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Shared trunk: Linear(obs_dim→hidden) → Tanh → Linear(hidden→hidden) → Tanh
        # ------------------------------------------------------------------
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        # ------------------------------------------------------------------
        # Actor head
        # ------------------------------------------------------------------
        # Mean of the Gaussian: Linear(hidden → action_dim)
        self.mean_head = nn.Linear(hidden_size, action_dim)

        # log_std is a *learnable parameter* (not a linear layer).
        # Initialised to zeros → initial std = exp(0) = 1.
        # Clamped to [LOG_STD_MIN, LOG_STD_MAX] on every forward pass.
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # ------------------------------------------------------------------
        # Critic head: Linear(hidden → 1) → squeeze to scalar
        # ------------------------------------------------------------------
        self.value_head = nn.Linear(hidden_size, 1)

        # ------------------------------------------------------------------
        # Weight initialisation — orthogonal for trunk layers, zero bias;
        # small uniform for output layers for stable early training.
        # ------------------------------------------------------------------
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action distribution parameters and value estimate.

        Parameters
        ----------
        obs:
            Observation tensor of shape ``(..., obs_dim)``.

        Returns
        -------
        mean:
            Action means, shape ``(..., action_dim)``.
        log_std:
            Clamped log-standard-deviations, shape ``(action_dim,)``
            (broadcast-compatible with ``mean``).
        value:
            Scalar value estimate, shape ``(...)``  — the trailing singleton
            dimension from ``value_head`` is squeezed away.
        """
        features = self.trunk(obs)

        # Actor outputs
        mean = self.mean_head(features)

        # Clamp log_std to [-4, 2] on every forward pass (Req 3.1)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

        # Critic output — squeeze the trailing dim-1 to get a scalar per sample
        value = self.value_head(features).squeeze(-1)

        return mean, log_std, value

    # ------------------------------------------------------------------
    # act
    # ------------------------------------------------------------------

    def act(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and compute its log-probability.

        Samples from ``Normal(mean, exp(log_std))`` and sums the
        per-dimension log-probabilities so that the returned ``log_prob``
        is the joint log-probability of the full action vector.

        Parameters
        ----------
        obs:
            Observation tensor of shape ``(..., obs_dim)``.

        Returns
        -------
        action:
            Sampled action of shape ``(..., action_dim)``.
        log_prob:
            Sum of per-dimension log-probabilities, shape ``(...)``.
            This is the scalar ``log π_θ(a|s)`` expected by PPO.
        """
        mean, log_std, _value = self.forward(obs)
        std = log_std.exp()

        dist = Normal(mean, std)
        action = dist.rsample()  # reparameterised sample — gradients flow through

        # Sum log-probs over action dimensions → scalar log π_θ(a|s)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------

class ActorInferenceModule(nn.Module):
    """Standalone actor module for TorchScript export and C++ inference.

    Takes an 8-dimensional observation tensor and produces the 3-dimensional
    action mean ``(delta_bid, delta_ask, size_frac)``.
    """

    def __init__(self, trunk: nn.Module, mean_head: nn.Module) -> None:
        super().__init__()
        self.trunk = trunk
        self.mean_head = mean_head

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.trunk(obs)
        return self.mean_head(features)


class PPOAgent:
    """PPO agent holding the ``ActorCritic`` network and its optimizer.

    This class owns the network, the Adam optimizer, and PPO hyperparameters.

    Parameters
    ----------
    obs_dim:
        Observation space dimension (default 8).
    action_dim:
        Action space dimension (default 3).
    hidden_size:
        Width of hidden layers in the shared trunk (default 256).
    clip_eps:
        PPO clip parameter ε ∈ (0, 0.5) (default 0.2).
    gamma:
        Discount factor γ ∈ (0, 1) (default 0.99).
    lam:
        GAE smoothing λ ∈ (0, 1) (default 0.95).
    lr:
        Adam learning rate (default 3e-4).
    entropy_coef:
        Entropy bonus coefficient (default 0.01).
    device:
        Torch device to run the network on.  Defaults to ``"cuda"`` if
        available, otherwise ``"cpu"``.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_size: int = HIDDEN_SIZE,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 3e-4,
        entropy_coef: float = 0.01,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef

        # Network
        self.model: ActorCritic = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
        ).to(self.device)

        # Optimizer — all parameters (trunk + actor head + critic head + log_std)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            eps=1e-5,
        )

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------

    def train_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, float]:
        """Run one PPO gradient update step.

        Parameters
        ----------
        obs:
            Batch of observations, shape ``(N, obs_dim)``.
        actions:
            Batch of actions taken, shape ``(N, action_dim)``.
        old_log_probs:
            Log-probabilities under the *old* policy, shape ``(N,)``.
        advantages:
            Normalised GAE advantages, shape ``(N,)``.
        returns:
            GAE returns (targets for value loss), shape ``(N,)``.

        Returns
        -------
        dict with keys ``policy_loss``, ``value_loss``, ``entropy``, ``total_loss``.

        Requirements: 3.3, 3.4, 3.5, 3.6, 3.7
        """
        self.model.train()

        # 1. Forward pass through actor-critic
        mean, log_std, values = self.model.forward(obs)

        # 2. Compute new log_probs and entropy from Normal(mean, exp(log_std))
        dist = Normal(mean, log_std.exp())
        new_log_probs = dist.log_prob(actions).sum(dim=-1)  # shape (N,)
        entropy = dist.entropy().sum(dim=-1)                # shape (N,)

        # 3. Compute probability ratio r = exp(new_log_prob - old_log_prob)
        ratio = (new_log_probs - old_log_probs).exp()

        # 4. Clipped surrogate objective (Req 3.3)
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 5. Value loss — 0.5 × MSE(V(s), returns) (Req 3.4, 3.5)
        value_loss = 0.5 * F.mse_loss(values, returns)

        # 6. Entropy bonus — encourages exploration (Req 3.4)
        entropy_loss = -self.entropy_coef * entropy.mean()

        # 7. Total loss
        loss = policy_loss + value_loss + entropy_loss

        # 8. Assert finite before backprop (Req 11.5)
        assert loss.isfinite(), f"Loss is not finite: {loss.item()}"

        # 9. Optimizer step with gradient clipping (Req 3.6)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "total_loss": loss.item(),
        }

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------

    def export(self, model_path: str) -> str:
        """Export the actor network as a frozen TorchScript model.

        Parameters
        ----------
        model_path:
            Destination path for the ``.pt`` TorchScript file.

        Returns
        -------
        str
            SHA-256 hex digest of the exported file.

        Requirements: 3.10, 6.9, 11.8
        """
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        actor_inference = ActorInferenceModule(self.model.trunk, self.model.mean_head).to("cpu")
        scripted = torch.jit.script(actor_inference)
        scripted.save(str(path))

        # Compute SHA-256
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        digest = sha256.hexdigest()

        # Persist checksum file alongside .pt
        checksum_path = path.with_name(path.name + ".sha256")
        checksum_path.write_text(digest)

        logger.info("Exported TorchScript model to %s with SHA-256 %s", path, digest)
        return digest
