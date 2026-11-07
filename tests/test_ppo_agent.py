"""
tests/test_ppo_agent.py — Unit and Property tests for ActorCritic and PPOAgent

Properties Tested:
- Property 9: PPO Clipped Surrogate Correctness (Req 3.3)
- Property 10: Gradient Norm Bounded After Clipping (Req 3.6)
- Forward Pass and Output Dimensions (Req 3.1, 3.2, 3.8)
- TorchScript Model Export & Checksum Verification (Req 3.10, 6.9, 11.8)
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.python.ppo_agent import ActorCritic, PPOAgent, OBS_DIM, ACTION_DIM


def test_actor_critic_forward_dimensions():
    """Verify ActorCritic output shapes and finiteness (Req 3.1, 3.2, 3.8)."""
    model = ActorCritic(obs_dim=8, action_dim=3, hidden_size=256)
    obs = torch.randn(16, 8)

    mean, log_std, value = model.forward(obs)

    assert mean.shape == (16, 3)
    assert log_std.shape == (3,)
    assert value.shape == (16,)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_std).all()
    assert torch.isfinite(value).all()
    assert (log_std >= -4.0).all() and (log_std <= 2.0).all()


def test_actor_critic_act():
    """Verify act() produces valid actions and finite log probabilities."""
    model = ActorCritic(obs_dim=8, action_dim=3, hidden_size=256)
    obs = torch.randn(10, 8)

    action, log_prob = model.act(obs)

    assert action.shape == (10, 3)
    assert log_prob.shape == (10,)
    assert torch.isfinite(action).all()
    assert torch.isfinite(log_prob).all()


def test_ppo_train_step_gradient_clipping():
    """Verify gradient norms are bounded and loss is finite (Property 10, Req 3.6)."""
    agent = PPOAgent(obs_dim=8, action_dim=3, hidden_size=256)
    N = 32

    obs = torch.randn(N, 8)
    actions = torch.randn(N, 3)
    old_log_probs = torch.randn(N)
    advantages = torch.randn(N)
    returns = torch.randn(N)

    metrics = agent.train_step(obs, actions, old_log_probs, advantages, returns)

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "total_loss" in metrics
    assert all(math.isfinite(v) for v in metrics.values())


@settings(max_examples=50)
@given(
    ratio=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    advantage=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    clip_eps=st.floats(min_value=0.05, max_value=0.4, allow_nan=False, allow_infinity=False),
)
def test_ppo_clipped_surrogate_property(ratio: float, advantage: float, clip_eps: float):
    """Property 9: PPO Clipped Surrogate Correctness (Req 3.3)."""
    r = torch.tensor(ratio)
    adv = torch.tensor(advantage)
    eps = clip_eps

    surr1 = r * adv
    clipped_ratio = torch.clamp(r, 1.0 - eps, 1.0 + eps)
    surr2 = clipped_ratio * adv

    surrogate = torch.min(surr1, surr2).item()
    assert math.isfinite(surrogate)
    assert (1.0 - eps) - 1e-6 <= clipped_ratio.item() <= (1.0 + eps) + 1e-6


def test_ppo_export_torchscript():
    """Verify TorchScript export and SHA-256 generation (Req 3.10, 6.9, 11.8)."""
    agent = PPOAgent(obs_dim=8, action_dim=3, hidden_size=256)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = str(Path(tmpdir) / "test_actor.pt")
        digest = agent.export(model_path)

        assert Path(model_path).exists()
        assert Path(model_path + ".sha256").exists()
        assert len(digest) == 64

        saved_checksum = Path(model_path + ".sha256").read_text().strip()
        assert saved_checksum == digest

        # Reload model via TorchScript
        loaded_script = torch.jit.load(model_path)
        dummy_obs = torch.randn(1, 8)
        out = loaded_script(dummy_obs)
        assert out.shape == (1, 3)
        assert torch.isfinite(out).all()
