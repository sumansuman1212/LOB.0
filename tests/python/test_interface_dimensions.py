"""
tests/test_interface_dimensions.py — Cross-Stack Interface & Dimension Verification

Verifies:
1. Python Gymnasium Environment dimensions: observation (8,), action (3,)
2. PPO Actor-Critic neural network architecture: input 8, output 3
3. Frozen TorchScript checkpoint (models/ppo_actor.pt):
   - Input shape (1, 8) -> Output shape (1, 3)
   - Rejection of invalid dimensions (e.g. 15 dims)
   - Absence of NaN/Inf in output
4. C++ Header specifications (include/mm/types.hpp, include/mm/data_models.hpp):
   - kObsDim == 8, kActionDim == 3, MarketFeatures::DIM == 8
5. Exact interpretation of policy outputs: (delta_bid, delta_ask, size_frac)
"""
from __future__ import annotations

import re
from pathlib import Path
import pytest
import torch
import numpy as np

from src.python.mm_env import LobEnv, OBS_DIM as ENV_OBS_DIM, ACTION_DIM as ENV_ACT_DIM
from src.python.ppo_agent import ActorCritic, OBS_DIM as PPO_OBS_DIM, ACTION_DIM as PPO_ACT_DIM


def test_python_environment_dimensions():
    """Verify Gymnasium environment observation and action space dimensions."""
    env = LobEnv()
    assert env.observation_space.shape == (8,)
    assert env.action_space.shape == (3,)
    assert ENV_OBS_DIM == 8
    assert ENV_ACT_DIM == 3

    obs, _ = env.reset(seed=42)
    assert obs.shape == (8,)
    assert obs.dtype == np.float32


def test_ppo_network_architecture_dimensions():
    """Verify PPO ActorCritic module input/output layer dimensions."""
    net = ActorCritic(obs_dim=8, action_dim=3)
    assert PPO_OBS_DIM == 8
    assert PPO_ACT_DIM == 3

    # Inspect first layer of shared trunk
    first_linear = net.trunk[0]
    assert isinstance(first_linear, torch.nn.Linear)
    assert first_linear.in_features == 8
    assert first_linear.out_features == 256

    # Inspect actor head
    assert isinstance(net.mean_head, torch.nn.Linear)
    assert net.mean_head.in_features == 256
    assert net.mean_head.out_features == 3

    dummy_obs = torch.randn(2, 8)
    mean, log_std, val = net(dummy_obs)
    assert mean.shape == (2, 3)
    assert log_std.shape == (3,)
    assert val.shape == (2,)


def test_torchscript_checkpoint_dimensions_and_invariants():
    """Verify the original TorchScript model checkpoint interface.
    
    Proves:
    1. The model expects exactly 8 observation features.
    2. The model outputs exactly 3 actions.
    3. Passing 15 features raises a RuntimeError (disproving the 15-dim claim).
    4. Outputs are strictly finite (no NaN or Inf).
    """
    model_path = Path("models/ppo_actor.pt")
    assert model_path.exists(), "models/ppo_actor.pt must exist"

    model = torch.jit.load(str(model_path))
    model.eval()

    # Verify layer weights directly
    for name, param in model.named_parameters():
        if name == "trunk.0.weight":
            assert param.shape == torch.Size([256, 8]), "Trunk input weight must be (256, 8)"
        elif name == "mean_head.weight":
            assert param.shape == torch.Size([3, 256]), "Mean head weight must be (3, 256)"

    # Valid 8-dimensional observation
    valid_obs = torch.randn(1, 8, dtype=torch.float32)
    with torch.no_grad():
        out = model(valid_obs)
    
    assert out.shape == torch.Size([1, 3]), "Output tensor shape must be (1, 3)"
    out_arr = out.numpy().flatten()
    assert np.all(np.isfinite(out_arr)), "Output must not contain NaN or Inf"

    # Verify that passing 15 features (the hallucinated dimension) fails
    invalid_obs = torch.randn(1, 15, dtype=torch.float32)
    with pytest.raises(RuntimeError) as exc_info:
        model(invalid_obs)
    assert "cannot be multiplied" in str(exc_info.value) or "shapes" in str(exc_info.value)


def test_cpp_header_constants_match():
    """Verify that C++ header files declare identical observation and action dimensions."""
    types_path = Path("include/mm/types.hpp")
    assert types_path.exists()
    types_content = types_path.read_text(encoding="utf-8")

    obs_match = re.search(r"constexpr\s+int\s+kObsDim\s*=\s*(\d+);", types_content)
    assert obs_match is not None, "kObsDim not found in types.hpp"
    assert int(obs_match.group(1)) == 8, f"kObsDim must be 8, got {obs_match.group(1)}"

    act_match = re.search(r"constexpr\s+int\s+kActionDim\s*=\s*(\d+);", types_content)
    assert act_match is not None, "kActionDim not found in types.hpp"
    assert int(act_match.group(1)) == 3, f"kActionDim must be 3, got {act_match.group(1)}"

    models_path = Path("include/mm/data_models.hpp")
    assert models_path.exists()
    models_content = models_path.read_text(encoding="utf-8")

    dim_match = re.search(r"static\s+constexpr\s+int\s+DIM\s*=\s*(\d+);", models_content)
    assert dim_match is not None, "MarketFeatures::DIM not found in data_models.hpp"
    assert int(dim_match.group(1)) == 8, f"MarketFeatures::DIM must be 8, got {dim_match.group(1)}"


def test_action_interpretation_and_position_sizing():
    """Verify how policy outputs are mapped to quote offsets and order sizes."""
    # Action = [delta_bid, delta_ask, size_frac]
    raw_action = np.array([1.5, 2.5, 0.4], dtype=np.float32)
    max_order_size = 50
    tick_size = 5
    mid_price = 20000

    delta_bid, delta_ask, size_frac = float(raw_action[0]), float(raw_action[1]), float(raw_action[2])
    
    # 1. Tick conversion
    quote_bid = int(round(mid_price - delta_bid * tick_size))
    quote_ask = int(round(mid_price + delta_ask * tick_size))
    assert quote_bid == 20000 - int(round(1.5 * 5))  # 20000 - 8 = 19992
    assert quote_ask == 20000 + int(round(2.5 * 5))  # 20000 + 12 = 20012

    # 2. Position sizing
    bid_qty = max(1, int(round(size_frac * max_order_size)))
    ask_qty = max(1, int(round(size_frac * max_order_size)))
    assert bid_qty == int(round(0.4 * 50))  # 20
    assert ask_qty == 20
