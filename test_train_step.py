"""Smoke test for PPOAgent.train_step() — task 7.2"""
import math
import torch
from src.python.ppo_agent import PPOAgent

agent = PPOAgent(obs_dim=8, action_dim=3, hidden_size=256,
                 clip_eps=0.2, entropy_coef=0.01, device='cpu')

N = 32
torch.manual_seed(42)
obs           = torch.randn(N, 8)
actions       = torch.randn(N, 3)
old_log_probs = torch.randn(N)
advantages    = torch.randn(N)
returns       = torch.randn(N)

result = agent.train_step(obs, actions, old_log_probs, advantages, returns)
print("train_step output:", result)

assert set(result.keys()) == {"policy_loss", "value_loss", "entropy", "total_loss"}
assert all(isinstance(v, float) for v in result.values()), "All values must be floats"
assert all(math.isfinite(v) for v in result.values()), "All values must be finite"

# Second call to verify repeated steps work
result2 = agent.train_step(obs, actions, old_log_probs, advantages, returns)
print("Second step:", result2)
assert all(math.isfinite(v) for v in result2.values())

print("All assertions passed.")
