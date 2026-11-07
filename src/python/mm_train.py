"""
mm_train.py — PPO Market-Making Policy Training Script

Orchestrates the training pipeline:
1. Environment initialization (LobEnv)
2. PPO rollout collection into TrajectoryBuffer
3. GAE-λ advantage computation
4. PPO clipped surrogate gradient updates (mini-batch SGD)
5. Metric tracking: Sharpe ratio, PnL drawdown vs fixed-spread baseline
6. TorchScript model export (.pt + .pt.sha256)

Requirements: 3.1–3.11, 8.1–8.7, 16.1
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

from .mm_env import LobEnv
from .ppo_agent import PPOAgent
from .trajectory_buffer import TrajectoryBuffer, Transition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mm_train")


def calculate_sharpe_and_drawdown(
    pnl_series: list[float],
) -> tuple[float, float]:
    """Compute annualized Sharpe ratio and maximum drawdown from step PnLs."""
    if len(pnl_series) < 2:
        return 0.0, 0.0

    pnls = np.array(pnl_series, dtype=np.float64)
    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls)) + 1e-8
    # Annualize assuming ~252 trading days, 23400 ticks/day
    sharpe = (mean_pnl / std_pnl) * math.sqrt(252 * 500)

    # Max Drawdown
    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = running_max - cum_pnl
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    return sharpe, max_dd


def train(
    num_episodes: int = 10,
    rollout_steps: int = 2048,
    batch_size: int = 256,
    ppo_epochs: int = 10,
    lr: float = 3e-4,
    clip_eps: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    entropy_coef: float = 0.01,
    max_position: int = 500,
    model_output_path: str = "models/ppo_actor.pt",
    seed: int = 42,
) -> PPOAgent:
    """Run full PPO training loop on LobEnv."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("Initializing LobEnv with max_position=%d...", max_position)
    env = LobEnv(
        max_position=max_position,
        session_steps=rollout_steps,
        seed=seed,
    )

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=256,
        clip_eps=clip_eps,
        gamma=gamma,
        lam=lam,
        lr=lr,
        entropy_coef=entropy_coef,
    )

    buffer = TrajectoryBuffer(capacity=rollout_steps)

    logger.info(
        "Starting PPO training: %d episodes, rollout=%d, batch_size=%d, epochs=%d",
        num_episodes,
        rollout_steps,
        batch_size,
        ppo_epochs,
    )

    all_episode_pnls: list[float] = []

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset(seed=seed + episode)
        episode_pnl_history: list[float] = []

        # 1. Rollout Collection
        for step in range(rollout_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
            with torch.no_grad():
                mean, log_std, value = agent.model.forward(obs_tensor)
                dist = torch.distributions.Normal(mean, log_std.exp())
                action_t = dist.sample()
                log_prob_t = dist.log_prob(action_t).sum()

            action_np = action_t.cpu().numpy()
            log_prob = float(log_prob_t.item())
            val = float(value.item())

            # Clip action to environment bounds
            action_np[0] = max(0.0, action_np[0])
            action_np[1] = max(0.0, action_np[1])
            action_np[2] = min(1.0, max(0.01, action_np[2]))

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            episode_pnl_history.append(reward)

            buffer.push(
                Transition(
                    obs=obs,
                    action=action_np,
                    log_prob=log_prob,
                    value=val,
                    reward=reward,
                    done=done,
                )
            )

            obs = next_obs
            if done or buffer.full():
                break

        # 2. Advantage Computation (GAE-λ)
        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
            _, _, last_val = agent.model.forward(last_obs_tensor)
            last_value = 0.0 if done else float(last_val.item())

        buffer.compute_advantages(gamma=gamma, lam=lam, last_value=last_value)

        # 3. PPO Update Epochs
        obs_batch = buffer.obs_tensor().to(agent.device)
        act_batch = buffer.action_tensor().to(agent.device)
        adv_batch = buffer.advantage_tensor().to(agent.device)
        ret_batch = buffer.return_tensor().to(agent.device)

        with torch.no_grad():
            # Get old log probs
            old_mean, old_log_std, _ = agent.model.forward(obs_batch)
            old_dist = torch.distributions.Normal(old_mean, old_log_std.exp())
            old_log_probs = old_dist.log_prob(act_batch).sum(dim=-1)

        num_samples = len(buffer)
        for _epoch in range(ppo_epochs):
            indices = torch.randperm(num_samples)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_idx = indices[start:end]

                agent.train_step(
                    obs=obs_batch[batch_idx],
                    actions=act_batch[batch_idx],
                    old_log_probs=old_log_probs[batch_idx],
                    advantages=adv_batch[batch_idx],
                    returns=ret_batch[batch_idx],
                )

        buffer.clear()

        # Metrics
        total_pnl = float(np.sum(episode_pnl_history))
        sharpe, max_dd = calculate_sharpe_and_drawdown(episode_pnl_history)
        all_episode_pnls.append(total_pnl)

        logger.info(
            "Episode %03d/%03d | Total PnL: %+.2f | Sharpe: %.2f | Max DD: %.2f | Fills: %d",
            episode,
            num_episodes,
            total_pnl,
            sharpe,
            max_dd,
            env.fill_count,
        )

    # 4. Export TorchScript Model (Req 3.10)
    out_path = Path(model_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    digest = agent.export(str(out_path))
    logger.info("Successfully trained and exported model to %s (SHA-256: %s)", out_path, digest)

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Market Making Agent")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--rollout", type=int, default=2048, help="Rollout buffer steps")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=10, help="PPO update epochs per rollout")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output", type=str, default="models/ppo_actor_checkpoint.pt", help="Export path")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        rollout_steps=args.rollout,
        batch_size=args.batch_size,
        ppo_epochs=args.epochs,
        lr=args.lr,
        model_output_path=args.output,
    )
