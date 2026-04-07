"""Evaluate a trained PPO policy on the cloud SRE simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO cloud SRE policy.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Training dependencies are missing. Run 'uv sync --extra train' first."
        ) from exc

    from .gym_env import CloudSreRlGymEnv

    env = CloudSreRlGymEnv(seed=args.seed)
    model = PPO.load(args.model_path)

    rewards = []
    budgets = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        terminated = False
        total_reward = 0.0
        last_info = {}
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, last_info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        budgets.append(env.simulator.error_budget_remaining)
        print(
            f"episode={episode} total_reward={total_reward:.3f} "
            f"remaining_budget={env.simulator.error_budget_remaining:.3f} "
            f"last_workload={last_info.get('workload_rpm', 0.0):.1f}"
        )

    print(
        "mean_reward="
        f"{np.mean(rewards):.3f} mean_remaining_budget={np.mean(budgets):.3f}"
    )


if __name__ == "__main__":
    main()
