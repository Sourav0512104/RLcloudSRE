"""PPO training entrypoint for the cloud SRE simulator."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .simulator import CloudSreRlConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on the cloud SRE environment.")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ppo"))
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Training dependencies are missing. Run 'uv sync --extra train' first."
        ) from exc

    from .gym_env import CloudSreRlGymEnv

    config = CloudSreRlConfig()
    env = make_vec_env(
        CloudSreRlGymEnv,
        n_envs=args.num_envs,
        seed=args.seed,
        env_kwargs={"config": config},
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        tensorboard_log=str(args.output_dir / "tensorboard"),
        seed=args.seed,
    )
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "cloud_sre_ppo"
    model.save(model_path)
    metadata = {
        "algorithm": "PPO",
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "config": asdict(config),
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Saved PPO model to {model_path}.zip")


if __name__ == "__main__":
    main()
