"""Gymnasium wrapper used for PPO training."""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium is required for PPO training. Install training extras with 'uv sync --extra train'."
    ) from exc

from .simulator import CloudSreRlConfig, CloudSreRlSimulator, action_from_vector


class CloudSreRlGymEnv(gym.Env[np.ndarray, np.ndarray]):
    """Local Gymnasium environment that mirrors the OpenEnv simulator."""

    metadata = {"render_modes": []}

    def __init__(self, config: CloudSreRlConfig | None = None, seed: int | None = None):
        super().__init__()
        self.simulator = CloudSreRlSimulator(config=config, seed=seed)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.simulator.feature_dim,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        observation = self.simulator.reset(seed=seed)
        return np.asarray(observation["feature_vector"], dtype=np.float32), observation["metadata"]

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.simulator.step(action_from_vector(action))
        return (
            np.asarray(observation["feature_vector"], dtype=np.float32),
            float(reward),
            bool(done),
            False,
            info,
        )


