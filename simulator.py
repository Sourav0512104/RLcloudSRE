"""Core cloud SRE simulator shared by the OpenEnv server and PPO trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    from .models import CloudSreRlAction
except ImportError:
    from models import CloudSreRlAction


@dataclass(slots=True)
class CloudSreRlConfig:
    episode_length: int = 96
    decision_interval_minutes: int = 15
    min_replicas: int = 1
    max_replicas: int = 24
    min_cpu_per_replica: float = 0.5
    max_cpu_per_replica: float = 4.0
    min_memory_gb: float = 1.0
    max_memory_gb: float = 16.0
    target_latency_ms: float = 180.0
    target_availability: float = 0.999
    initial_error_budget: float = 1.0


def action_from_vector(action: np.ndarray | list[float]) -> CloudSreRlAction:
    """Map a 6D continuous action vector into the OpenEnv action schema."""

    clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    return CloudSreRlAction(
        replica_delta=float(clipped[0]),
        cpu_delta=float(clipped[1]),
        memory_delta=float(clipped[2]),
        cache_delta=float(clipped[3]),
        autoheal_delta=float(clipped[4]),
        shedding_delta=float(clipped[5]),
    )


class CloudSreRlSimulator:
    """Stochastic simulator of a cloud service under SRE control."""

    feature_dim = 15

    def __init__(self, config: CloudSreRlConfig | None = None, seed: int | None = None):
        self.config = config or CloudSreRlConfig()
        self.rng = np.random.default_rng(seed)
        self.seed_value = seed
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed_value = seed

        self.step_index = 0
        self.replicas = 6
        self.cpu_per_replica = 1.5
        self.memory_per_replica_gb = 4.0
        self.cache_ratio = 0.35
        self.autoheal_level = 0.35
        self.load_shedding = 0.02
        self.queue_depth = 12.0
        self.error_budget_remaining = self.config.initial_error_budget
        self.incident_severity = 0.0
        self.last_action = np.zeros(6, dtype=np.float32)
        self.last_workload = 620.0
        return self._build_observation(reward=0.0, done=False, info={"reset": True})

    def set_operating_point(
        self,
        *,
        replicas: int | None = None,
        cpu_per_replica: float | None = None,
        memory_per_replica_gb: float | None = None,
        cache_ratio: float | None = None,
        autoheal_level: float | None = None,
        load_shedding: float | None = None,
        queue_depth: float | None = None,
        error_budget_remaining: float | None = None,
        incident_severity: float | None = None,
        workload_rpm: float | None = None,
    ) -> None:
        """Override the simulator state for task-specific scenarios."""

        if replicas is not None:
            self.replicas = int(np.clip(replicas, self.config.min_replicas, self.config.max_replicas))
        if cpu_per_replica is not None:
            self.cpu_per_replica = float(
                np.clip(
                    cpu_per_replica,
                    self.config.min_cpu_per_replica,
                    self.config.max_cpu_per_replica,
                )
            )
        if memory_per_replica_gb is not None:
            self.memory_per_replica_gb = float(
                np.clip(
                    memory_per_replica_gb,
                    self.config.min_memory_gb,
                    self.config.max_memory_gb,
                )
            )
        if cache_ratio is not None:
            self.cache_ratio = float(np.clip(cache_ratio, 0.0, 1.0))
        if autoheal_level is not None:
            self.autoheal_level = float(np.clip(autoheal_level, 0.0, 1.0))
        if load_shedding is not None:
            self.load_shedding = float(np.clip(load_shedding, 0.0, 0.35))
        if queue_depth is not None:
            self.queue_depth = float(max(queue_depth, 0.0))
        if error_budget_remaining is not None:
            self.error_budget_remaining = float(np.clip(error_budget_remaining, 0.0, 1.0))
        if incident_severity is not None:
            self.incident_severity = float(np.clip(incident_severity, 0.0, 1.0))
        if workload_rpm is not None:
            self.last_workload = float(max(workload_rpm, 0.0))

    def observe(self, *, info: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a telemetry snapshot without advancing the simulator."""

        return self._build_observation(
            reward=0.0,
            done=False,
            info=info or {"snapshot": True},
        )

    def get_snapshot(self) -> dict[str, Any]:
        """Capture mutable simulator state for branching comparisons."""

        return {
            "step_index": self.step_index,
            "replicas": self.replicas,
            "cpu_per_replica": self.cpu_per_replica,
            "memory_per_replica_gb": self.memory_per_replica_gb,
            "cache_ratio": self.cache_ratio,
            "autoheal_level": self.autoheal_level,
            "load_shedding": self.load_shedding,
            "queue_depth": self.queue_depth,
            "error_budget_remaining": self.error_budget_remaining,
            "incident_severity": self.incident_severity,
            "last_action": self.last_action.copy(),
            "last_workload": self.last_workload,
            "seed_value": self.seed_value,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore mutable simulator state from a snapshot."""

        self.step_index = snapshot["step_index"]
        self.replicas = snapshot["replicas"]
        self.cpu_per_replica = snapshot["cpu_per_replica"]
        self.memory_per_replica_gb = snapshot["memory_per_replica_gb"]
        self.cache_ratio = snapshot["cache_ratio"]
        self.autoheal_level = snapshot["autoheal_level"]
        self.load_shedding = snapshot["load_shedding"]
        self.queue_depth = snapshot["queue_depth"]
        self.error_budget_remaining = snapshot["error_budget_remaining"]
        self.incident_severity = snapshot["incident_severity"]
        self.last_action = snapshot["last_action"].copy()
        self.last_workload = snapshot["last_workload"]
        self.seed_value = snapshot["seed_value"]
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = snapshot["rng_state"]

    def step(self, action: CloudSreRlAction) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        action_vector = np.array(
            [
                action.replica_delta,
                action.cpu_delta,
                action.memory_delta,
                action.cache_delta,
                action.autoheal_delta,
                action.shedding_delta,
            ],
            dtype=np.float32,
        )
        self._apply_action(action_vector)
        self.step_index += 1

        workload_rpm, spike = self._sample_workload()
        incident = self._sample_incident()
        effective_capacity = self._effective_capacity(incident)
        served_rpm = workload_rpm * (1.0 - self.load_shedding)
        utilization = served_rpm / max(effective_capacity, 1.0)

        overload = max(served_rpm - effective_capacity, 0.0)
        drain = max(effective_capacity - served_rpm, 0.0)
        self.queue_depth = max(0.0, self.queue_depth * 0.58 + overload / 10.0 - drain / 22.0)

        latency_ms = (
            48.0
            + 95.0 * utilization**2
            + 0.50 * self.queue_depth
            + incident * 220.0
            + self.load_shedding * 35.0
        )
        error_rate = np.clip(
            0.0015
            + max(utilization - 0.84, 0.0) ** 2 * 0.22
            + incident * 0.11
            + self.queue_depth / 7000.0,
            0.0,
            0.65,
        )
        availability = float(np.clip(1.0 - error_rate - incident * 0.012, 0.70, 0.9999))
        burn_ratio = max(0.0, (self.config.target_availability - availability) / 0.001)
        self.error_budget_remaining = float(
            np.clip(self.error_budget_remaining - burn_ratio * 0.0065, 0.0, 1.0)
        )
        hourly_cost_usd = self._hourly_cost()

        latency_score = np.clip(
            1.0 - max(latency_ms - self.config.target_latency_ms, 0.0) / 260.0,
            -1.0,
            1.0,
        )
        availability_score = np.clip((availability - 0.97) / 0.03, -1.0, 1.0)
        efficiency_score = np.clip(1.0 - abs(utilization - 0.72) / 0.72, -1.0, 1.0)
        cost_score = np.clip(1.0 - hourly_cost_usd / 320.0, -1.0, 1.0)
        performance_score = float(
            np.clip(
                0.45 * latency_score + 0.55 * availability_score + 0.25 * efficiency_score,
                -1.0,
                1.0,
            )
        )

        action_churn = float(np.mean(np.abs(action_vector - self.last_action)))
        reward = float(
            1.4 * availability_score
            + 1.0 * latency_score
            + 0.5 * efficiency_score
            + 0.35 * cost_score
            - 0.8 * min(burn_ratio / 4.0, 2.0)
            - 0.25 * self.load_shedding
            - 0.10 * action_churn
        )
        done = self.step_index >= self.config.episode_length or self.error_budget_remaining <= 0.0

        info = {
            "workload_rpm": workload_rpm,
            "effective_capacity_rpm": effective_capacity,
            "spike": spike,
            "incident_sample": incident,
            "burn_ratio": burn_ratio,
        }
        observation = self._build_observation(
            reward=reward,
            done=done,
            info=info,
            incoming_rpm=workload_rpm,
            effective_rpm=served_rpm,
            utilization=utilization,
            latency_ms=latency_ms,
            error_rate=error_rate,
            availability=availability,
            slo_burn_rate=burn_ratio,
            hourly_cost_usd=hourly_cost_usd,
            performance_score=performance_score,
        )
        self.last_action = action_vector
        self.last_workload = workload_rpm
        return observation, reward, done, info

    def _apply_action(self, action: np.ndarray) -> None:
        replica_step = int(np.round(action[0] * 3.0))
        self.replicas = int(
            np.clip(self.replicas + replica_step, self.config.min_replicas, self.config.max_replicas)
        )
        self.cpu_per_replica = float(
            np.clip(
                self.cpu_per_replica + action[1] * 0.35,
                self.config.min_cpu_per_replica,
                self.config.max_cpu_per_replica,
            )
        )
        self.memory_per_replica_gb = float(
            np.clip(
                self.memory_per_replica_gb + action[2] * 0.60,
                self.config.min_memory_gb,
                self.config.max_memory_gb,
            )
        )
        self.cache_ratio = float(np.clip(self.cache_ratio + action[3] * 0.12, 0.0, 1.0))
        self.autoheal_level = float(np.clip(self.autoheal_level + action[4] * 0.10, 0.0, 1.0))
        self.load_shedding = float(np.clip(self.load_shedding + action[5] * 0.08, 0.0, 0.35))

    def _sample_workload(self) -> tuple[float, bool]:
        phase = 2.0 * np.pi * (self.step_index % self.config.episode_length) / self.config.episode_length
        diurnal = 180.0 * np.sin(phase - 0.8) + 80.0 * np.sin(2.0 * phase + 0.4)
        noise = float(self.rng.normal(0.0, 35.0))
        spike = bool(self.rng.random() < 0.08)
        spike_load = float(self.rng.uniform(120.0, 360.0)) if spike else 0.0
        workload = max(180.0, 620.0 + diurnal + noise + spike_load)
        return workload, spike

    def _sample_incident(self) -> float:
        decay = self.incident_severity * 0.62
        new_incident = 0.0
        chance = 0.03 + 0.05 * max(self.last_workload - 700.0, 0.0) / 400.0
        chance *= 1.0 - 0.45 * self.autoheal_level
        if self.rng.random() < chance:
            new_incident = float(self.rng.uniform(0.12, 0.70))
        self.incident_severity = float(np.clip(decay + new_incident, 0.0, 1.0))
        return self.incident_severity

    def _effective_capacity(self, incident: float) -> float:
        per_replica = self.cpu_per_replica * 175.0 + self.memory_per_replica_gb * 18.0
        cache_boost = 1.0 + 0.18 * self.cache_ratio
        mitigation = 1.0 - incident * (0.48 - 0.22 * self.autoheal_level)
        return self.replicas * per_replica * cache_boost * mitigation

    def _hourly_cost(self) -> float:
        return (
            self.replicas * (self.cpu_per_replica * 8.5 + self.memory_per_replica_gb * 1.7)
            + self.cache_ratio * 18.0
            + self.autoheal_level * 8.0
            + self.load_shedding * 6.0
        )

    def _build_observation(
        self,
        reward: float,
        done: bool,
        info: dict[str, Any],
        incoming_rpm: float | None = None,
        effective_rpm: float | None = None,
        utilization: float | None = None,
        latency_ms: float | None = None,
        error_rate: float | None = None,
        availability: float | None = None,
        slo_burn_rate: float | None = None,
        hourly_cost_usd: float | None = None,
        performance_score: float | None = None,
    ) -> dict[str, Any]:
        incoming = float(incoming_rpm if incoming_rpm is not None else self.last_workload)
        effective = float(effective_rpm if effective_rpm is not None else incoming * (1.0 - self.load_shedding))
        util = float(utilization if utilization is not None else 0.0)
        latency = float(latency_ms if latency_ms is not None else 110.0)
        err = float(error_rate if error_rate is not None else 0.002)
        avail = float(availability if availability is not None else 0.999)
        burn = float(slo_burn_rate if slo_burn_rate is not None else 0.0)
        cost = float(hourly_cost_usd if hourly_cost_usd is not None else self._hourly_cost())
        perf = float(performance_score if performance_score is not None else 0.0)
        feature_vector = [
            self.step_index / self.config.episode_length,
            incoming / 1200.0,
            effective / 1200.0,
            self.replicas / self.config.max_replicas,
            self.cpu_per_replica / self.config.max_cpu_per_replica,
            self.memory_per_replica_gb / self.config.max_memory_gb,
            self.cache_ratio,
            self.autoheal_level,
            self.load_shedding / 0.35,
            min(util, 2.0) / 2.0,
            min(self.queue_depth, 1200.0) / 1200.0,
            min(latency, 600.0) / 600.0,
            err,
            self.error_budget_remaining,
            self.incident_severity,
        ]
        metadata = {
            **info,
            "config": asdict(self.config),
        }
        return {
            "time_index": self.step_index,
            "incoming_rpm": incoming,
            "effective_rpm": effective,
            "replicas": self.replicas,
            "cpu_per_replica": self.cpu_per_replica,
            "memory_per_replica_gb": self.memory_per_replica_gb,
            "cache_ratio": self.cache_ratio,
            "autoheal_level": self.autoheal_level,
            "load_shedding": self.load_shedding,
            "utilization": util,
            "queue_depth": self.queue_depth,
            "latency_ms": latency,
            "error_rate": err,
            "availability": avail,
            "slo_burn_rate": burn,
            "error_budget_remaining": self.error_budget_remaining,
            "hourly_cost_usd": cost,
            "incident_severity": self.incident_severity,
            "performance_score": perf,
            "feature_vector": feature_vector,
            "done": done,
            "reward": reward,
            "metadata": metadata,
        }
