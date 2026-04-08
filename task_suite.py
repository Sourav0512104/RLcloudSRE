"""Hackathon task definitions and deterministic graders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from openenv.core.rubrics import Rubric, RubricDict

try:
    from .models import CloudSreRlAction, CloudSreRlObservation
    from .simulator import CloudSreRlSimulator
except ImportError:
    from models import CloudSreRlAction, CloudSreRlObservation
    from simulator import CloudSreRlSimulator


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    title: str
    objective: str
    horizon: int
    setup: Callable[[CloudSreRlSimulator], None]


def _setup_spike_response(simulator: CloudSreRlSimulator) -> None:
    simulator.set_operating_point(
        replicas=4,
        cpu_per_replica=1.0,
        memory_per_replica_gb=3.0,
        cache_ratio=0.20,
        autoheal_level=0.25,
        load_shedding=0.0,
        queue_depth=180.0,
        error_budget_remaining=0.72,
        incident_severity=0.18,
        workload_rpm=1080.0,
    )


def _setup_cost_efficiency(simulator: CloudSreRlSimulator) -> None:
    simulator.set_operating_point(
        replicas=14,
        cpu_per_replica=2.6,
        memory_per_replica_gb=7.0,
        cache_ratio=0.65,
        autoheal_level=0.55,
        load_shedding=0.02,
        queue_depth=12.0,
        error_budget_remaining=0.94,
        incident_severity=0.0,
        workload_rpm=290.0,
    )


def _setup_incident_recovery(simulator: CloudSreRlSimulator) -> None:
    simulator.set_operating_point(
        replicas=5,
        cpu_per_replica=1.1,
        memory_per_replica_gb=3.2,
        cache_ratio=0.25,
        autoheal_level=0.15,
        load_shedding=0.06,
        queue_depth=260.0,
        error_budget_remaining=0.41,
        incident_severity=0.46,
        workload_rpm=760.0,
    )


TASKS: list[TaskSpec] = [
    TaskSpec(
        task_id="traffic_spike_response",
        title="Mitigate a sudden traffic spike",
        objective="Protect latency and availability during an abrupt workload surge.",
        horizon=10,
        setup=_setup_spike_response,
    ),
    TaskSpec(
        task_id="cost_efficiency",
        title="Reduce overprovisioning",
        objective="Lower cloud cost during low demand without hurting user experience.",
        horizon=10,
        setup=_setup_cost_efficiency,
    ),
    TaskSpec(
        task_id="incident_recovery",
        title="Recover from an active incident",
        objective="Restore reliability and preserve remaining error budget while an incident is live.",
        horizon=12,
        setup=_setup_incident_recovery,
    ),
]


def list_tasks() -> list[TaskSpec]:
    return TASKS


def grade_task(task_id: str, initial: dict, final: dict, reward_trace: list[float]) -> dict[str, float]:
    """Return normalized grader outputs for a task."""

    latency_score = max(0.0, min(1.0, 1.0 - max(final["latency_ms"] - 180.0, 0.0) / 240.0))
    availability_score = max(0.0, min(1.0, (final["availability"] - 0.97) / 0.03))
    budget_score = max(0.0, min(1.0, final["error_budget_remaining"]))
    cost_score = max(0.0, min(1.0, 1.0 - final["hourly_cost_usd"] / 260.0))
    queue_score = max(0.0, min(1.0, 1.0 - final["queue_depth"] / 300.0))
    incident_score = max(0.0, min(1.0, 1.0 - final["incident_severity"]))
    reward_score = max(0.0, min(1.0, (sum(reward_trace) / max(len(reward_trace), 1) + 1.5) / 3.5))

    if task_id == "traffic_spike_response":
        graders = {
            "latency_control": latency_score,
            "availability_protection": availability_score,
            "queue_reduction": queue_score,
            "reward_quality": reward_score,
        }
    elif task_id == "cost_efficiency":
        utilization_score = max(0.0, min(1.0, 1.0 - abs(final["utilization"] - 0.72) / 0.72))
        safety_score = max(
            0.0,
            min(1.0, 0.6 * availability_score + 0.4 * max(0.0, min(1.0, 1.0 - final["error_rate"] / 0.05))),
        )
        graders = {
            "cost_reduction": cost_score,
            "utilization_efficiency": utilization_score,
            "service_safety": safety_score,
            "reward_quality": reward_score,
        }
    else:
        recovery_gain = max(0.0, min(1.0, initial["incident_severity"] - final["incident_severity"] + 0.5))
        graders = {
            "incident_mitigation": incident_score if final["incident_severity"] < initial["incident_severity"] else recovery_gain,
            "availability_restoration": availability_score,
            "budget_preservation": budget_score,
            "reward_quality": reward_score,
        }

    graders["aggregate"] = sum(graders.values()) / len(graders)
    return graders


class TaskGraderRubric(Rubric):
    """Expose a task grader through OpenEnv's rubric interface."""

    def __init__(self, task_id: str):
        super().__init__()
        self.task_id = task_id
        self.initial_snapshot: dict | None = None
        self.reward_trace: list[float] = []

    def reset(self) -> None:
        self.initial_snapshot = None
        self.reward_trace = []

    def begin_episode(self, initial_snapshot: dict) -> None:
        self.initial_snapshot = initial_snapshot
        self.reward_trace = []

    def forward(self, action: CloudSreRlAction, observation: CloudSreRlObservation) -> float:
        if self.initial_snapshot is None:
            self.initial_snapshot = observation.model_dump()
        reward = float(observation.reward or 0.0)
        self.reward_trace.append(reward)
        graders = grade_task(
            self.task_id,
            self.initial_snapshot,
            observation.model_dump(),
            self.reward_trace,
        )
        return float(graders["aggregate"])


class CloudSreTaskRubric(Rubric):
    """Named rubric container exposing one grader per hackathon task."""

    def __init__(self):
        super().__init__()
        self.tasks = RubricDict({task.task_id: TaskGraderRubric(task.task_id) for task in TASKS})
        self.active_task_id = TASKS[0].task_id

    def reset_for_task(self, task_id: str, initial_snapshot: dict) -> None:
        self.reset()
        self.active_task_id = task_id
        task_rubric = self.tasks[task_id]
        if isinstance(task_rubric, TaskGraderRubric):
            task_rubric.begin_episode(initial_snapshot)

    def reset(self) -> None:
        for rubric in self.tasks.values():
            rubric.reset()

    def forward(self, action: CloudSreRlAction, observation: CloudSreRlObservation) -> float:
        return self.tasks[self.active_task_id](action, observation)
