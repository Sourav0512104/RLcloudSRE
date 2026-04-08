"""Root-level task definitions and graders for validator discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    from cloud_sre_rl.task_suite import grade_task, list_tasks as _list_task_specs
except ImportError:
    from task_suite import grade_task, list_tasks as _list_task_specs


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    id: str
    name: str
    difficulty: str
    description: str
    objective: str
    max_steps: int
    has_grader: bool = True


def _difficulty_for(task_id: str) -> str:
    mapping = {
        "traffic_spike_response": "easy",
        "cost_efficiency": "medium",
        "incident_recovery": "hard",
    }
    return mapping.get(task_id, "medium")


TASKS: list[TaskDefinition] = [
    TaskDefinition(
        id=task.task_id,
        name=task.title,
        difficulty=_difficulty_for(task.task_id),
        description=task.objective,
        objective=task.objective,
        max_steps=task.horizon,
    )
    for task in _list_task_specs()
]


class TaskGrader:
    """Simple deterministic grader wrapper per task."""

    def __init__(self, task_id: str):
        self.task_id = task_id

    def grade(
        self,
        *,
        initial: dict[str, Any] | None = None,
        final: dict[str, Any] | None = None,
        reward_trace: list[float] | None = None,
    ) -> float:
        initial = initial or {
            "latency_ms": 180.0,
            "availability": 0.97,
            "error_budget_remaining": 1.0,
            "hourly_cost_usd": 120.0,
            "queue_depth": 0.0,
            "incident_severity": 0.0,
            "utilization": 0.72,
            "error_rate": 0.0,
        }
        final = final or initial
        reward_trace = reward_trace or [0.0]
        result = grade_task(self.task_id, initial, final, reward_trace)
        return float(max(0.0, min(1.0, result["aggregate"])))


def list_tasks() -> list[TaskDefinition]:
    return TASKS


def list_task_dicts() -> list[dict[str, Any]]:
    return [asdict(task) for task in TASKS]


def get_task_by_id(task_id: str) -> TaskDefinition:
    for task in TASKS:
        if task.id == task_id:
            return task
    raise KeyError(task_id)
