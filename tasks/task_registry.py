"""Validator-facing task registry."""

from __future__ import annotations

from typing import Any

try:
    from cloud_sre_rl.task_suite import list_tasks
except ImportError:
    from task_suite import list_tasks


def load_tasks() -> list[dict[str, Any]]:
    """Return all hackathon tasks in a simple serializable format."""
    return [
        {
            "id": task.task_id,
            "name": task.title,
            "description": task.objective,
            "objective": task.objective,
            "difficulty": "medium",
            "horizon": task.horizon,
        }
        for task in list_tasks()
    ]
