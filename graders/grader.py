"""Validator-facing grader wrapper."""

from __future__ import annotations

from typing import Any

try:
    from cloud_sre_rl.task_suite import grade_task
except ImportError:
    from task_suite import grade_task


def grade(
    action: dict[str, Any] | None,
    task: dict[str, Any],
    *,
    initial: dict[str, Any] | None = None,
    final: dict[str, Any] | None = None,
    reward_trace: list[float] | None = None,
) -> float:
    """
    Return a normalized aggregate grade for a task in [0.0, 1.0].

    This wrapper is intentionally permissive about inputs so external validators
    can import it without needing the full environment runtime.
    """
    del action

    task_id = task.get("id") or task.get("task_id")
    if not task_id:
        return 0.0

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

    try:
        graders = grade_task(task_id, initial, final, reward_trace)
    except Exception:
        return 0.0
    return float(max(0.0, min(1.0, graders.get("aggregate", 0.0))))
