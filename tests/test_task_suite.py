from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloud_sre_rl.simulator import CloudSreRlSimulator
from cloud_sre_rl.tasks import TaskGrader, list_task_dicts
from cloud_sre_rl.task_suite import CloudSreTaskRubric, grade_task, list_tasks
from graders.grader import grade
from tasks.task_registry import load_tasks


def test_task_suite_has_three_tasks():
    tasks = list_tasks()
    assert len(tasks) >= 3


def test_graders_are_normalized():
    simulator = CloudSreRlSimulator(seed=5)
    simulator.reset(seed=5)
    task = list_tasks()[0]
    task.setup(simulator)
    initial = simulator.observe()
    final, reward, _, _ = simulator.step(__import__("cloud_sre_rl").CloudSreRlAction())
    graders = grade_task(task.task_id, initial, final, [reward])
    for value in graders.values():
        assert 0.0 <= value <= 1.0


def test_task_rubric_exposes_three_named_graders():
    rubric = CloudSreTaskRubric()
    named = [name for name, _ in rubric.named_rubrics()]
    assert "tasks.traffic_spike_response" in named
    assert "tasks.cost_efficiency" in named
    assert "tasks.incident_recovery" in named


def test_validator_task_registry_has_three_tasks():
    tasks = load_tasks()
    assert len(tasks) >= 3
    assert {task["id"] for task in tasks} >= {
        "traffic_spike_response",
        "cost_efficiency",
        "incident_recovery",
    }


def test_validator_grader_returns_normalized_score():
    score = grade({}, {"id": "traffic_spike_response"})
    assert 0.0 <= score <= 1.0


def test_root_tasks_module_exposes_three_tasks_with_graders():
    tasks = list_task_dicts()
    assert len(tasks) >= 3
    assert all(task["has_grader"] for task in tasks)
    assert 0.0 <= TaskGrader("traffic_spike_response").grade() <= 1.0
