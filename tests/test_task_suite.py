from cloud_sre_rl.simulator import CloudSreRlSimulator
from cloud_sre_rl.task_suite import CloudSreTaskRubric, grade_task, list_tasks


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
