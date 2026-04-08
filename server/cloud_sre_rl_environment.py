"""OpenEnv wrapper around the cloud SRE simulator."""

from __future__ import annotations

import os
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CloudSreRlAction, CloudSreRlObservation
    from ..simulator import CloudSreRlSimulator
    from ..task_suite import CloudSreTaskRubric, list_tasks
except ImportError:
    from models import CloudSreRlAction, CloudSreRlObservation
    from simulator import CloudSreRlSimulator
    from task_suite import CloudSreTaskRubric, list_tasks


class CloudSreRlEnvironment(Environment):
    """Serve the cloud SRE simulator over the OpenEnv HTTP and WS interfaces."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__(rubric=CloudSreTaskRubric())
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._simulator = CloudSreRlSimulator()
        self._active_task_id = list_tasks()[0].task_id

    def reset(self, task_name: str | None = None) -> CloudSreRlObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        observation = self._simulator.reset()
        task_name = task_name or os.getenv("CLOUD_SRE_RL_TASK")
        if task_name:
            task_map = {task.task_id: task for task in list_tasks()}
            task = task_map.get(task_name)
            if task is not None:
                task.setup(self._simulator)
                observation = self._simulator.observe(info={"task_id": task_name, "reset": True})
                self._active_task_id = task_name
        if self.rubric is not None and isinstance(self.rubric, CloudSreTaskRubric):
            self.rubric.reset_for_task(self._active_task_id, observation)
        return CloudSreRlObservation(**observation)

    def step(self, action: CloudSreRlAction) -> CloudSreRlObservation:  # type: ignore[override]
        observation, _, done, _ = self._simulator.step(action)
        self._state.step_count += 1
        if done:
            observation["metadata"]["terminal_step"] = self._state.step_count
        result = CloudSreRlObservation(**observation)
        if self.rubric is not None:
            result.reward = self._apply_rubric(action, result)
        return result

    @property
    def state(self) -> State:
        return self._state
