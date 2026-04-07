"""OpenEnv wrapper around the cloud SRE simulator."""

from __future__ import annotations

import os
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CloudSreRlAction, CloudSreRlObservation
    from ..simulator import CloudSreRlSimulator
    from ..task_suite import list_tasks
except ImportError:
    from models import CloudSreRlAction, CloudSreRlObservation
    from simulator import CloudSreRlSimulator
    from task_suite import list_tasks


class CloudSreRlEnvironment(Environment):
    """Serve the cloud SRE simulator over the OpenEnv HTTP and WS interfaces."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._simulator = CloudSreRlSimulator()

    def reset(self) -> CloudSreRlObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        observation = self._simulator.reset()
        task_name = os.getenv("CLOUD_SRE_RL_TASK")
        if task_name:
            task_map = {task.task_id: task for task in list_tasks()}
            task = task_map.get(task_name)
            if task is not None:
                task.setup(self._simulator)
                observation = self._simulator.observe(info={"task_id": task_name, "reset": True})
        return CloudSreRlObservation(**observation)

    def step(self, action: CloudSreRlAction) -> CloudSreRlObservation:  # type: ignore[override]
        observation, _, done, _ = self._simulator.step(action)
        self._state.step_count += 1
        if done:
            observation["metadata"]["terminal_step"] = self._state.step_count
        return CloudSreRlObservation(**observation)

    @property
    def state(self) -> State:
        return self._state
