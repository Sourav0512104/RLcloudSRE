"""Cloud SRE RL OpenEnv client."""

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CloudSreRlAction, CloudSreRlObservation


class CloudSreRlEnv(EnvClient[CloudSreRlAction, CloudSreRlObservation, State]):
    """Typed client for a running OpenEnv cloud SRE environment."""

    def _step_payload(self, action: CloudSreRlAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[CloudSreRlObservation]:
        observation_payload = dict(payload.get("observation", {}))
        observation_payload.setdefault("done", payload.get("done", False))
        observation_payload.setdefault("reward", payload.get("reward"))
        observation = CloudSreRlObservation(**observation_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
