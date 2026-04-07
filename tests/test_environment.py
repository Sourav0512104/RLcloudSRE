from cloud_sre_rl.models import CloudSreRlAction
from cloud_sre_rl.server.cloud_sre_rl_environment import CloudSreRlEnvironment
from cloud_sre_rl.simulator import CloudSreRlSimulator


def test_reset_returns_feature_vector():
    simulator = CloudSreRlSimulator(seed=3)
    observation = simulator.reset(seed=3)
    assert len(observation["feature_vector"]) == simulator.feature_dim
    assert observation["error_budget_remaining"] == 1.0


def test_step_updates_telemetry():
    env = CloudSreRlEnvironment()
    env.reset()
    observation = env.step(
        CloudSreRlAction(
            replica_delta=0.6,
            cpu_delta=0.3,
            memory_delta=0.2,
            cache_delta=0.1,
            autoheal_delta=0.4,
            shedding_delta=-0.2,
        )
    )
    assert observation.time_index == 1
    assert observation.replicas >= 1
    assert observation.latency_ms > 0
    assert 0.0 <= observation.error_budget_remaining <= 1.0


def test_episode_terminates():
    simulator = CloudSreRlSimulator(seed=9)
    simulator.reset(seed=9)
    done = False
    steps = 0
    while not done:
        _, _, done, _ = simulator.step(CloudSreRlAction())
        steps += 1
        assert steps <= simulator.config.episode_length
    assert steps >= 1
