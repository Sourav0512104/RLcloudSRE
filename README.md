# Cloud SRE RL with OpenEnv

This project implements a complete reinforcement learning environment for a cloud SRE scenario and a PPO training stack around it.

The agent controls:
- replica count
- CPU and memory per replica
- cache aggressiveness
- incident auto-healing level
- load shedding pressure

The simulator scores the policy on:
- performance: latency, queue depth, utilization
- reliability: availability, error rate, SLO burn, incident impact
- cost: hourly infrastructure spend

## Layout

- `server/cloud_sre_rl_environment.py`: OpenEnv environment server wrapper
- `simulator.py`: shared cloud infrastructure simulator
- `gym_env.py`: Gymnasium wrapper used for PPO training
- `train_ppo.py`: PPO training entrypoint
- `evaluate.py`: deterministic policy evaluation
- `tests/test_environment.py`: environment smoke tests

## Setup

Install the base environment:

```bash
uv sync
```

Install PPO dependencies:

```bash
uv sync --extra train
```

## Train PPO

```bash
uv run --extra train train-ppo --total-timesteps 200000 --num-envs 4
```

Artifacts are written to `artifacts/ppo/`.

## Evaluate a saved policy

```bash
uv run --extra train evaluate-ppo artifacts/ppo/cloud_sre_ppo.zip --episodes 5
```

## Run the OpenEnv server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## OpenEnv usage

```python
from cloud_sre_rl import CloudSreRlAction, CloudSreRlEnv

with CloudSreRlEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    result = env.step(
        CloudSreRlAction(
            replica_delta=0.4,
            cpu_delta=0.2,
            memory_delta=0.1,
            cache_delta=0.3,
            autoheal_delta=0.5,
            shedding_delta=-0.1,
        )
    )
    print(result.reward, result.observation.latency_ms, result.observation.availability)
```

## Validate

```bash
openenv validate .
pytest
```

## Hugging Face submission notes

Define these environment variables in the Space settings:

```bash
API_BASE_URL=<llm api base url>
MODEL_NAME=<model id>
HF_TOKEN=<api key>
```

The required baseline script is `inference.py`. It:
- uses the OpenAI client for model decisions
- enumerates 3 hackathon tasks
- emits `[START]`, `[STEP]`, and `[END]` structured stdout logs
- computes normalized grader scores in the `0.0-1.0` range

## Reward design

The reward combines:
- low latency
- high availability
- efficient utilization
- low infrastructure cost
- low SLO burn
- low load shedding and action churn

Episodes run for 96 control intervals, which models one day at 15-minute decision resolution.
