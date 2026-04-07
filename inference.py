"""
Inference script for the Cloud SRE RL hackathon submission.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from cloud_sre_rl import CloudSreRlAction, CloudSreRlEnv
from cloud_sre_rl.task_suite import grade_task, list_tasks

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "openai/gpt-4.1-mini"
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
TASK_NAME = os.getenv("CLOUD_SRE_RL_TASK", "traffic_spike_response")
BENCHMARK = os.getenv("CLOUD_SRE_RL_BENCHMARK", "cloud_sre_rl")
MAX_STEPS = int(os.getenv("CLOUD_SRE_RL_MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("CLOUD_SRE_RL_TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("CLOUD_SRE_RL_MAX_TOKENS", "220"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("CLOUD_SRE_RL_SUCCESS_THRESHOLD", "0.6"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a cloud service as an SRE policy.
    Choose one action for the current telemetry snapshot.
    Reply by calling the tool with six values in [-1, 1].
    Prefer preserving availability and error budget, then latency, then cost.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def require_api_key() -> str:
    if not API_KEY:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")
    return API_KEY


def get_task_spec():
    tasks = {task.task_id: task for task in list_tasks()}
    if TASK_NAME not in tasks:
        available = ", ".join(sorted(tasks))
        raise RuntimeError(f"Unknown task '{TASK_NAME}'. Available tasks: {available}")
    return tasks[TASK_NAME]


def build_user_prompt(task_spec, step: int, telemetry: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Task: {task_spec.task_id}
        Objective: {task_spec.objective}
        Step: {step}
        Telemetry:
        {json.dumps(telemetry, separators=(",", ":"))}
        Previous steps:
        {history_block}
        Choose the next SRE control action.
        """
    ).strip()


def get_model_action(client: OpenAI, task_spec, step: int, telemetry: dict, history: List[str]) -> CloudSreRlAction:
    tool_schema = {
        "type": "function",
        "function": {
            "name": "apply_sre_action",
            "description": "Select the next cloud SRE control action.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "replica_delta": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                    "cpu_delta": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                    "memory_delta": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                    "cache_delta": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                    "autoheal_delta": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                    "shedding_delta": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                },
                "required": [
                    "replica_delta",
                    "cpu_delta",
                    "memory_delta",
                    "cache_delta",
                    "autoheal_delta",
                    "shedding_delta",
                ],
            },
        },
    }

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(task_spec, step, telemetry, history)},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        tool_choice={"type": "function", "function": {"name": "apply_sre_action"}},
        tools=[tool_schema],
        stream=False,
    )
    tool_call = completion.choices[0].message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    return CloudSreRlAction(**arguments)


def connect_env() -> CloudSreRlEnv:
    if IMAGE_NAME:
        return CloudSreRlEnv.from_docker_image(IMAGE_NAME)
    if ENV_BASE_URL:
        return CloudSreRlEnv(base_url=ENV_BASE_URL)
    return CloudSreRlEnv(base_url="http://127.0.0.1:8000")


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=require_api_key())
    task_spec = get_task_spec()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env: Optional[CloudSreRlEnv] = None
    initial_snapshot = None
    final_snapshot = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = connect_env()
        result = env.reset()
        initial_snapshot = result.observation.model_dump()
        final_snapshot = initial_snapshot

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            observation = result.observation
            telemetry = {
                "time_index": observation.time_index,
                "incoming_rpm": round(observation.incoming_rpm, 3),
                "replicas": observation.replicas,
                "cpu_per_replica": round(observation.cpu_per_replica, 3),
                "memory_per_replica_gb": round(observation.memory_per_replica_gb, 3),
                "cache_ratio": round(observation.cache_ratio, 3),
                "autoheal_level": round(observation.autoheal_level, 3),
                "load_shedding": round(observation.load_shedding, 3),
                "utilization": round(observation.utilization, 3),
                "queue_depth": round(observation.queue_depth, 3),
                "latency_ms": round(observation.latency_ms, 3),
                "error_rate": round(observation.error_rate, 6),
                "availability": round(observation.availability, 6),
                "error_budget_remaining": round(observation.error_budget_remaining, 6),
                "hourly_cost_usd": round(observation.hourly_cost_usd, 3),
                "incident_severity": round(observation.incident_severity, 3),
            }

            action = get_model_action(client, task_spec, step, telemetry, history)
            result = env.step(action)
            obs = result.observation

            reward = float(result.reward or 0.0)
            done = bool(result.done)
            error = None
            rewards.append(reward)
            steps_taken = step
            final_snapshot = obs.model_dump()

            action_str = json.dumps(action.model_dump(), separators=(",", ":"))
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"step={step} latency={obs.latency_ms:.2f} availability={obs.availability:.4f} reward={reward:.2f}"
            )
            if done:
                break

        if initial_snapshot is not None and final_snapshot is not None:
            graders = grade_task(task_spec.task_id, initial_snapshot, final_snapshot, rewards)
            score = min(max(float(graders["aggregate"]), 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
