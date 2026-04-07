"""Custom Gradio UI for the cloud SRE environment."""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any

import gradio as gr
from openai import OpenAI

try:
    from ..models import CloudSreRlAction
    from ..simulator import CloudSreRlSimulator
    from ..task_suite import list_tasks
except ImportError:
    from models import CloudSreRlAction
    from simulator import CloudSreRlSimulator
    from task_suite import list_tasks


TASK_MAP = {task.task_id: task for task in list_tasks()}

CUSTOM_UI_CSS = """
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(17, 138, 178, 0.18), transparent 28%),
    radial-gradient(circle at top right, rgba(6, 214, 160, 0.15), transparent 24%),
    linear-gradient(180deg, #f4fbff 0%, #edf7f2 100%);
}
.hero-shell {
  padding: 20px 24px;
  border-radius: 24px;
  background: linear-gradient(135deg, #12344d 0%, #1f6f78 55%, #3ba99c 100%);
  color: #ffffff;
  box-shadow: 0 18px 40px rgba(18, 52, 77, 0.18);
  margin-bottom: 12px;
}
.hero-shell h1, .hero-shell p {
  color: #ffffff !important;
}
.info-shell {
  padding: 18px 20px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.78);
  border: 1px solid rgba(18, 52, 77, 0.08);
  box-shadow: 0 8px 24px rgba(18, 52, 77, 0.08);
}
.section-shell {
  padding: 18px;
  border-radius: 22px;
  background: rgba(255, 255, 255, 0.88);
  border: 1px solid rgba(18, 52, 77, 0.08);
  box-shadow: 0 10px 26px rgba(18, 52, 77, 0.08);
}
.metric-card {
  background: linear-gradient(180deg, #ffffff 0%, #f7fbfd 100%);
  border: 1px solid rgba(18, 52, 77, 0.08);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 8px 20px rgba(18, 52, 77, 0.08);
  min-height: 94px;
}
.metric-label {
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #587387;
  margin-bottom: 6px;
}
.metric-value {
  font-size: 28px;
  font-weight: 700;
  color: #12344d;
}
.metric-hint {
  font-size: 12px;
  color: #6d8798;
  margin-top: 6px;
}
.action-note {
  padding: 12px 14px;
  border-left: 4px solid #118ab2;
  background: rgba(17, 138, 178, 0.08);
  border-radius: 12px;
}
"""


def _format_metrics(obs: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        f"{obs.get('latency_ms', 0.0):.2f} ms",
        f"{obs.get('availability', 0.0):.4f}",
        f"${obs.get('hourly_cost_usd', 0.0):.2f}/hr",
        f"{obs.get('error_budget_remaining', 0.0):.4f}",
        f"{obs.get('queue_depth', 0.0):.2f}",
        f"{obs.get('incident_severity', 0.0):.2f}",
    )


def _metric_html(label: str, value: str, hint: str) -> str:
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"<div class='metric-hint'>{hint}</div>"
        f"</div>"
    )


def _metric_panel(obs: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    latency, availability, cost, budget, queue, incident = _format_metrics(obs)
    return (
        _metric_html("Latency", latency, "Request response speed. Lower is better."),
        _metric_html("Availability", availability, "How often the service stays up. Higher is better."),
        _metric_html("Hourly Cost", cost, "Estimated infra spend for the current operating point."),
        _metric_html("Error Budget", budget, "Reliability headroom before SLO trouble."),
        _metric_html("Queue Depth", queue, "How much work is backing up. Lower is better."),
        _metric_html("Incident Severity", incident, "Operational trouble level. Lower is better."),
    )


def _telemetry_rows(obs: dict[str, Any], reward: float | None = None) -> list[list[str]]:
    return [
        ["Incoming RPM", f"{obs.get('incoming_rpm', 0.0):.2f}"],
        ["Effective RPM", f"{obs.get('effective_rpm', 0.0):.2f}"],
        ["Replicas", str(obs.get("replicas", 0))],
        ["CPU / Replica", f"{obs.get('cpu_per_replica', 0.0):.2f}"],
        ["Memory / Replica (GB)", f"{obs.get('memory_per_replica_gb', 0.0):.2f}"],
        ["Cache Ratio", f"{obs.get('cache_ratio', 0.0):.2f}"],
        ["Autoheal Level", f"{obs.get('autoheal_level', 0.0):.2f}"],
        ["Load Shedding", f"{obs.get('load_shedding', 0.0):.2f}"],
        ["Utilization", f"{obs.get('utilization', 0.0):.2f}"],
        ["Error Rate", f"{obs.get('error_rate', 0.0):.4f}"],
        ["SLO Burn Rate", f"{obs.get('slo_burn_rate', 0.0):.2f}"],
        ["Reward", f"{(reward if reward is not None else obs.get('reward', 0.0)):.3f}"],
    ]


def _reward_history(logs: list[dict[str, Any]]) -> list[list[float]]:
    if not logs:
        return []
    return [[entry.get("step_count", i + 1), entry.get("reward", 0.0) or 0.0] for i, entry in enumerate(logs)]


def _action_history(logs: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for entry in logs[-10:]:
        action = entry.get("action", {})
        rows.append(
            [
                entry.get("step_count", 0),
                action.get("replica_delta", 0.0),
                action.get("cpu_delta", 0.0),
                action.get("memory_delta", 0.0),
                action.get("cache_delta", 0.0),
                action.get("autoheal_delta", 0.0),
                action.get("shedding_delta", 0.0),
                entry.get("reward", 0.0),
            ]
        )
    return rows


def _task_description(task_name: str) -> str:
    task = TASK_MAP[task_name]
    return textwrap.dedent(
        f"""
        ### Active Scenario: `{task.task_id}`
        **Goal:** {task.objective}

        **What this means in plain English**
        - `traffic_spike_response`: traffic suddenly jumps and you need to keep the service fast and stable.
        - `cost_efficiency`: demand is low and you should reduce cloud spend without hurting users.
        - `incident_recovery`: the system is already having trouble and you must protect reliability.
        """
    ).strip()


def _hero_banner() -> str:
    return """
    <div class="hero-shell">
      <h1>Cloud SRE Mission Control</h1>
      <p>
        This dashboard turns the environment into an operator training console.
        Explore incident response, cost tuning, and traffic spike management with presets,
        manual controls, or an LLM suggestion.
      </p>
    </div>
    """


def _beginner_guide() -> str:
    return textwrap.dedent(
        """
        ## What this environment represents
        This is a simulated cloud service that an SRE team is operating. Each step is one control decision.

        ## What the action knobs mean
        - **Replica Delta**: adds or removes service instances. More replicas usually improve stability, but cost more.
        - **CPU Delta**: changes CPU allocated to each replica. More CPU can reduce latency under load.
        - **Memory Delta**: changes memory allocated to each replica. More memory can help caching and stability.
        - **Cache Delta**: changes how aggressively the system uses cache. More cache can improve speed, but also costs something.
        - **Autoheal Delta**: changes how strongly automated recovery is applied. Higher values help when incidents appear.
        - **Shedding Delta**: changes how much traffic the system intentionally drops to protect itself during overload.

        ## What the metrics mean
        - **Latency**: how long requests take. Lower is better.
        - **Availability**: how often the service is up. Higher is better.
        - **Hourly Cost**: infrastructure spend. Lower is better if performance remains healthy.
        - **Error Budget**: how much reliability headroom remains before SLO trouble. Higher is better.
        - **Queue Depth**: how much work is backing up. Lower is better.
        - **Incident Severity**: how serious the current operational problem is. Lower is better.

        ## How to explore
        1. Choose a scenario with the task selector.
        2. Click **Reset Scenario**.
        3. Try either a preset, manual slider values, or **Ask LLM Policy**.
        4. Click **Apply Action**.
        5. Compare your action to the zero-action baseline panel.
        """
    ).strip()


def _explain(obs: dict[str, Any]) -> str:
    latency = obs.get("latency_ms", 0.0)
    availability = obs.get("availability", 0.0)
    cost = obs.get("hourly_cost_usd", 0.0)
    queue = obs.get("queue_depth", 0.0)
    incident = obs.get("incident_severity", 0.0)

    verdicts = []
    if latency < 100 and availability > 0.998:
        verdicts.append("The service is stable and fast.")
    elif latency > 180 or queue > 50:
        verdicts.append("The service is under pressure; extra capacity or mitigation may help.")
    else:
        verdicts.append("The service is operating in a moderate state.")

    if cost > 170:
        verdicts.append("Cost is elevated, so scaling up further should have a clear benefit.")
    else:
        verdicts.append("Cost is still in a reasonable range.")

    if incident > 0.2:
        verdicts.append("An incident is active, so auto-healing and safe capacity matter more.")
    else:
        verdicts.append("There is no major incident active right now.")

    return " ".join(verdicts)


def _preset_action(name: str) -> tuple[float, float, float, float, float, float]:
    presets = {
        "stabilize": (0.4, 0.2, 0.2, 0.2, 0.3, -0.1),
        "cut_cost": (-0.4, -0.2, -0.2, -0.1, 0.0, 0.0),
        "incident": (0.5, 0.3, 0.2, 0.1, 0.6, 0.05),
    }
    return presets[name]


def _compare_to_baseline(snapshot: dict[str, Any], action: CloudSreRlAction) -> str:
    simulator = CloudSreRlSimulator()
    simulator.load_snapshot(snapshot)
    _, baseline_reward, _, _ = simulator.step(CloudSreRlAction())
    baseline_obs = simulator.observe(info={"comparison": "baseline"})

    simulator = CloudSreRlSimulator()
    simulator.load_snapshot(snapshot)
    acted_obs, acted_reward, _, _ = simulator.step(action)

    delta_reward = acted_reward - baseline_reward
    verdict = "better" if delta_reward > 0 else "worse" if delta_reward < 0 else "equal"
    return textwrap.dedent(
        f"""
        ### Action vs Zero-Action Baseline
        Your action reward: **{acted_reward:.3f}**
        Zero-action reward: **{baseline_reward:.3f}**
        Verdict: your action was **{verdict}** by **{abs(delta_reward):.3f}** reward.

        **Why**
        - Latency: {acted_obs['latency_ms']:.2f} ms vs baseline {baseline_obs['latency_ms']:.2f} ms
        - Availability: {acted_obs['availability']:.4f} vs baseline {baseline_obs['availability']:.4f}
        - Hourly cost: ${acted_obs['hourly_cost_usd']:.2f} vs baseline ${baseline_obs['hourly_cost_usd']:.2f}
        """
    ).strip()


def _llm_action(task_name: str, observation: dict[str, Any]) -> tuple[CloudSreRlAction, str]:
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("HF_TOKEN is not set for the LLM policy button.")

    client = OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
        api_key=api_key,
    )
    model_name = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
    task = TASK_MAP[task_name]

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

    system_prompt = (
        "You are an SRE control policy. Choose one action in [-1, 1] for each knob. "
        "Optimize reliability first, then latency, then cost."
    )
    user_prompt = json.dumps(
        {
            "task": {"id": task.task_id, "objective": task.objective},
            "observation": observation,
        },
        separators=(",", ":"),
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=220,
        tool_choice={"type": "function", "function": {"name": "apply_sre_action"}},
        tools=[tool_schema],
        stream=False,
    )
    tool_call = completion.choices[0].message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    action = CloudSreRlAction(**arguments)
    explanation = (
        f"LLM policy suggestion from `{model_name}`. Review the proposed knob changes, "
        "then click Apply Action to see whether they beat the baseline."
    )
    return action, explanation


def build_custom_gradio_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    """Build a custom operator dashboard mounted as the Custom tab."""

    async def do_reset(task_name: str):
        data = await web_manager.reset_environment({"task_name": task_name})
        obs = data["observation"]
        logs = web_manager.episode_state.model_dump().get("action_logs", [])
        latency, availability, cost, budget, queue, incident = _metric_panel(obs)
        return (
            latency,
            availability,
            cost,
            budget,
            queue,
            incident,
            _telemetry_rows(obs, reward=float(data.get("reward") or 0.0)),
            _reward_history(logs),
            _action_history(logs),
            _explain(obs),
            _task_description(task_name),
            "Reset complete. Start with a preset, move the sliders yourself, or ask the LLM for a suggestion.",
            json.dumps(data, indent=2),
            "Reset a scenario before comparing actions.",
        )

    async def do_step(task_name: str, replica, cpu, memory, cache, autoheal, shedding):
        snapshot = None
        if hasattr(web_manager.env, "_simulator"):
            snapshot = web_manager.env._simulator.get_snapshot()

        action = CloudSreRlAction(
            replica_delta=replica,
            cpu_delta=cpu,
            memory_delta=memory,
            cache_delta=cache,
            autoheal_delta=autoheal,
            shedding_delta=shedding,
        )
        data = await web_manager.step_environment(action.model_dump())
        obs = data["observation"]
        logs = web_manager.episode_state.model_dump().get("action_logs", [])
        latency, availability, cost, budget, queue, incident = _metric_panel(obs)
        comparison = (
            _compare_to_baseline(snapshot, action)
            if snapshot is not None
            else "Baseline comparison is unavailable."
        )
        return (
            latency,
            availability,
            cost,
            budget,
            queue,
            incident,
            _telemetry_rows(obs, reward=float(data.get("reward") or 0.0)),
            _reward_history(logs),
            _action_history(logs),
            _explain(obs),
            _task_description(task_name),
            f"Applied action. Reward={data.get('reward', 0.0):.3f}",
            json.dumps(data, indent=2),
            comparison,
        )

    def apply_preset(name: str):
        return _preset_action(name)

    def suggest_with_llm(task_name: str):
        current = web_manager.episode_state.current_observation
        if not current:
            raise gr.Error("Reset the scenario first so the LLM has telemetry to analyze.")
        try:
            action, explanation = _llm_action(task_name, current)
        except Exception as exc:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, f"LLM policy unavailable: {exc}")
        return (
            action.replica_delta,
            action.cpu_delta,
            action.memory_delta,
            action.cache_delta,
            action.autoheal_delta,
            action.shedding_delta,
            explanation,
        )

    with gr.Blocks(css=CUSTOM_UI_CSS) as demo:
        gr.HTML(_hero_banner())
        with gr.Accordion("What Am I Looking At?", open=True, elem_classes=["info-shell"]):
            gr.Markdown(_beginner_guide())

        with gr.Row():
            task_name = gr.Dropdown(
                choices=list(TASK_MAP.keys()),
                value="traffic_spike_response",
                label="Scenario / Task",
            )
            reset_btn = gr.Button("Reset Scenario", variant="secondary")
            llm_btn = gr.Button("Ask LLM Policy")
            step_btn = gr.Button("Apply Action", variant="primary")

        task_info = gr.Markdown(_task_description("traffic_spike_response"), elem_classes=["info-shell"])

        with gr.Row():
            latency = gr.HTML(_metric_html("Latency", "--", "Request response speed. Lower is better."))
            availability = gr.HTML(_metric_html("Availability", "--", "How often the service stays up."))
            cost = gr.HTML(_metric_html("Hourly Cost", "--", "Current infra spend estimate."))
            budget = gr.HTML(_metric_html("Error Budget", "--", "Remaining SLO reliability headroom."))
            queue = gr.HTML(_metric_html("Queue Depth", "--", "Backlogged work waiting to be processed."))
            incident = gr.HTML(_metric_html("Incident Severity", "--", "Operational trouble level."))

        with gr.Row():
            with gr.Column(scale=2, elem_classes=["section-shell"]):
                gr.Markdown("## Control Knobs")
                gr.Markdown(
                    "<div class='action-note'>Start by picking a scenario, then either use a preset, ask the LLM for a suggestion, or move the sliders yourself.</div>"
                )
                with gr.Row():
                    replica = gr.Slider(-1, 1, value=0, step=0.1, label="Replica Delta")
                    cpu = gr.Slider(-1, 1, value=0, step=0.1, label="CPU Delta")
                    memory = gr.Slider(-1, 1, value=0, step=0.1, label="Memory Delta")
                with gr.Row():
                    cache = gr.Slider(-1, 1, value=0, step=0.1, label="Cache Delta")
                    autoheal = gr.Slider(-1, 1, value=0, step=0.1, label="Autoheal Delta")
                    shedding = gr.Slider(-1, 1, value=0, step=0.1, label="Shedding Delta")
                with gr.Row():
                    stabilize_btn = gr.Button("Stabilize Service")
                    cost_btn = gr.Button("Cut Cost")
                    incident_btn = gr.Button("Incident Recovery")
                status = gr.Textbox(label="Status", interactive=False)
                explanation = gr.Markdown("Reset the environment to begin.")
                baseline_compare = gr.Markdown("Reset a scenario before comparing actions.")

            with gr.Column(scale=1, elem_classes=["section-shell"]):
                telemetry_table = gr.Dataframe(
                    headers=["Metric", "Value"],
                    value=[],
                    label="Telemetry Snapshot",
                    interactive=False,
                    wrap=True,
                )

        with gr.Row(equal_height=True):
            reward_plot = gr.Dataframe(
                headers=["Step", "Reward"],
                value=[],
                label="Reward Over Time",
                interactive=False,
                wrap=True,
                elem_classes=["section-shell"],
            )
            action_table = gr.Dataframe(
                headers=[
                    "Step",
                    "Replica",
                    "CPU",
                    "Memory",
                    "Cache",
                    "Autoheal",
                    "Shedding",
                    "Reward",
                ],
                value=[],
                label="Recent Actions",
                interactive=False,
                wrap=True,
                elem_classes=["section-shell"],
            )

        raw_json = gr.Code(label="Raw JSON Response", language="json", interactive=False, elem_classes=["section-shell"])

        task_name.change(lambda name: _task_description(name), inputs=[task_name], outputs=[task_info])

        reset_btn.click(
            do_reset,
            inputs=[task_name],
            outputs=[
                latency,
                availability,
                cost,
                budget,
                queue,
                incident,
                telemetry_table,
                reward_plot,
                action_table,
                explanation,
                task_info,
                status,
                raw_json,
                baseline_compare,
            ],
        )

        step_btn.click(
            do_step,
            inputs=[task_name, replica, cpu, memory, cache, autoheal, shedding],
            outputs=[
                latency,
                availability,
                cost,
                budget,
                queue,
                incident,
                telemetry_table,
                reward_plot,
                action_table,
                explanation,
                task_info,
                status,
                raw_json,
                baseline_compare,
            ],
        )

        llm_btn.click(
            suggest_with_llm,
            inputs=[task_name],
            outputs=[replica, cpu, memory, cache, autoheal, shedding, status],
        )

        stabilize_btn.click(
            lambda: apply_preset("stabilize"),
            outputs=[replica, cpu, memory, cache, autoheal, shedding],
        )
        cost_btn.click(
            lambda: apply_preset("cut_cost"),
            outputs=[replica, cpu, memory, cache, autoheal, shedding],
        )
        incident_btn.click(
            lambda: apply_preset("incident"),
            outputs=[replica, cpu, memory, cache, autoheal, shedding],
        )

    return demo
