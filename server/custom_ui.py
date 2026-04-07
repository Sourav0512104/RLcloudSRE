"""Custom Gradio UI for the cloud SRE environment."""

from __future__ import annotations

import json
from typing import Any

import gradio as gr
import pandas as pd


def _format_metrics(obs: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        f"{obs.get('latency_ms', 0.0):.2f} ms",
        f"{obs.get('availability', 0.0):.4f}",
        f"${obs.get('hourly_cost_usd', 0.0):.2f}/hr",
        f"{obs.get('error_budget_remaining', 0.0):.4f}",
        f"{obs.get('queue_depth', 0.0):.2f}",
        f"{obs.get('incident_severity', 0.0):.2f}",
    )


def _telemetry_table(obs: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("Incoming RPM", f"{obs.get('incoming_rpm', 0.0):.2f}"),
        ("Effective RPM", f"{obs.get('effective_rpm', 0.0):.2f}"),
        ("Replicas", str(obs.get("replicas", 0))),
        ("CPU / Replica", f"{obs.get('cpu_per_replica', 0.0):.2f}"),
        ("Memory / Replica (GB)", f"{obs.get('memory_per_replica_gb', 0.0):.2f}"),
        ("Cache Ratio", f"{obs.get('cache_ratio', 0.0):.2f}"),
        ("Autoheal Level", f"{obs.get('autoheal_level', 0.0):.2f}"),
        ("Load Shedding", f"{obs.get('load_shedding', 0.0):.2f}"),
        ("Utilization", f"{obs.get('utilization', 0.0):.2f}"),
        ("Error Rate", f"{obs.get('error_rate', 0.0):.4f}"),
        ("SLO Burn Rate", f"{obs.get('slo_burn_rate', 0.0):.2f}"),
        ("Reward", f"{obs.get('reward', 0.0):.3f}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def _reward_history(logs: list[dict[str, Any]]) -> pd.DataFrame:
    if not logs:
        return pd.DataFrame({"step": [], "reward": []})
    return pd.DataFrame(
        {
            "step": [entry.get("step_count", i + 1) for i, entry in enumerate(logs)],
            "reward": [entry.get("reward", 0.0) or 0.0 for entry in logs],
        }
    )


def _action_history(logs: list[dict[str, Any]]) -> pd.DataFrame:
    if not logs:
        return pd.DataFrame(
            columns=[
                "step",
                "replica_delta",
                "cpu_delta",
                "memory_delta",
                "cache_delta",
                "autoheal_delta",
                "shedding_delta",
                "reward",
            ]
        )
    rows = []
    for entry in logs[-10:]:
        action = entry.get("action", {})
        rows.append(
            {
                "step": entry.get("step_count", 0),
                "replica_delta": action.get("replica_delta", 0.0),
                "cpu_delta": action.get("cpu_delta", 0.0),
                "memory_delta": action.get("memory_delta", 0.0),
                "cache_delta": action.get("cache_delta", 0.0),
                "autoheal_delta": action.get("autoheal_delta", 0.0),
                "shedding_delta": action.get("shedding_delta", 0.0),
                "reward": entry.get("reward", 0.0),
            }
        )
    return pd.DataFrame(rows)


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
        verdicts.append("The service is under pressure; capacity or mitigation likely needs to increase.")
    else:
        verdicts.append("The service is operating in a moderate state.")

    if cost > 170:
        verdicts.append("Infrastructure cost is elevated, so any further scaling should be justified.")
    else:
        verdicts.append("Cost is still in a reasonable range.")

    if incident > 0.2:
        verdicts.append("Incident severity is meaningful; auto-healing and safe capacity are important.")
    else:
        verdicts.append("No major incident is active right now.")

    return " ".join(verdicts)


def _preset_action(name: str) -> tuple[float, float, float, float, float, float]:
    presets = {
        "stabilize": (0.4, 0.2, 0.2, 0.2, 0.3, -0.1),
        "cut_cost": (-0.4, -0.2, -0.2, -0.1, 0.0, 0.0),
        "incident": (0.5, 0.3, 0.2, 0.1, 0.6, 0.05),
    }
    return presets[name]


def build_custom_gradio_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    """Build a custom operator dashboard mounted as the Custom tab."""

    async def do_reset():
        data = await web_manager.reset_environment()
        obs = data["observation"]
        logs = web_manager.episode_state.model_dump().get("action_logs", [])
        latency, availability, cost, budget, queue, incident = _format_metrics(obs)
        return (
            latency,
            availability,
            cost,
            budget,
            queue,
            incident,
            _telemetry_table(obs),
            _reward_history(logs),
            _action_history(logs),
            _explain(obs),
            json.dumps(data, indent=2),
            "Environment reset. Choose an action or a preset.",
        )

    async def do_step(replica, cpu, memory, cache, autoheal, shedding):
        action = {
            "replica_delta": replica,
            "cpu_delta": cpu,
            "memory_delta": memory,
            "cache_delta": cache,
            "autoheal_delta": autoheal,
            "shedding_delta": shedding,
        }
        data = await web_manager.step_environment(action)
        obs = data["observation"]
        logs = web_manager.episode_state.model_dump().get("action_logs", [])
        latency, availability, cost, budget, queue, incident = _format_metrics(obs)
        return (
            latency,
            availability,
            cost,
            budget,
            queue,
            incident,
            _telemetry_table(obs),
            _reward_history(logs),
            _action_history(logs),
            _explain(obs),
            json.dumps(data, indent=2),
            f"Step complete. Reward={data.get('reward', 0.0):.3f}",
        )

    def apply_preset(name: str):
        return _preset_action(name)

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Cloud SRE Control Center
            Use this dashboard like an SRE operator console. Reset the environment, try a control move,
            and watch how latency, availability, error budget, and cost respond.
            """
        )

        with gr.Row():
            latency = gr.Textbox(label="Latency", interactive=False)
            availability = gr.Textbox(label="Availability", interactive=False)
            cost = gr.Textbox(label="Hourly Cost", interactive=False)
            budget = gr.Textbox(label="Error Budget", interactive=False)
            queue = gr.Textbox(label="Queue Depth", interactive=False)
            incident = gr.Textbox(label="Incident Severity", interactive=False)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## Control Knobs")
                with gr.Row():
                    replica = gr.Slider(-1, 1, value=0, step=0.1, label="Replica Delta")
                    cpu = gr.Slider(-1, 1, value=0, step=0.1, label="CPU Delta")
                    memory = gr.Slider(-1, 1, value=0, step=0.1, label="Memory Delta")
                with gr.Row():
                    cache = gr.Slider(-1, 1, value=0, step=0.1, label="Cache Delta")
                    autoheal = gr.Slider(-1, 1, value=0, step=0.1, label="Autoheal Delta")
                    shedding = gr.Slider(-1, 1, value=0, step=0.1, label="Shedding Delta")
                with gr.Row():
                    reset_btn = gr.Button("Reset")
                    step_btn = gr.Button("Apply Action", variant="primary")
                with gr.Row():
                    stabilize_btn = gr.Button("Stabilize Service")
                    cost_btn = gr.Button("Cut Cost")
                    incident_btn = gr.Button("Incident Recovery")
                status = gr.Textbox(label="Status", interactive=False)
                explanation = gr.Markdown("Reset the environment to begin.")

            with gr.Column(scale=1):
                telemetry_table = gr.Dataframe(
                    headers=["Metric", "Value"],
                    row_count=12,
                    col_count=(2, "fixed"),
                    label="Telemetry Snapshot",
                    interactive=False,
                )

        with gr.Row():
            reward_plot = gr.LinePlot(
                x="step",
                y="reward",
                label="Reward Over Time",
                title="Reward Trend",
            )
            action_table = gr.Dataframe(
                label="Recent Actions",
                interactive=False,
            )

        raw_json = gr.Code(label="Raw JSON Response", language="json", interactive=False)

        reset_btn.click(
            do_reset,
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
                raw_json,
                status,
            ],
        )

        step_btn.click(
            do_step,
            inputs=[replica, cpu, memory, cache, autoheal, shedding],
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
                raw_json,
                status,
            ],
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
