"""Typed action and observation models for the cloud SRE environment."""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CloudSreRlAction(Action):
    """Continuous control knobs exposed to the RL policy."""

    replica_delta: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Scale replicas up or down. -1 reduces capacity, +1 adds capacity.",
    )
    cpu_delta: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Adjust CPU allocated per replica.",
    )
    memory_delta: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Adjust memory allocated per replica.",
    )
    cache_delta: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Adjust cache aggressiveness to trade cost for latency.",
    )
    autoheal_delta: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Increase or decrease automation used for incident mitigation.",
    )
    shedding_delta: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Adjust request shedding pressure to protect reliability during overload.",
    )


class CloudSreRlObservation(Observation):
    """Telemetry returned after each control step."""

    time_index: int = Field(default=0, description="Step index within the episode.")
    incoming_rpm: float = Field(default=0.0, description="Incoming traffic in requests per minute.")
    effective_rpm: float = Field(default=0.0, description="Traffic after shedding and mitigation.")
    replicas: int = Field(default=0, description="Replica count currently serving traffic.")
    cpu_per_replica: float = Field(default=0.0, description="vCPU assigned to each replica.")
    memory_per_replica_gb: float = Field(default=0.0, description="Memory in GB assigned to each replica.")
    cache_ratio: float = Field(default=0.0, description="Cache aggressiveness level.")
    autoheal_level: float = Field(default=0.0, description="Automation intensity for self-healing.")
    load_shedding: float = Field(default=0.0, description="Fraction of traffic intentionally shed.")
    utilization: float = Field(default=0.0, description="Effective utilization of provisioned capacity.")
    queue_depth: float = Field(default=0.0, description="Accumulated queued requests.")
    latency_ms: float = Field(default=0.0, description="Approximate p95 latency in milliseconds.")
    error_rate: float = Field(default=0.0, description="Fraction of requests failing.")
    availability: float = Field(default=0.0, description="Availability ratio during the decision interval.")
    slo_burn_rate: float = Field(default=0.0, description="How fast the service is burning error budget.")
    error_budget_remaining: float = Field(default=1.0, description="Normalized error budget remaining.")
    hourly_cost_usd: float = Field(default=0.0, description="Estimated infrastructure cost per hour.")
    incident_severity: float = Field(default=0.0, description="Current incident severity from 0 to 1.")
    performance_score: float = Field(default=0.0, description="Composite service health score.")
    feature_vector: list[float] = Field(
        default_factory=list,
        description="Normalized feature vector used by the PPO trainer.",
    )

