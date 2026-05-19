"""Pydantic models for the evaluation module."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ComparisonRow(BaseModel):
    """One row in the master model comparison table."""

    experiment_id: str
    approach: str  # "classical" | "transformer"
    model_name: str
    dataset_name: str
    recall_1: float
    precision_1: float
    recall_0: float
    precision_0: float
    mcc: float
    roc_auc: float
    f1_macro: float
    log_loss: float
    accuracy: float
    latency_p50_ms: float
    latency_p95_ms: float
    passes_production_threshold: bool
    mlflow_run_id: str = ""
    notes: str = ""


class CostSimResult(BaseModel):
    """Estimated daily cost savings from a routing strategy."""

    strategy_name: str
    daily_messages: int = Field(default=10_000)
    pct_routed_to_mini: float  # fraction sent to gpt-4o-mini
    pct_routed_to_4o: float  # fraction sent to gpt-4o
    daily_cost_usd: float
    baseline_cost_usd: float  # cost if all messages went to gpt-4o
    daily_savings_usd: float
    monthly_savings_usd: float
    recall_1: float  # safety metric for this strategy
    notes: str = ""


class AblationResult(BaseModel):
    """Result of one ablation configuration."""

    config_name: str  # e.g. "script_only", "best_classical", "best_transformer"
    recall_1: float
    precision_1: float
    mcc: float
    accuracy: float
    latency_p50_ms: float
    cost_per_day_usd: float
    notes: str = ""
