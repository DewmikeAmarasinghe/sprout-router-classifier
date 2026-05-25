"""Pydantic models for the evaluation module."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.training.pymodels import MetricsResult


class ComparisonRow(BaseModel):
    """One row in the master model comparison table."""

    experiment_id: str
    approach: str  # "classical" | "transformer"
    model_name: str
    dataset_name: str
    passes_production_threshold: bool
    mlflow_run_id: str = ""
    notes: str = ""

    # All metrics — kept in sync with MetricsResult via from_experiment().
    accuracy: float = 0.0
    precision_0: float = 0.0
    precision_1: float = 0.0
    recall_0: float = 0.0
    recall_1: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    mcc: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    log_loss: float = 0.0
    ece: float = 0.0
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    @classmethod
    def from_experiment(
        cls,
        experiment_id: str,
        approach: str,
        model_name: str,
        dataset_name: str,
        metrics: MetricsResult,
        mlflow_run_id: str = "",
        notes: str = "",
    ) -> ComparisonRow:
        return cls(
            experiment_id=experiment_id,
            approach=approach,
            model_name=model_name,
            dataset_name=dataset_name,
            passes_production_threshold=metrics.passes_production_threshold,
            mlflow_run_id=mlflow_run_id,
            notes=notes,
            **metrics.model_dump(),
        )


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
