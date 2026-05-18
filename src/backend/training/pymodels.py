"""
Pydantic models shared by classical and transformer training modules.

ExperimentResult is the standard output from any training run.
All trainers return this model so the comparator can work uniformly.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class MetricsResult(BaseModel):
    """All evaluation metrics for one experiment."""

    accuracy: float = 0.0
    precision_0: float = 0.0
    precision_1: float = 0.0
    recall_0: float = 0.0
    recall_1: float = 0.0  # MUST be >= 0.97 in production
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    mcc: float = 0.0  # Matthews Correlation Coefficient
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    log_loss: float = 0.0
    ece: float = 0.0  # Expected Calibration Error

    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    @property
    def passes_production_threshold(self) -> bool:
        """True if this model is safe to deploy (recall_1 >= 0.97)."""
        return self.recall_1 >= 0.97

    def summary_line(self) -> str:
        flag = "" if self.passes_production_threshold else " ⚠️ recall_1 < 0.97"
        return (
            f"recall_1={self.recall_1:.4f}  "
            f"precision_1={self.precision_1:.4f}  "
            f"mcc={self.mcc:.4f}  "
            f"roc_auc={self.roc_auc:.4f}  "
            f"p95={self.latency_p95_ms:.1f}ms{flag}"
        )


class ExperimentResult(BaseModel):
    """Standard result from one training experiment (classical or transformer)."""

    experiment_id: str  # "{vectorizer}__{classifier}" or "{model_name}"
    approach: str  # "classical" or "transformer"
    dataset_name: str
    vectorizer_key: str = ""  # classical only
    classifier_key: str = ""  # classical only
    model_name: str = ""  # transformer only
    metrics: MetricsResult = Field(default_factory=MetricsResult)
    model_path: str = ""  # path to saved model artifact
    mlflow_run_id: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""

    def to_comparison_row(self) -> dict:
        """Flatten to one row for master_comparison.csv."""
        return {
            "experiment_id": self.experiment_id,
            "approach": self.approach,
            "vectorizer_key": self.vectorizer_key,
            "classifier_key": self.classifier_key,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "recall_1": self.metrics.recall_1,
            "precision_1": self.metrics.precision_1,
            "recall_0": self.metrics.recall_0,
            "precision_0": self.metrics.precision_0,
            "mcc": self.metrics.mcc,
            "roc_auc": self.metrics.roc_auc,
            "pr_auc": self.metrics.pr_auc,
            "f1_macro": self.metrics.f1_macro,
            "log_loss": self.metrics.log_loss,
            "ece": self.metrics.ece,
            "latency_p50_ms": self.metrics.latency_p50_ms,
            "latency_p95_ms": self.metrics.latency_p95_ms,
            "passes": self.metrics.passes_production_threshold,
            "model_path": self.model_path,
            "mlflow_run_id": self.mlflow_run_id,
            "timestamp": self.timestamp,
        }
