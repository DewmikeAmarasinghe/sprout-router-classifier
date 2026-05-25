"""
Pydantic models shared by classical and transformer training modules.

ExperimentResult is the standard output from any training run.
All trainers return this model so the comparator can work uniformly.

DESIGN NOTE — settings_manager in model methods:
    passes_production_threshold and summary_line read PRODUCTION_RECALL_THRESHOLD
    from settings_manager at call-time, not at import-time. This means changing
    settings.py (or editing it via the Gradio settings panel) takes effect
    immediately without restarting the process.
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
    recall_1: float = 0.0
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

    @classmethod
    def field_names(cls) -> list[str]:
        """All metric field names, in model definition order."""
        return list(cls.model_fields.keys())

    @staticmethod
    def format_cell(field: str, value: float) -> str:
        """Format one metric value for display in Gradio tables."""
        if field.startswith("latency_"):
            return f"{value:.1f}ms"
        return f"{value:.4f}"

    @property
    def passes_production_threshold(self) -> bool:
        """True if this model is safe to deploy.

        Reads PRODUCTION_RECALL_THRESHOLD from settings_manager at call-time
        so that changing settings.py is reflected immediately without restart.
        """
        from backend.shared.settings_manager import settings_manager

        threshold = float(settings_manager.get("PRODUCTION_RECALL_THRESHOLD"))
        return self.recall_1 >= threshold

    def summary_line(self) -> str:
        """One-line human-readable summary showing whether this model passes.

        Uses the live settings threshold so the ⚠ flag updates with settings changes.
        """
        from backend.shared.settings_manager import settings_manager

        threshold = float(settings_manager.get("PRODUCTION_RECALL_THRESHOLD"))
        flag = "" if self.recall_1 >= threshold else f" ⚠️ recall_1 < {threshold}"
        return (
            f"recall_1={self.recall_1:.4f}  "
            f"precision_1={self.precision_1:.4f}  "
            f"mcc={self.mcc:.4f}  "
            f"roc_auc={self.roc_auc:.4f}  "
            f"p95={self.latency_p95_ms:.1f}ms{flag}"
        )


# Table headers derived from MetricsResult — add a field to the model and it appears in UI.
CLASSICAL_TABLE_HEADERS: list[str] = (
    ["Vectorizer", "Classifier"] + MetricsResult.field_names() + ["Pass"]
)
TRANSFORMER_TABLE_HEADERS: list[str] = ["Model"] + MetricsResult.field_names() + ["Pass"]
EVALUATION_TABLE_HEADERS: list[str] = (
    ["Experiment", "Approach"] + MetricsResult.field_names() + ["Pass"]
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
        """Flatten to one row for master_comparison.csv and result.json."""
        return {
            "experiment_id": self.experiment_id,
            "approach": self.approach,
            "vectorizer_key": self.vectorizer_key,
            "classifier_key": self.classifier_key,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            **self.metrics.model_dump(),
            "passes": self.metrics.passes_production_threshold,
            "model_path": self.model_path,
            "mlflow_run_id": self.mlflow_run_id,
            "timestamp": self.timestamp,
        }
