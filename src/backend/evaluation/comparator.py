"""
ModelComparator — aggregate all trained model results.

IMPORTANT: result.json files are saved in flat format (all metric fields at top level).
This module reconstructs the nested ExperimentResult / MetricsResult from the flat dict.

Usage:
    comparator = ModelComparator()
    rows       = comparator.compare("v1")
    best       = comparator.best_model("v1")
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from backend.evaluation.pymodels import ComparisonRow
from backend.shared.path_resolver import get_experiment_path
from backend.training.pymodels import ExperimentResult, MetricsResult

log = logging.getLogger(__name__)

# All metric field names that may appear at the top level of a flat result.json
METRIC_FIELD_NAMES = set(MetricsResult.field_names())


class ModelComparator:
    """Scans disk for all experiment results and builds a unified comparison table."""

    def compare(self, dataset_name: str) -> list[ComparisonRow]:
        """Load all result.json files and return a sorted comparison table."""
        results = load_all_results(dataset_name)

        if not results:
            log.warning(
                f"No experiment results found for '{dataset_name}'. "
                "Run phase_5_train_classical.py and/or phase_6_train_transformers.py first."
            )
            return []

        rows = [to_comparison_row(r) for r in results]
        rows.sort(key=lambda r: (r.passes_production_threshold, r.recall_1), reverse=True)

        save_csv(rows, dataset_name)
        return rows

    def best_model(self, dataset_name: str) -> ComparisonRow | None:
        """Return the best model meeting the production threshold (recall_1 >= PRODUCTION_RECALL_THRESHOLD).

        Selection priority: passes threshold → highest MCC → fastest latency.
        Returns None if no results found.
        """
        rows = self.compare(dataset_name)
        if not rows:
            return None

        passing = [r for r in rows if r.passes_production_threshold]
        if passing:
            return max(passing, key=lambda r: (r.mcc, -r.latency_p50_ms))

        log.warning("No model meets production threshold. Returning best available by recall_1.")
        return max(rows, key=lambda r: r.recall_1)

    def load_saved_csv(self, dataset_name: str) -> pd.DataFrame:
        """Load a previously saved master_comparison.csv."""
        csv_path = get_experiment_path(dataset_name, "classical").parent / "master_comparison.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"No comparison table at {csv_path}. Run ModelComparator.compare() first."
            )
        return pd.read_csv(csv_path)


def load_all_results(dataset_name: str) -> list[ExperimentResult]:
    """Scan disk for result.json files from both classical and transformer experiments.

    Handles two JSON formats:
    - Flat (current): all metric fields at top level, no nested 'metrics' key
    - Nested (new): full ExperimentResult.model_dump() structure with nested 'metrics'
    """
    exp_dir = get_experiment_path(dataset_name, "classical").parent
    results: list[ExperimentResult] = []

    for approach in ("classical", "transformers"):
        results_dir = exp_dir / approach / "models"
        if not results_dir.exists():
            continue

        for result_file in sorted(results_dir.rglob("result.json")):
            try:
                data = json.loads(result_file.read_text())
                result = parse_result_json(data, dataset_name)
                if result:
                    results.append(result)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Could not load {result_file}: {exc}")

    log.info(f"Loaded {len(results)} experiment results for '{dataset_name}'")
    return results


def parse_result_json(data: dict, dataset_name: str) -> ExperimentResult | None:
    """Parse result.json data in either flat or nested format.

    Flat format (from trainer saving result.to_comparison_row()):
        {"recall_1": 0.98, "mcc": 0.64, "experiment_id": "...", ...}

    Nested format (from trainer saving result.model_dump()):
        {"experiment_id": "...", "metrics": {"recall_1": 0.98, ...}, ...}
    """
    if "metrics" in data and isinstance(data["metrics"], dict):
        # Nested format — standard Pydantic model_validate
        return ExperimentResult.model_validate(data)

    # Flat format — reconstruct nested MetricsResult from top-level fields
    metrics_data = MetricsResult().model_dump()
    for name in MetricsResult.field_names():
        if name in data:
            metrics_data[name] = data[name]
    metrics = MetricsResult.model_validate(metrics_data)

    # Infer experiment_id from vectorizer+classifier keys if not stored directly
    experiment_id = data.get("experiment_id") or (
        f"{data.get('vectorizer_key', '')}_{data.get('classifier_key', '')}"
    )

    return ExperimentResult(
        experiment_id=experiment_id,
        approach=data.get("approach", "classical"),
        dataset_name=data.get("dataset_name", dataset_name),
        model_name=data.get("model_name", experiment_id),
        metrics=metrics,
        model_path=data.get("model_path", ""),
        mlflow_run_id=data.get("mlflow_run_id", ""),
        notes=data.get("notes", ""),
    )


def to_comparison_row(result: ExperimentResult) -> ComparisonRow:
    """Convert ExperimentResult to a ComparisonRow for display."""
    return ComparisonRow.from_experiment(
        experiment_id=result.experiment_id,
        approach=result.approach,
        model_name=result.model_name or result.experiment_id,
        dataset_name=result.dataset_name,
        metrics=result.metrics,
        mlflow_run_id=result.mlflow_run_id,
        notes=result.notes,
    )


def save_csv(rows: list[ComparisonRow], dataset_name: str) -> None:
    exp_root = get_experiment_path(dataset_name, "classical").parent
    csv_path = exp_root / "master_comparison.csv"
    df = pd.DataFrame([r.model_dump() for r in rows])
    df.to_csv(csv_path, index=False)
    log.info(f"Saved master comparison: {csv_path}")
