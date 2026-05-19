"""
ModelComparator — aggregate all trained model results.

Scans experiments/{dataset}/classical/results/ and
         experiments/{dataset}/transformers/results/
for result.json files written by ClassicalMLTrainer and TransformerTrainer.

Builds a unified master_comparison.csv ranked by recall_1 descending,
and returns the best model meeting the production threshold.

Usage:
    comparator = ModelComparator()
    table      = comparator.compare("v1")
    best       = comparator.best_model("v1")
    print(f"Best: {best.experiment_id}  recall_1={best.recall_1:.4f}")
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from backend.evaluation.pymodels import ComparisonRow
from backend.shared.path_resolver import get_experiment_path
from backend.training.pymodels import ExperimentResult

log = logging.getLogger(__name__)

PRODUCTION_RECALL_THRESHOLD = 0.97


class ModelComparator:
    """Scans disk for all experiment results and builds a unified comparison table."""

    def compare(self, dataset_name: str) -> list[ComparisonRow]:
        """Load all result.json files and return a sorted comparison table.

        Args:
            dataset_name: e.g. "v1".

        Returns:
            List of ComparisonRow sorted by recall_1 descending.
        """
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
        """Return the best experiment result meeting the production threshold.

        Preference: passes threshold → highest MCC → fastest latency.
        Returns None if no results are found.
        """
        rows = self.compare(dataset_name)
        if not rows:
            return None

        passing = [r for r in rows if r.passes_production_threshold]
        if passing:
            return max(passing, key=lambda r: (r.mcc, -r.latency_p50_ms))

        log.warning("No model meets production threshold. Returning best available.")
        return max(rows, key=lambda r: r.recall_1)

    def load_saved_csv(self, dataset_name: str) -> pd.DataFrame:
        """Load the previously saved master_comparison.csv."""
        csv_path = get_experiment_path(dataset_name, "classical").parent / "master_comparison.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"No comparison table at {csv_path}. Run ModelComparator.compare() first."
            )
        return pd.read_csv(csv_path)


def load_all_results(dataset_name: str) -> list[ExperimentResult]:
    """Scan disk for result.json files from both classical and transformer experiments."""
    exp_dir = get_experiment_path(dataset_name, "classical").parent
    results: list[ExperimentResult] = []

    for approach in ("classical", "transformers"):
        results_dir = exp_dir / approach / "models"
        if not results_dir.exists():
            continue
        for result_file in results_dir.rglob("result.json"):
            try:
                data = json.loads(result_file.read_text())
                results.append(ExperimentResult.model_validate(data))
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Could not load {result_file}: {exc}")

    log.info(f"Loaded {len(results)} experiment results for '{dataset_name}'")
    return results


def to_comparison_row(result: ExperimentResult) -> ComparisonRow:
    """Convert ExperimentResult to a ComparisonRow."""
    m = result.metrics
    return ComparisonRow(
        experiment_id=result.experiment_id,
        approach=result.approach,
        model_name=result.model_name,
        dataset_name=result.dataset_name,
        recall_1=m.recall_1,
        precision_1=m.precision_1,
        recall_0=m.recall_0,
        precision_0=m.precision_0,
        mcc=m.mcc,
        roc_auc=m.roc_auc,
        f1_macro=m.f1_macro,
        log_loss=m.log_loss,
        accuracy=m.accuracy,
        latency_p50_ms=m.latency_p50_ms,
        latency_p95_ms=m.latency_p95_ms,
        passes_production_threshold=m.passes_production_threshold,
        mlflow_run_id=result.mlflow_run_id,
        notes=result.notes,
    )


def save_csv(rows: list[ComparisonRow], dataset_name: str) -> None:
    """Save master_comparison.csv alongside the experiment directories."""
    exp_root = get_experiment_path(dataset_name, "classical").parent
    csv_path = exp_root / "master_comparison.csv"

    df = pd.DataFrame([r.model_dump() for r in rows])
    df.to_csv(csv_path, index=False)
    log.info(f"Saved master comparison: {csv_path}")
