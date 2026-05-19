"""
AblationRunner — compare trained models against baselines.

Baselines tested against the test set (accessed ONCE here, never during development):
    script_only   — is_pure_script() → label=1, everything else → label=0
    all_gpt4o     — always predicts label=1 (100% recall, 0% savings)
    all_mini      — always predicts label=0 (0% safety — worst case)

Then the best classical model and best transformer model are evaluated
against the same test set and compared to the baselines.

TEST SET POLICY:
    test.csv is loaded ONCE here in Phase 7. Never in training or validation.
    All model selection was done on val.csv.

Usage:
    runner   = AblationRunner()
    results  = runner.run("v1", predictor)
    # results: list[AblationResult] sorted by recall_1 descending
"""

from __future__ import annotations

import logging

import pandas as pd

from backend.evaluation.comparator import ModelComparator
from backend.evaluation.pymodels import AblationResult
from backend.router.predictor import RouterPredictor
from backend.shared.metrics import compute_all_metrics
from backend.shared.path_resolver import get_dataset_path
from backend.shared.script_detector import is_pure_script

log = logging.getLogger(__name__)


class AblationRunner:
    """Runs ablation study comparing baselines against trained routing models."""

    def run(self, dataset_name: str) -> list[AblationResult]:
        """Load test.csv and evaluate all strategies.

        Args:
            dataset_name: e.g. "v1".

        Returns:
            List of AblationResult sorted by recall_1 descending.
        """
        test_path = get_dataset_path(dataset_name) / "test.csv"
        if not test_path.exists():
            raise FileNotFoundError(
                f"test.csv not found at {test_path}. Run phase_3_split.py first."
            )

        test_df = pd.read_csv(test_path)
        texts = test_df["text"].tolist()
        labels = test_df["label"].astype(int).tolist()

        log.info(f"Ablation on test set: {len(test_df):,} rows")

        results = [
            evaluate_script_only(texts, labels),
            evaluate_constant_prediction("all_gpt4o", label=1, texts=texts, labels=labels),
            evaluate_constant_prediction("all_mini", label=0, texts=texts, labels=labels),
        ]

        best_row = ModelComparator().best_model(dataset_name)
        if best_row:
            results.append(
                evaluate_trained_model(
                    texts=texts,
                    labels=labels,
                    experiment_id=best_row.experiment_id,
                    dataset_name=dataset_name,
                    approach=best_row.approach,
                )
            )

        results.sort(key=lambda r: r.recall_1, reverse=True)

        print_summary(results)
        return results


def evaluate_script_only(texts: list[str], labels: list[int]) -> AblationResult:
    """Layer 1 only: is_pure_script() → 1, else → 0."""
    preds = [1 if is_pure_script(t) else 0 for t in texts]
    metrics = compute_all_metrics(labels, preds, [float(p) for p in preds])
    return AblationResult(
        config_name="script_only",
        recall_1=metrics["recall_1"],
        precision_1=metrics["precision_1"],
        mcc=metrics["mcc"],
        accuracy=metrics["accuracy"],
        latency_p50_ms=0.1,
        cost_per_day_usd=0.0,
        notes="Unicode rule only — no ML. Catches pure Sinhala/Tamil script only.",
    )


def evaluate_constant_prediction(
    config_name: str,
    label: int,
    texts: list[str],
    labels: list[int],
) -> AblationResult:
    """Baseline: always predict the same label."""
    preds = [label] * len(labels)
    metrics = compute_all_metrics(labels, preds, [float(label)] * len(labels))
    notes = (
        "Baseline: 100% recall, 0 cost savings."
        if label == 1
        else "Unsafe baseline: 0% recall_1, maximum cost savings but terrible UX."
    )
    return AblationResult(
        config_name=config_name,
        recall_1=metrics["recall_1"],
        precision_1=metrics["precision_1"],
        mcc=metrics["mcc"],
        accuracy=metrics["accuracy"],
        latency_p50_ms=0.0,
        cost_per_day_usd=0.0,
        notes=notes,
    )


def evaluate_trained_model(
    texts: list[str],
    labels: list[int],
    experiment_id: str,
    dataset_name: str,
    approach: str,
) -> AblationResult:
    """Evaluate the best trained model on test.csv."""
    from backend.shared.path_resolver import get_experiment_path

    if approach == "classical":
        model_path = (
            get_experiment_path(dataset_name, "classical") / "models" / f"{experiment_id}.pkl"
        )
        predictor = RouterPredictor.from_pkl(model_path)
    else:
        checkpoint_dir = (
            get_experiment_path(dataset_name, "transformers") / "models" / experiment_id
        )
        predictor = RouterPredictor.from_hf_checkpoint(checkpoint_dir)

    predictions = predictor.predict_batch(texts)
    preds = [p.label for p in predictions]
    probas = [p.confidence for p in predictions]
    metrics = compute_all_metrics(labels, preds, probas)

    return AblationResult(
        config_name=f"router_{experiment_id}",
        recall_1=metrics["recall_1"],
        precision_1=metrics["precision_1"],
        mcc=metrics["mcc"],
        accuracy=metrics["accuracy"],
        latency_p50_ms=metrics.get("latency_p50_ms", 0.0),
        cost_per_day_usd=0.0,
        notes=f"Best trained model: {approach} / {experiment_id}",
    )


def print_summary(results: list[AblationResult]) -> None:
    print("\n" + "═" * 80)
    print("  ABLATION RESULTS (test set)")
    print("═" * 80)
    print(f"  {'Config':<35} {'recall_1':>8} {'prec_1':>7} {'MCC':>7} {'pass':>5}")
    print("─" * 80)
    for r in results:
        flag = "✅" if r.recall_1 >= 0.97 else "❌"
        print(
            f"  {r.config_name:<35} "
            f"{r.recall_1:>8.4f} "
            f"{r.precision_1:>7.4f} "
            f"{r.mcc:>7.4f} "
            f"{flag:>5}"
        )
    print("═" * 80 + "\n")
