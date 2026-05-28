"""
ClassicalMLTrainer — fits, evaluates, and logs one (vectorizer, classifier) experiment.

Each experiment:
    1. Loads train.csv + val.csv from data/datasets/{dataset}/
    2. Builds sklearn Pipeline: vectorizer → classifier
    3. Fits on train set
    4. Predicts probabilities on val set
    5. Computes all metrics (recall_1, precision_1, MCC, ROC-AUC, latency, ...)
    6. Logs params + metrics + model artifact to MLflow
    7. Saves model as pickle to experiments/{dataset}/classical/models/
    8. Returns ExperimentResult

TEST SET POLICY:
    val.csv is used for development evaluation.
    test.csv is NEVER loaded here — only in phase_7_evaluate.py.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from backend.evaluation.plots import plot_confusion_matrix, plot_roc_curve
from backend.shared.metrics import compute_all_metrics, time_inference
from backend.shared.path_resolver import get_dataset_path, get_experiment_path
from backend.shared.settings_manager import settings_manager
from backend.training.classical.classifiers import build_classifier
from backend.training.classical.config import (
    ACTIVE_COMBOS,
)
from backend.training.classical.vectorizers import build_vectorizer
from backend.training.pymodels import ExperimentResult, MetricsResult

log = logging.getLogger(__name__)


class ClassicalMLTrainer:
    """Train and evaluate classical ML models on the generated dataset."""

    def train_experiment(
        self,
        dataset_name: str,
        vectorizer_key: str,
        classifier_key: str,
        vectorizer_params: dict | None = None,
        classifier_params: dict | None = None,
    ) -> ExperimentResult:
        """Run one (vectorizer, classifier) experiment.

        Args:
            dataset_name: e.g. "v1"
            vectorizer_key: Key from VECTORIZER_REGISTRY.
            classifier_key: Key from CLASSIFIER_REGISTRY.
            vectorizer_params: Overrides for vectorizer default params.
            classifier_params: Overrides for classifier default params.

        Returns:
            ExperimentResult with all metrics and artifact path.
        """
        experiment_id = f"{vectorizer_key}__{classifier_key}"
        log.info(f"Starting experiment: {experiment_id} on dataset '{dataset_name}'")

        dataset_dir = get_dataset_path(dataset_name)
        models_dir = get_experiment_path(dataset_name, "classical") / "models" / experiment_id
        models_dir.mkdir(parents=True, exist_ok=True)

        train_df = pd.read_csv(dataset_dir / "train.csv")
        val_df = pd.read_csv(dataset_dir / "val.csv")

        X_train = train_df["text"].tolist()
        y_train = train_df["label"].tolist()
        X_val = val_df["text"].tolist()
        y_val = val_df["label"].tolist()

        log.info(f"  train: {len(X_train):,} rows  val: {len(X_val):,} rows")

        vectorizer = build_vectorizer(vectorizer_key, **(vectorizer_params or {}))
        classifier = build_classifier(classifier_key, **(classifier_params or {}))

        log.info(f"  Fitting vectorizer ({vectorizer_key})...")
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)

        log.info(f"  Fitting classifier ({classifier_key})...")
        classifier.fit(X_train_vec, y_train)

        log.info("  Evaluating on val set...")
        y_pred = classifier.predict(X_val_vec)
        y_proba = [p[1] for p in classifier.predict_proba(X_val_vec)]

        raw_metrics = compute_all_metrics(y_val, y_pred, y_proba)

        log.info("  Benchmarking latency...")
        _, latency_stats = time_inference(
            predict_fn=lambda texts: classifier.predict(vectorizer.transform(texts)),
            texts=X_val[:500],
            n_warmup=10,
        )

        metrics = MetricsResult(
            **raw_metrics,
            latency_mean_ms=latency_stats["mean_ms"],
            latency_p50_ms=latency_stats["p50_ms"],
            latency_p95_ms=latency_stats["p95_ms"],
            latency_p99_ms=latency_stats["max_ms"],
        )

        log.info(f"  Results: {metrics.summary_line()}")
        if not metrics.passes_production_threshold:
            log.warning(
                f"  ⚠️  recall_1 = {metrics.recall_1:.4f} < {float(settings_manager.get('PRODUCTION_RECALL_THRESHOLD'))} — not production-safe"
            )

        plot_confusion_matrix(y_val, y_pred, models_dir, title=experiment_id)
        plot_roc_curve(y_val, y_proba, metrics.roc_auc, models_dir, title=experiment_id)

        model_path = models_dir / "model.pkl"
        with model_path.open("wb") as f:
            pickle.dump({"vectorizer": vectorizer, "classifier": classifier}, f)

        mlflow_run_id = log_to_mlflow(
            dataset_name=dataset_name,
            vectorizer_key=vectorizer_key,
            classifier_key=classifier_key,
            metrics=metrics,
            model_path=model_path,
            train_rows=len(X_train),
            val_rows=len(X_val),
        )

        result = ExperimentResult(
            experiment_id=experiment_id,
            approach="classical",
            dataset_name=dataset_name,
            vectorizer_key=vectorizer_key,
            classifier_key=classifier_key,
            metrics=metrics,
            model_path=str(model_path),
            mlflow_run_id=mlflow_run_id,
        )

        save_result_json(result, models_dir)
        return result

    def train_combos(
        self,
        dataset_name: str,
        combos: list[tuple[str, str]] | None = None,
        skip_on_failure: bool = True,
    ) -> list[ExperimentResult]:
        """Run a list of (vectorizer, classifier) experiments.

        Args:
            dataset_name: Dataset version to train on.
            combos: List of (vectorizer_key, classifier_key) pairs to run.
                    Defaults to ACTIVE_COMBOS when None.
            skip_on_failure: If True, log error and continue on any failure.

        Returns:
            List of ExperimentResult for all completed experiments.
        """
        if combos is None:
            combos = ACTIVE_COMBOS

        results: list[ExperimentResult] = []

        for vec_key, clf_key in combos:
            try:
                result = self.train_experiment(dataset_name, vec_key, clf_key)
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                if skip_on_failure:
                    log.error(f"Experiment {vec_key}__{clf_key} failed: {exc}")
                else:
                    raise

        print_summary(results)
        return results

    def load_model(self, model_path: str) -> tuple[Any, Any]:
        """Load a saved (vectorizer, classifier) pair from pickle."""
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        return bundle["vectorizer"], bundle["classifier"]


def log_to_mlflow(
    dataset_name: str,
    vectorizer_key: str,
    classifier_key: str,
    metrics: MetricsResult,
    model_path: Path,
    train_rows: int,
    val_rows: int,
) -> str:
    """Log experiment to MLflow. Returns run_id."""
    run_id = ""
    try:
        mlflow.set_experiment("classical_ml")
        with mlflow.start_run(run_name=f"{vectorizer_key}__{classifier_key}") as run:
            mlflow.log_params(
                {
                    "dataset_name": dataset_name,
                    "vectorizer_key": vectorizer_key,
                    "classifier_key": classifier_key,
                    "train_rows": train_rows,
                    "val_rows": val_rows,
                }
            )
            mlflow.log_metrics(
                {
                    "recall_1": metrics.recall_1,
                    "precision_1": metrics.precision_1,
                    "recall_0": metrics.recall_0,
                    "precision_0": metrics.precision_0,
                    "f1_macro": metrics.f1_macro,
                    "mcc": metrics.mcc,
                    "roc_auc": metrics.roc_auc,
                    "log_loss": metrics.log_loss,
                    "latency_p50_ms": metrics.latency_p50_ms,
                    "latency_p95_ms": metrics.latency_p95_ms,
                }
            )
            mlflow.log_artifact(str(model_path))
            run_id = run.info.run_id
    except Exception as exc:  # noqa: BLE001
        log.warning(f"MLflow logging skipped: {exc}")
    return run_id


def save_result_json(result: ExperimentResult, directory: Path) -> None:
    import json

    path = directory / "result.json"
    path.write_text(json.dumps(result.to_comparison_row(), indent=2))


def print_summary(results: list[ExperimentResult]) -> None:
    if not results:
        return
    print("\n" + "═" * 75)
    print(f"  CLASSICAL ML RESULTS ({len(results)} experiments)")
    print("═" * 75)
    print(f"  {'Experiment':<40} {'recall_1':>8} {'prec_1':>7} {'MCC':>7} {'pass':>5}")
    print("─" * 75)
    for r in sorted(results, key=lambda x: -x.metrics.recall_1):
        flag = "✅" if r.metrics.passes_production_threshold else "❌"
        print(
            f"  {r.experiment_id:<40} "
            f"{r.metrics.recall_1:>8.4f} "
            f"{r.metrics.precision_1:>7.4f} "
            f"{r.metrics.mcc:>7.4f} "
            f"{flag:>5}"
        )
    print("═" * 75)
    passing = [r for r in results if r.metrics.passes_production_threshold]
    if passing:
        best = max(passing, key=lambda r: r.metrics.mcc)
        print(
            f"\n  Best (production): {best.experiment_id}  "
            f"recall_1={best.metrics.recall_1:.4f}  MCC={best.metrics.mcc:.4f}"
        )
    else:
        best = max(results, key=lambda r: r.metrics.recall_1)
        print(f"\n  ⚠ No model passed recall threshold. Top by recall: {best.experiment_id}")
    print(f"  Model path: {best.model_path}")
    print("═" * 75 + "\n")
