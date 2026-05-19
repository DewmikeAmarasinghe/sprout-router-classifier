"""
TransformerTrainer — fine-tune and evaluate one transformer model.

Mirrors the ClassicalMLTrainer interface: train_experiment() returns an
ExperimentResult so the comparator can work uniformly across approaches.

WORKFLOW per experiment:
    1. Load train.csv + val.csv from data/datasets/{dataset_name}/
    2. Tokenize via dataset.py → HuggingFace Datasets
    3. Load AutoModelForSequenceClassification (num_labels=2)
    4. Build TrainingArguments from TRAIN_CONFIG + overrides
    5. Build Trainer with compute_metrics callback
    6. trainer.train()
    7. Evaluate on val set → all metrics
    8. Benchmark latency on 500 val samples (CPU inference)
    9. Log everything to MLflow
    10. Save HuggingFace checkpoint
    11. Return ExperimentResult

TEST SET POLICY:
    val.csv is used here for development evaluation.
    test.csv is NEVER loaded — only in phase_8_evaluate.py.

HUGGINGFACE USAGE:
    Only for pre-trained model weights (xlm-roberta-base, etc.).
    All training data comes from our generation pipeline — no HF datasets used.
"""

from __future__ import annotations

import json
import logging
import time
import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
import pandas as pd
from scipy.special import softmax

from backend.shared.metrics import compute_all_metrics
from backend.shared.path_resolver import get_dataset_path, get_experiment_path
from backend.training.pymodels import ExperimentResult, MetricsResult
from backend.training.transformers.config import TRAIN_CONFIG, TRANSFORMER_REGISTRY

if TYPE_CHECKING:
    from transformers import Trainer, TrainingArguments

log = logging.getLogger(__name__)


class TransformerTrainer:
    """Fine-tune and evaluate transformer models on the generated dataset."""

    def train_experiment(
        self,
        dataset_name: str,
        model_key: str,
        param_overrides: dict | None = None,
    ) -> ExperimentResult:
        """Fine-tune one model and return evaluation results.

        Args:
            dataset_name: e.g. "v1"
            model_key: Key from TRANSFORMER_REGISTRY (e.g. "xlmr-base").
            param_overrides: Dict of TRAIN_CONFIG keys to override.

        Returns:
            ExperimentResult with all metrics, model path, and MLflow run id.
        """
        from transformers import AutoModelForSequenceClassification, Trainer

        from backend.training.transformers.dataset import load_and_tokenize

        if model_key not in TRANSFORMER_REGISTRY:
            raise ValueError(
                f"Unknown model key {model_key!r}. Available: {sorted(TRANSFORMER_REGISTRY)}"
            )

        spec = TRANSFORMER_REGISTRY[model_key]
        config = {**TRAIN_CONFIG, **(param_overrides or {})}
        max_length = int(config["max_length"])

        log.info(
            f"Starting transformer experiment: {model_key} ({spec.hf_name}) on '{dataset_name}'"
        )

        dataset_dir = get_dataset_path(dataset_name)
        output_dir = get_experiment_path(dataset_name, "transformers") / "models" / model_key
        output_dir.mkdir(parents=True, exist_ok=True)

        train_df = pd.read_csv(dataset_dir / "train.csv")
        val_df = pd.read_csv(dataset_dir / "val.csv")
        log.info(f"  train: {len(train_df):,} rows  val: {len(val_df):,} rows")

        train_ds, val_ds, tokenizer = load_and_tokenize(
            train_df, val_df, spec.hf_name, max_length=max_length
        )

        log.info(f"  Loading model: {spec.hf_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            spec.hf_name,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )

        training_args = build_training_args(config, output_dir)

        def compute_metrics(eval_pred: Any) -> dict[str, float]:
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            proba = softmax(logits, axis=-1)[:, 1]
            raw = compute_all_metrics(labels.tolist(), preds.tolist(), proba.tolist())
            return {
                "f1_macro": raw["f1_macro"],
                "recall_1": raw["recall_1"],
                "mcc": raw["mcc"],
                "roc_auc": raw["roc_auc"],
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        log.info("  Training ...")
        train_start = time.time()
        trainer.train()
        train_time_s = time.time() - train_start
        log.info(f"  Training done in {train_time_s:.0f}s")

        log.info("  Evaluating on val set ...")
        trainer.evaluate()

        log.info("  Benchmarking latency (500 samples, CPU) ...")
        latency_stats = benchmark_latency(
            model, tokenizer, val_df["text"].tolist()[:500], max_length
        )

        log.info("  Measuring peak inference RAM ...")
        peak_ram_mb = measure_peak_ram(model, tokenizer, val_df["text"].tolist()[:100], max_length)

        val_logits, val_labels = get_logits_and_labels(trainer, val_ds)
        val_preds = np.argmax(val_logits, axis=-1)
        val_proba = softmax(val_logits, axis=-1)[:, 1]
        raw_metrics = compute_all_metrics(
            val_labels.tolist(), val_preds.tolist(), val_proba.tolist()
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
            log.warning(f"  recall_1 = {metrics.recall_1:.4f} < 0.97 — not production-safe")

        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        log.info(f"  Checkpoint saved: {output_dir}")

        mlflow_run_id = log_to_mlflow(
            dataset_name=dataset_name,
            model_key=model_key,
            spec=spec,
            config=config,
            metrics=metrics,
            train_rows=len(train_df),
            val_rows=len(val_df),
            train_time_s=train_time_s,
            peak_ram_mb=peak_ram_mb,
            output_dir=output_dir,
        )

        result = ExperimentResult(
            experiment_id=model_key,
            approach="transformer",
            dataset_name=dataset_name,
            model_name=model_key,
            metrics=metrics,
            model_path=str(output_dir),
            mlflow_run_id=mlflow_run_id,
            notes=spec.hf_name,
        )

        save_result_json(result, output_dir)
        return result

    def train_all_models(
        self,
        dataset_name: str,
        skip_on_failure: bool = True,
    ) -> list[ExperimentResult]:
        """Fine-tune all models defined in TRANSFORMER_REGISTRY."""
        results: list[ExperimentResult] = []
        for model_key in TRANSFORMER_REGISTRY:
            try:
                result = self.train_experiment(dataset_name, model_key)
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                if skip_on_failure:
                    log.error(f"Transformer experiment {model_key} failed: {exc}")
                else:
                    raise
        print_summary(results)
        return results


# ── helpers ──────────────────────────────────────────────────────────────────


def build_training_args(config: dict, output_dir: Path) -> TrainingArguments:
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=int(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        learning_rate=float(config["learning_rate"]),
        warmup_ratio=float(config["warmup_ratio"]),
        weight_decay=float(config["weight_decay"]),
        fp16=bool(config["fp16"]),
        eval_strategy=str(config["eval_strategy"]),
        save_strategy=str(config["save_strategy"]),
        load_best_model_at_end=bool(config["load_best_model_at_end"]),
        metric_for_best_model=str(config["metric_for_best_model"]),
        greater_is_better=bool(config["greater_is_better"]),
        logging_steps=int(config["logging_steps"]),
        dataloader_num_workers=int(config["dataloader_num_workers"]),
        report_to="none",
    )


def get_logits_and_labels(
    trainer: Trainer,
    val_ds: Any,
) -> tuple[np.ndarray, np.ndarray]:
    pred_output = trainer.predict(val_ds)
    logits = np.array(pred_output.predictions)
    labels = np.array(pred_output.label_ids)
    return logits, labels


def benchmark_latency(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    batch_size: int = 32,
) -> dict[str, float]:
    """Measure CPU inference latency on a list of texts."""
    import torch

    model.eval()
    device = torch.device("cpu")
    model_cpu = model.to(device)

    latencies: list[float] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        start = time.perf_counter()
        with torch.no_grad():
            model_cpu(**encoded)
        elapsed_ms = (time.perf_counter() - start) * 1000 / len(batch_texts)
        latencies.extend([elapsed_ms] * len(batch_texts))

    arr = np.array(latencies)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }


def measure_peak_ram(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
) -> float:
    """Return peak inference RAM usage in MB via tracemalloc."""
    import torch

    model.eval()
    device = torch.device("cpu")
    model_cpu = model.to(device)

    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    tracemalloc.start()
    with torch.no_grad():
        model_cpu(**encoded)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


def log_to_mlflow(
    dataset_name: str,
    model_key: str,
    spec: Any,
    config: dict,
    metrics: MetricsResult,
    train_rows: int,
    val_rows: int,
    train_time_s: float,
    peak_ram_mb: float,
    output_dir: Path,
) -> str:
    """Log experiment to MLflow. Returns run_id."""
    run_id = ""
    try:
        mlflow.set_experiment(f"transformers_{dataset_name}")
        with mlflow.start_run(run_name=model_key) as run:
            mlflow.log_params(
                {
                    "model_key": model_key,
                    "hf_name": spec.hf_name,
                    "dataset_name": dataset_name,
                    "train_rows": train_rows,
                    "val_rows": val_rows,
                    **{k: v for k, v in config.items()},
                }
            )
            mlflow.log_metrics(
                {
                    "recall_1": metrics.recall_1,
                    "precision_1": metrics.precision_1,
                    "recall_0": metrics.recall_0,
                    "precision_0": metrics.precision_0,
                    "mcc": metrics.mcc,
                    "roc_auc": metrics.roc_auc,
                    "pr_auc": metrics.pr_auc,
                    "f1_macro": metrics.f1_macro,
                    "log_loss": metrics.log_loss,
                    "ece": metrics.ece,
                    "accuracy": metrics.accuracy,
                    "latency_p50_ms": metrics.latency_p50_ms,
                    "latency_p95_ms": metrics.latency_p95_ms,
                    "train_time_s": train_time_s,
                    "peak_ram_mb": peak_ram_mb,
                }
            )
            mlflow.log_artifact(str(output_dir))
            run_id = run.info.run_id
    except Exception as exc:  # noqa: BLE001
        log.warning(f"MLflow logging skipped: {exc}")
    return run_id


def save_result_json(result: ExperimentResult, directory: Path) -> None:
    path = directory / "result.json"
    path.write_text(json.dumps(result.to_comparison_row(), indent=2))


def print_summary(results: list[ExperimentResult]) -> None:
    if not results:
        return
    print("\n" + "═" * 75)
    print(f"  TRANSFORMER RESULTS ({len(results)} models)")
    print("═" * 75)
    print(f"  {'Model':<30} {'recall_1':>8} {'prec_1':>7} {'MCC':>7} {'pass':>5}")
    print("─" * 75)
    for r in sorted(results, key=lambda x: -x.metrics.recall_1):
        flag = "✅" if r.metrics.passes_production_threshold else "❌"
        print(
            f"  {r.experiment_id:<30} "
            f"{r.metrics.recall_1:>8.4f} "
            f"{r.metrics.precision_1:>7.4f} "
            f"{r.metrics.mcc:>7.4f} "
            f"{flag:>5}"
        )
    print("═" * 75)
    best = max(results, key=lambda x: (x.metrics.passes_production_threshold, x.metrics.mcc))
    print(f"\n  Best: {best.experiment_id}  MCC={best.metrics.mcc:.4f}")
    print(f"  Checkpoint: {best.model_path}")
    print("═" * 75 + "\n")
