"""
Evaluation metrics for the binary router classifier.

All metrics use sklearn under the hood. The key metric for this project is
recall(label=1) >= 0.97 — we must never send a complex/code-mixed message
to gpt-4o-mini. False negatives are more costly than false positives.

See docs/05_EVALUATION_PLAN.md for full metric rationale.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_all_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_proba: list[float] | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics for a binary classifier.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        y_proba: Predicted probabilities for label=1 (optional).
                 Required for ROC-AUC, PR-AUC, log loss, ECE.

    Returns:
        Dict of metric_name → float value.

    Key metrics to watch:
        recall_1        ← MUST be >= 0.97 in production
        precision_1     ← want this high to avoid unnecessary gpt-4o routing
        mcc             ← summary statistic: 1.0 = perfect, 0.0 = random
        roc_auc         ← discrimination ability (requires y_proba)
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision_0": float(precision_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)),
        "precision_1": float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)),
        "recall_0": float(recall_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)),
        "recall_1": float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred_arr)),
    }

    if y_proba is not None:
        y_proba_arr = np.array(y_proba)
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_proba_arr))
        metrics["pr_auc"] = float(average_precision_score(y_true_arr, y_proba_arr))
        metrics["log_loss"] = float(log_loss(y_true_arr, y_proba_arr))
        metrics["ece"] = float(_compute_ece(y_true_arr, y_proba_arr))

    return metrics


def compute_latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    """Compute latency percentile statistics.

    Args:
        latencies_ms: List of inference latencies in milliseconds.

    Returns:
        Dict with mean, p50, p95, p99 in milliseconds.
    """
    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
    }


def estimate_daily_cost(
    n_queries: int,
    label1_ratio: float,
    gpt4o_input_per_1m: float,
    gpt4o_mini_input_per_1m: float,
    avg_input_tokens: int = 200,
    avg_output_tokens: int = 150,
    gpt4o_output_per_1m: float = 10.00,
    gpt4o_mini_output_per_1m: float = 0.60,
) -> dict[str, float]:
    """Estimate daily API cost for a given routing strategy.

    Args:
        n_queries: Daily message volume.
        label1_ratio: Fraction routed to gpt-4o (0.0–1.0).
        gpt4o_input_per_1m: gpt-4o input price per 1M tokens.
        gpt4o_mini_input_per_1m: gpt-4o-mini input price per 1M tokens.
        avg_input_tokens: Average input tokens per request.
        avg_output_tokens: Average output tokens per request.

    Returns:
        Dict with cost breakdown and totals.
    """
    n_gpt4o = n_queries * label1_ratio
    n_mini = n_queries * (1 - label1_ratio)

    def cost(n: float, input_per_1m: float, output_per_1m: float) -> float:
        input_cost = n * avg_input_tokens / 1_000_000 * input_per_1m
        output_cost = n * avg_output_tokens / 1_000_000 * output_per_1m
        return input_cost + output_cost

    gpt4o_cost = cost(n_gpt4o, gpt4o_input_per_1m, gpt4o_output_per_1m)
    mini_cost = cost(n_mini, gpt4o_mini_input_per_1m, gpt4o_mini_output_per_1m)
    total_cost = gpt4o_cost + mini_cost

    all_gpt4o_cost = cost(n_queries, gpt4o_input_per_1m, gpt4o_output_per_1m)
    savings = all_gpt4o_cost - total_cost

    return {
        "daily_cost_usd": round(total_cost, 4),
        "gpt4o_cost_usd": round(gpt4o_cost, 4),
        "mini_cost_usd": round(mini_cost, 4),
        "all_gpt4o_cost_usd": round(all_gpt4o_cost, 4),
        "savings_usd": round(savings, 4),
        "savings_pct": round(savings / max(all_gpt4o_cost, 1e-9) * 100, 1),
        "label1_ratio": label1_ratio,
        "n_queries": n_queries,
    }


def print_metrics_report(
    metrics: dict[str, float],
    model_name: str = "model",
) -> None:
    """Pretty-print a metrics report to stdout."""
    print(f"\n{'═' * 50}")
    print(f"  {model_name}")
    print(f"{'═' * 50}")

    key_metrics = [
        ("recall_1", "Recall (label=1)  ← MUST >= 0.97"),
        ("precision_1", "Precision (label=1)"),
        ("recall_0", "Recall (label=0)"),
        ("precision_0", "Precision (label=0)"),
        ("accuracy", "Accuracy"),
        ("mcc", "MCC"),
        ("f1_macro", "F1 Macro"),
    ]
    for key, label in key_metrics:
        if key in metrics:
            flag = " ⚠️" if key == "recall_1" and metrics[key] < 0.97 else ""
            print(f"  {label:<35} {metrics[key]:.4f}{flag}")

    optional = [
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("log_loss", "Log Loss"),
        ("ece", "ECE (calibration)"),
    ]
    for key, label in optional:
        if key in metrics:
            print(f"  {label:<35} {metrics[key]:.4f}")

    print(f"{'═' * 50}\n")


# ── Internal helpers ──────────────────────────────────────────────────────────


def _compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error — how well predicted probabilities match reality."""
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )
    bin_sizes = np.histogram(y_proba, bins=n_bins, range=(0, 1))[0]
    total = y_true.shape[0]
    ece = float(
        np.sum(
            np.abs(fraction_of_positives - mean_predicted)
            * bin_sizes[: len(fraction_of_positives)]
            / total
        )
    )
    return ece


def time_inference(
    predict_fn: Any,
    texts: list[str],
    n_warmup: int = 10,
) -> tuple[list[int], dict[str, float]]:
    """Time inference latency for a predict function.

    Args:
        predict_fn: Callable that takes a list[str] and returns list[int].
        texts: Test texts to predict on.
        n_warmup: Number of warmup calls before timing.

    Returns:
        (predictions, latency_stats_dict)
    """
    for _ in range(n_warmup):
        predict_fn(texts[:1])

    latencies: list[float] = []
    all_preds: list[int] = []

    for text in texts:
        start = time.perf_counter()
        pred = predict_fn([text])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        all_preds.extend(pred)

    return all_preds, compute_latency_stats(latencies)
