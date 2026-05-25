"""
ThresholdTuner — find the optimal confidence threshold.

STRATEGY:
    Sweep thresholds [0.30, 0.35, ..., 0.90].
    For each: compute recall_1, precision_1, accuracy on val.csv.
    Return the HIGHEST threshold that still satisfies recall_1 >= min_recall_1.

    Higher threshold = more messages routed to gpt-4o (safer, more expensive).
    Lower threshold  = more messages routed to gpt-4o-mini (cheaper, riskier).

    Default target: recall_1 >= PRODUCTION_RECALL_THRESHOLD. At most 3% of label=1 messages get
    incorrectly sent to gpt-4o-mini — acceptable UX degradation.

OUTPUT:
    threshold_curve.json in experiments/{dataset}/router/

Usage:
    tuner   = ThresholdTuner()
    result  = tuner.find_optimal_threshold("v1", predictor, min_recall_1=PRODUCTION_RECALL_THRESHOLD)
    print(f"Optimal threshold: {result.optimal_threshold}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backend.shared.path_resolver import get_dataset_path, get_experiment_path
from backend.shared.settings_manager import settings_manager

log = logging.getLogger(__name__)

THRESHOLD_SWEEP = [round(t, 2) for t in np.arange(0.30, 0.91, 0.05).tolist()]


@dataclass
class ThresholdResult:
    """Result of one threshold sweep."""

    optimal_threshold: float
    optimal_recall_1: float
    optimal_precision_1: float
    optimal_accuracy: float
    min_recall_1_target: float
    curve: list[dict] = field(default_factory=list)
    output_path: str = ""


class ThresholdTuner:
    """Sweeps confidence thresholds and finds the optimal routing cutoff."""

    def find_optimal_threshold(
        self,
        dataset_name: str,
        predictor: Any,
        min_recall_1: float = float(settings_manager.get("PRODUCTION_RECALL_THRESHOLD")),
        safe_default_label: int = 1,
    ) -> ThresholdResult:
        """Sweep thresholds on val.csv and return the best one.

        Args:
            dataset_name: Dataset version, e.g. "v1".
            predictor: A RouterPredictor instance with a trained model.
            min_recall_1: Minimum acceptable recall for label=1 (default PRODUCTION_RECALL_THRESHOLD).
            safe_default_label: Label used when confidence < threshold.

        Returns:
            ThresholdResult with optimal_threshold and full curve data.
        """
        val_path = get_dataset_path(dataset_name) / "val.csv"
        if not val_path.exists():
            raise FileNotFoundError(f"val.csv not found at {val_path}. Run phase_3_split.py first.")

        val_df = pd.read_csv(val_path)
        texts = val_df["text"].tolist()
        labels = val_df["label"].astype(int).tolist()

        log.info(f"Sweeping {len(THRESHOLD_SWEEP)} thresholds on {len(val_df):,} val rows")

        # Get raw confidence scores — bypass script_detector for the sweep
        # (script_detector rows are always routed correctly regardless of threshold)
        confidences: list[float] = [predictor._get_confidence(t) for t in texts]

        curve: list[dict] = []
        for threshold in THRESHOLD_SWEEP:
            preds = [
                (1 if conf >= 0.5 else 0) if conf >= threshold else safe_default_label
                for conf in confidences
            ]
            row = compute_threshold_metrics(labels, preds, threshold)
            curve.append(row)
            log.debug(
                f"  threshold={threshold:.2f}  recall_1={row['recall_1']:.4f}  "
                f"precision_1={row['precision_1']:.4f}  accuracy={row['accuracy']:.4f}"
            )

        candidates = [r for r in curve if r["recall_1"] >= min_recall_1]
        if not candidates:
            log.warning(
                f"No threshold achieves recall_1 >= {min_recall_1:.2f}. "
                "Using the threshold with highest recall_1."
            )
            best = max(curve, key=lambda r: (r["recall_1"], r["precision_1"]))
        else:
            best = max(candidates, key=lambda r: (r["threshold"], r["precision_1"]))

        output_dir = get_experiment_path(dataset_name, "router")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "threshold_curve.json"

        output_data = {
            "optimal_threshold": best["threshold"],
            "optimal_recall_1": best["recall_1"],
            "optimal_precision_1": best["precision_1"],
            "optimal_accuracy": best["accuracy"],
            "min_recall_1_target": min_recall_1,
            "val_rows": len(val_df),
            "curve": curve,
        }
        output_path.write_text(json.dumps(output_data, indent=2))

        log.info(
            f"Optimal threshold: {best['threshold']}  "
            f"recall_1={best['recall_1']:.4f}  precision_1={best['precision_1']:.4f}"
        )
        log.info(f"Saved to: {output_path}")

        return ThresholdResult(
            optimal_threshold=best["threshold"],
            optimal_recall_1=best["recall_1"],
            optimal_precision_1=best["precision_1"],
            optimal_accuracy=best["accuracy"],
            min_recall_1_target=min_recall_1,
            curve=curve,
            output_path=str(output_path),
        )

    def load_curve(self, dataset_name: str) -> dict:
        """Load previously saved threshold curve."""
        path = get_experiment_path(dataset_name, "router") / "threshold_curve.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No threshold curve found at {path}. Run find_optimal_threshold() first."
            )
        return json.loads(path.read_text())


def compute_threshold_metrics(
    labels: list[int],
    preds: list[int],
    threshold: float,
) -> dict:
    """Compute recall_1, precision_1, accuracy for one threshold setting."""
    tp = sum(1 for t, p in zip(labels, preds, strict=True) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(labels, preds, strict=True) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(labels, preds, strict=True) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(labels, preds, strict=True) if t == 0 and p == 0)

    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0

    return {
        "threshold": threshold,
        "recall_1": round(recall_1, 4),
        "precision_1": round(precision_1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
