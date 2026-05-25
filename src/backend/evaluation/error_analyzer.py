"""
ErrorAnalyzer — understand where the model fails.

Analyzes false negatives (label=1 predicted as 0) grouped by scenario and language.
Run this on val.csv (NOT test.csv) during development to guide:
  - Which scenarios need more training data
  - Whether to raise the confidence threshold for specific scenario types
  - Which edge cases the script_detector should handle instead

Usage:
    analyzer = ErrorAnalyzer()
    report   = analyzer.analyze("v1", predictor)
    print(report.worst_scenario)    # e.g. "continuation"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from backend.shared.path_resolver import get_dataset_path, get_experiment_path

log = logging.getLogger(__name__)


@dataclass
class ErrorReport:
    """Summary of false negative analysis."""

    total_fn: int
    total_examples: int
    fn_rate: float
    by_scenario: dict[str, dict]
    by_language: dict[str, dict]
    worst_scenario: str
    worst_language: str
    sample_errors: list[dict]


class ErrorAnalyzer:
    """Groups false negatives from val.csv by scenario and language."""

    def analyze(
        self,
        dataset_name: str,
        predictor: Any,
        max_samples: int = 20,
    ) -> ErrorReport:
        """Run error analysis on val.csv.

        Args:
            dataset_name: e.g. "v1".
            predictor: A RouterPredictor instance.
            max_samples: Max false negative examples to include in the report.

        Returns:
            ErrorReport with breakdowns by scenario and language.
        """
        val_path = get_dataset_path(dataset_name) / "val.csv"
        if not val_path.exists():
            raise FileNotFoundError(f"val.csv not found at {val_path}. Run phase_3_split.py first.")

        val_df = pd.read_csv(val_path)
        texts = val_df["text"].tolist()

        predictions = [predictor.predict(t) for t in texts]
        val_df["predicted"] = [p.label for p in predictions]
        val_df["confidence"] = [round(p.confidence, 4) for p in predictions]

        false_negatives = val_df[(val_df["label"] == 1) & (val_df["predicted"] == 0)].copy()

        total_fn = len(false_negatives)
        total_label_1 = int((val_df["label"] == 1).sum())
        fn_rate = total_fn / total_label_1 if total_label_1 > 0 else 0.0

        log.info(
            f"Error analysis: {total_fn} false negatives ({fn_rate:.1%} of label=1) in val set"
        )

        by_scenario = group_by_column(false_negatives, val_df, "scenario")
        by_language = group_by_column(false_negatives, val_df, "language")

        worst_scenario = max(by_scenario, key=lambda k: by_scenario[k]["rate"], default="")
        worst_language = max(by_language, key=lambda k: by_language[k]["rate"], default="")

        sample_errors = (
            false_negatives[["text", "language", "scenario", "label", "predicted", "confidence"]]
            .head(max_samples)
            .to_dict("records")
        )

        report = ErrorReport(
            total_fn=total_fn,
            total_examples=total_label_1,
            fn_rate=round(fn_rate, 4),
            by_scenario=by_scenario,
            by_language=by_language,
            worst_scenario=worst_scenario,
            worst_language=worst_language,
            sample_errors=sample_errors,
        )

        save_report(report, dataset_name)
        return report


def group_by_column(
    false_negatives: pd.DataFrame,
    full_df: pd.DataFrame,
    column: str,
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for value, group_fn in false_negatives.groupby(column):
        total_in_group = int((full_df[full_df["label"] == 1][column] == value).sum())
        count = len(group_fn)
        rate = count / total_in_group if total_in_group > 0 else 0.0
        result[str(value)] = {
            "false_negatives": count,
            "total_label_1": total_in_group,
            "rate": round(rate, 4),
        }
    return dict(sorted(result.items(), key=lambda kv: -kv[1]["rate"]))


def save_report(report: ErrorReport, dataset_name: str) -> None:
    output_dir = get_experiment_path(dataset_name, "classical").parent
    path = output_dir / "error_analysis.json"
    path.write_text(
        json.dumps(
            {
                "total_fn": report.total_fn,
                "total_examples": report.total_examples,
                "fn_rate": report.fn_rate,
                "worst_scenario": report.worst_scenario,
                "worst_language": report.worst_language,
                "by_scenario": report.by_scenario,
                "by_language": report.by_language,
                "sample_errors": report.sample_errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(f"Saved error analysis: {path}")
