"""
Phase 7 — Full Model Evaluation.

Run after phase_5 (classical) and/or phase_6 (transformers).
Error analysis runs automatically — no flag needed.

Usage:
    python phases/phase_7_evaluate.py --dataset v1
    python phases/phase_7_evaluate.py --dataset v1 --json      # machine-readable JSON
    python phases/phase_7_evaluate.py --dataset v1 --ablate    # test.csv (ONCE, final eval)
    python phases/phase_7_evaluate.py --dataset v1 --model tfidf_combined__svm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)


def run_comparison(dataset: str, as_json: bool = False) -> list:
    from backend.evaluation.comparator import ModelComparator

    comparator = ModelComparator()
    rows = comparator.compare(dataset)

    if not rows:
        log.warning("No experiment results found. Run phase_5 and/or phase_6 first.")
        return []

    if as_json:
        print(json.dumps([r.model_dump() for r in rows], indent=2))
    else:
        print("\n" + "═" * 90)
        print("  MODEL COMPARISON")
        print("═" * 90)
        print(
            f"  {'Experiment':<38} {'Approach':<14} {'recall_1':>8} "
            f"{'prec_1':>7} {'MCC':>7} {'p50ms':>6} {'pass':>5}"
        )
        print("─" * 90)
        for r in rows:
            flag = "✅" if r.passes_production_threshold else "❌"
            print(
                f"  {r.experiment_id:<38} {r.approach:<14} {r.recall_1:>8.4f} "
                f"{r.precision_1:>7.4f} {r.mcc:>7.4f} {r.latency_p50_ms:>6.1f} {flag:>5}"
            )

        best = comparator.best_model(dataset)
        if best:
            print(
                f"\n  {'✅' if best.passes_production_threshold else '⚠'} "
                f"Best: {best.experiment_id}  recall_1={best.recall_1:.4f}  MCC={best.mcc:.4f}"
            )
        print("═" * 90)

    return rows


def run_cost_simulation(
    dataset: str,
    rows: list,
    daily_messages: int = 10_000,
    as_json: bool = False,
) -> None:
    from backend.evaluation.cost_simulator import CostSimulator

    results = CostSimulator().simulate_all(dataset, rows, daily_messages)
    if as_json:
        print(json.dumps([r.model_dump() for r in results], indent=2))
    else:
        print("\n" + "═" * 90)
        print(f"  COST SIMULATION  ({daily_messages:,} messages/day)")
        print("═" * 90)
        print(
            f"  {'Strategy':<42} {'% mini':>6} {'daily $':>9} "
            f"{'savings/day':>12} {'savings/mo':>11} {'recall_1':>9}"
        )
        print("─" * 90)
        for r in results[:10]:
            print(
                f"  {r.strategy_name:<42} {r.pct_routed_to_mini:>6.1%} "
                f"${r.daily_cost_usd:>8.4f} ${r.daily_savings_usd:>11.4f} "
                f"${r.monthly_savings_usd:>10.2f} {r.recall_1:>9.4f}"
            )
        print("═" * 90)


def run_error_analysis(dataset: str, model_key: str | None, as_json: bool = False) -> None:
    """Runs automatically after comparison. Shows false negative breakdown."""
    from backend.evaluation.comparator import ModelComparator
    from backend.evaluation.error_analyzer import ErrorAnalyzer
    from backend.router.predictor import RouterPredictor
    from backend.shared.path_resolver import get_experiment_path

    comparator = ModelComparator()
    best = comparator.best_model(dataset)

    if not best:
        log.warning("No best model found — skipping error analysis.")
        return

    experiment_id = model_key or best.experiment_id
    approach = best.approach

    if approach == "classical":
        model_path = get_experiment_path(dataset, "classical") / "models" / experiment_id
        predictor = RouterPredictor.from_pkl(model_path)
    else:
        ckpt_dir = get_experiment_path(dataset, "transformers") / "models" / experiment_id
        predictor = RouterPredictor.from_hf_checkpoint(ckpt_dir)

    log.info(f"Error analysis: {experiment_id} on val.csv")
    report = ErrorAnalyzer().analyze(dataset, predictor)

    if as_json:
        # ErrorReport may or may not be a Pydantic model; vars() works for both
        data: dict = vars(report)
        print(json.dumps(data, indent=2, default=str))
    else:
        print("\n" + "═" * 90)
        print(f"  ERROR ANALYSIS  (model: {experiment_id})")
        print("═" * 90)
        print(
            f"  False negatives: {report.total_fn} / {report.total_examples} ({report.fn_rate:.1%})"
        )
        print(f"  Worst scenario:  {report.worst_scenario}")
        print(f"  Worst language:  {report.worst_language}")
        print("\n  Sample false negatives (label=1 routed to gpt-4o-mini):")
        for err in report.sample_errors[:5]:
            print(f"    [{err['language']} / {err['scenario']}] {err['text'][:80]}")
        print("═" * 90)


def run_ablation(dataset: str) -> None:
    from backend.evaluation.ablation import AblationRunner

    print("\nRunning ablation on test.csv (ONCE — final evaluation only)...")
    AblationRunner().run(dataset)
    print(f"  Saved: experiments/{dataset}/ablation_results.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="v1")
    parser.add_argument("--daily-messages", type=int, default=10_000)
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="Run ablation on test.csv (final eval only — use once)",
    )
    parser.add_argument(
        "--model", default=None, help="Specific model for error analysis (default: best model)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print results as pretty JSON instead of tables"
    )
    args = parser.parse_args()

    rows = run_comparison(args.dataset, as_json=args.json)

    if rows:
        run_cost_simulation(args.dataset, rows, args.daily_messages, as_json=args.json)
        run_error_analysis(args.dataset, args.model, as_json=args.json)

    if args.ablate:
        run_ablation(args.dataset)


if __name__ == "__main__":
    main()
