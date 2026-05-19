"""
Phase 7 — Full Model Evaluation.

Aggregates all trained model results, runs cost simulation, and optionally
runs ablation and error analysis.

Run this after completing phase_5 (classical) and/or phase_6 (transformers).

Usage:
    python phases/phase_7_evaluate.py --dataset v1
    python phases/phase_7_evaluate.py --dataset v1 --ablate          # test.csv comparison
    python phases/phase_7_evaluate.py --dataset v1 --error-analysis  # val.csv FN breakdown

ABLATION NOTE:
    --ablate evaluates the best model against test.csv. Run this ONCE at the end,
    after all model selection decisions have been made using val.csv.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)


def run_comparison(dataset: str) -> list:
    from backend.evaluation.comparator import ModelComparator

    comparator = ModelComparator()
    rows = comparator.compare(dataset)

    if not rows:
        log.warning("No experiment results found. Run phase_5 and/or phase_6 first.")
        return []

    print("\n" + "═" * 90)
    print("  MODEL COMPARISON")
    print("═" * 90)
    print(
        f"  {'Experiment':<35} {'Approach':<14} {'recall_1':>8} "
        f"{'prec_1':>7} {'MCC':>7} {'p50ms':>6} {'pass':>5}"
    )
    print("─" * 90)
    for r in rows:
        flag = "✅" if r.passes_production_threshold else "❌"
        print(
            f"  {r.experiment_id:<35} {r.approach:<14} {r.recall_1:>8.4f} "
            f"{r.precision_1:>7.4f} {r.mcc:>7.4f} {r.latency_p50_ms:>6.1f} {flag:>5}"
        )

    best = comparator.best_model(dataset)
    if best:
        print(
            f"\n  ✅ Best model: {best.experiment_id}  recall_1={best.recall_1:.4f}  MCC={best.mcc:.4f}"
        )
        print(f"  Saved: experiments/{dataset}/master_comparison.csv")
    print("═" * 90)
    return rows


def run_cost_simulation(dataset: str, rows: list, daily_messages: int = 10_000) -> None:
    from backend.evaluation.cost_simulator import CostSimulator

    results = CostSimulator().simulate_all(dataset, rows, daily_messages)

    print("\n" + "═" * 90)
    print(f"  COST SIMULATION  ({daily_messages:,} messages/day)")
    print("═" * 90)
    print(
        f"  {'Strategy':<40} {'% mini':>6} {'daily $':>9} "
        f"{'savings/day':>11} {'savings/mo':>11} {'recall_1':>9}"
    )
    print("─" * 90)
    for r in results[:8]:
        print(
            f"  {r.strategy_name:<40} {r.pct_routed_to_mini:>6.1%} "
            f"${r.daily_cost_usd:>8.4f} ${r.daily_savings_usd:>10.4f} "
            f"${r.monthly_savings_usd:>10.2f} {r.recall_1:>9.4f}"
        )
    print(f"\n  Saved: experiments/{dataset}/cost_simulation.json")
    print("═" * 90)


def run_ablation(dataset: str) -> None:
    from backend.evaluation.ablation import AblationRunner

    print("\nRunning ablation on test.csv (ONCE — final evaluation)...")
    AblationRunner().run(dataset)
    print(f"\n  Saved: experiments/{dataset}/ablation_results.json")


def run_error_analysis(dataset: str, model_key: str) -> None:
    from backend.evaluation.comparator import ModelComparator
    from backend.evaluation.error_analyzer import ErrorAnalyzer
    from backend.router.predictor import RouterPredictor
    from backend.shared.path_resolver import get_experiment_path

    comparator = ModelComparator()
    best = comparator.best_model(dataset)

    if not best:
        log.warning("No best model found. Run comparison first.")
        return

    experiment_id = model_key or best.experiment_id
    approach = best.approach

    if approach == "classical":
        model_path = get_experiment_path(dataset, "classical") / "models" / f"{experiment_id}.pkl"
        predictor = RouterPredictor.from_pkl(model_path)
    else:
        checkpoint_dir = get_experiment_path(dataset, "transformers") / "models" / experiment_id
        predictor = RouterPredictor.from_hf_checkpoint(checkpoint_dir)

    print(f"\nRunning error analysis with {experiment_id} on val.csv...")
    report = ErrorAnalyzer().analyze(dataset, predictor)

    print(
        f"\n  False negatives: {report.total_fn} / {report.total_examples} ({report.fn_rate:.1%})"
    )
    print(f"  Worst scenario:  {report.worst_scenario}")
    print(f"  Worst language:  {report.worst_language}")
    print("\n  Sample false negatives:")
    for err in report.sample_errors[:5]:
        print(f"    [{err['language']} / {err['scenario']}] {err['text'][:80]}")
    print(f"\n  Saved: experiments/{dataset}/error_analysis.json")


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
        help="Evaluate best model on test.csv (run once, final evaluation)",
    )
    parser.add_argument(
        "--error-analysis", action="store_true", help="Run false-negative breakdown on val.csv"
    )
    parser.add_argument(
        "--model", default=None, help="Model key for error analysis (defaults to best model)"
    )
    args = parser.parse_args()

    rows = run_comparison(args.dataset)

    if rows:
        run_cost_simulation(args.dataset, rows, args.daily_messages)

    if args.ablate:
        run_ablation(args.dataset)

    if args.error_analysis:
        run_error_analysis(args.dataset, args.model)


if __name__ == "__main__":
    main()
