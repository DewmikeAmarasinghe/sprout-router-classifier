"""
Phase 7 — Full Model Evaluation.

Run after phase_5 (classical) and/or phase_6 (transformers).

Usage:
    python phases/phase_7_evaluate.py --dataset v1
    python phases/phase_7_evaluate.py --dataset v1 --json      # machine-readable JSON
    python phases/phase_7_evaluate.py --dataset v1 --json | head -100
    python phases/phase_7_evaluate.py --dataset v1 --ablate    # test.csv (ONCE, final eval)
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
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

# Handle SIGPIPE gracefully (e.g. `python phase_7_evaluate.py --json | head -100`).
# Without this, Python raises BrokenPipeError when the downstream process (head)
# closes the pipe before we finish writing. SIG_DFL restores the OS default
# behaviour — silent exit — which is the expected shell behaviour.
if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


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
        W = 115
        print("\n" + "═" * W)
        print("  MODEL COMPARISON")
        print("═" * W)
        print(
            f"  {'Experiment':<38} {'Approach':<12} "
            f"{'rec_1':>6} {'prec_1':>6} {'rec_0':>6} {'MCC':>6} "
            f"{'roc_auc':>8} {'f1':>6} {'log_loss':>9} {'p50ms':>6} {'p95ms':>6} {'pass':>5}"
        )
        print("─" * W)
        for r in rows:
            flag = "✅" if r.passes_production_threshold else "❌"
            print(
                f"  {r.experiment_id:<38} {r.approach:<12} "
                f"{r.recall_1:>6.4f} {r.precision_1:>6.4f} {r.recall_0:>6.4f} {r.mcc:>6.4f} "
                f"{r.roc_auc:>8.4f} {r.f1_macro:>6.4f} {r.log_loss:>9.4f} "
                f"{r.latency_p50_ms:>6.1f} {r.latency_p95_ms:>6.1f} {flag:>5}"
            )

        best = comparator.best_model(dataset)
        if best:
            print(
                f"\n  {'✅' if best.passes_production_threshold else '⚠'} "
                f"Best: {best.experiment_id}  recall_1={best.recall_1:.4f}  MCC={best.mcc:.4f}"
            )
        print("═" * W)

    return rows


def run_cost_simulation(
    dataset: str,
    rows: list,
    daily_messages: int | None = None,
    as_json: bool = False,
) -> None:
    from backend.evaluation.cost_simulator import CostSimulator
    from backend.shared.settings_manager import settings_manager

    n_messages = (
        daily_messages
        if daily_messages is not None
        else int(settings_manager.get("DAILY_MESSAGES_ESTIMATE"))
    )
    results = CostSimulator().simulate_all(dataset, rows, n_messages)
    if as_json:
        print(json.dumps([r.model_dump() for r in results], indent=2))
        return

    baseline = next((r for r in results if r.strategy_name == "all_gpt4o"), None)
    router_rows = [r for r in results if r.strategy_name.startswith("router_")]

    print("\n" + "═" * 90)
    print(f"  COST SIMULATION  ({n_messages:,} messages/day)  — savings vs all-gpt-4o baseline")
    print("═" * 90)
    print(
        f"  {'Strategy':<42} {'% mini':>6} {'daily $':>9} "
        f"{'savings/day':>12} {'savings/mo':>11} {'recall_1':>9}"
    )
    print("─" * 90)
    if baseline:
        print(
            f"  {'all_gpt4o (current baseline)':<42} {baseline.pct_routed_to_mini:>6.1%} "
            f"${baseline.daily_cost_usd:>8.4f} ${'0.0000':>11} ${'0.00':>10} {'—':>9}"
        )
        print("─" * 90)
    for r in router_rows[:10]:
        print(
            f"  {r.strategy_name:<42} {r.pct_routed_to_mini:>6.1%} "
            f"${r.daily_cost_usd:>8.4f} ${r.daily_savings_usd:>11.4f} "
            f"${r.monthly_savings_usd:>10.2f} {r.recall_1:>9.4f}"
        )
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
    parser.add_argument(
        "--daily-messages",
        type=int,
        default=None,
        help="Override daily message volume (default: DAILY_MESSAGES_ESTIMATE from settings.py)",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="Run ablation on test.csv (final eval only — use once)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print results as pretty JSON instead of tables"
    )
    args = parser.parse_args()

    rows = run_comparison(args.dataset, as_json=args.json)

    if rows:
        run_cost_simulation(args.dataset, rows, args.daily_messages, as_json=args.json)

    if args.ablate:
        run_ablation(args.dataset)


if __name__ == "__main__":
    main()
