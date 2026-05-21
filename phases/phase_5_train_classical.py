"""
Phase 5 — Classical ML Training.

Trains all (vectorizer, classifier) combos in ACTIVE_COMBOS.
After --all completes, automatically runs HPO (Optuna, n_trials=10) on the best model
and retrains it with the tuned hyperparameters.

HPO uses the val set — it NEVER touches test.csv.

Usage:
    python phases/phase_5_train_classical.py --all
    python phases/phase_5_train_classical.py --all --no-hpo          # skip auto-HPO
    python phases/phase_5_train_classical.py --all --n-trials 20     # more HPO trials
    python phases/phase_5_train_classical.py --vectorizer tfidf_combined --classifier svm
    python phases/phase_5_train_classical.py --dataset v2 --all
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)


def run_auto_hpo(dataset: str, results: list, n_trials: int) -> None:
    """Run Optuna HPO on the best model from train_all_combos().

    HOW HPO WORKS:
      Optuna runs n_trials iterations. Each trial:
        1. Samples hyperparameters (C, learning_rate, n_estimators, etc.)
           from a defined search space for the classifier.
        2. Trains the (vectorizer, classifier) pair on train.csv.
        3. Evaluates on val.csv → returns MCC as the objective.
        4. MedianPruner: if a trial's result is below the median of completed
           trials, it's flagged (though for classical ML this is instantaneous,
           so pruning has minimal effect — all trials complete in seconds).

      After n_trials, Optuna picks the best hyperparameters and retrains
      the model with them. The retrained model overwrites the experiment dir.

    This does NOT use test.csv. Only val.csv is used for evaluation.
    """
    if not results:
        return

    best_initial = max(results, key=lambda r: r.metrics.recall_1)
    log.info(
        f"Auto-HPO: targeting {best_initial.experiment_id}  "
        f"(initial recall_1={best_initial.metrics.recall_1:.4f})  "
        f"n_trials={n_trials}"
    )

    from backend.training.classical.hpo import ClassicalHPORunner
    from backend.training.classical.trainer import ClassicalMLTrainer

    hpo_result = ClassicalHPORunner().run(
        dataset_name=dataset,
        vectorizer_key=best_initial.vectorizer_key,
        classifier_key=best_initial.classifier_key,
        n_trials=n_trials,
        optimize_metric="mcc",
    )

    log.info(f"HPO best params: {hpo_result.best_params}")
    log.info("Retraining with tuned hyperparameters...")

    tuned = ClassicalMLTrainer().train_experiment(
        dataset,
        best_initial.vectorizer_key,
        best_initial.classifier_key,
        classifier_params=hpo_result.best_params,
    )

    improvement = tuned.metrics.recall_1 - best_initial.metrics.recall_1
    sign = "+" if improvement >= 0 else ""
    log.info(
        f"HPO result: recall_1={tuned.metrics.recall_1:.4f}  "
        f"({sign}{improvement:.4f} vs initial)  MCC={tuned.metrics.mcc:.4f}"
    )
    if tuned.metrics.passes_production_threshold:
        log.info("✅ Model now passes production threshold (recall_1 ≥ 0.97)")
    else:
        log.warning("⚠ Still below 0.97 after HPO. Consider more trials or training transformers.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="v1")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--vectorizer", default=None)
    parser.add_argument("--classifier", default=None)
    parser.add_argument(
        "--no-hpo",
        action="store_true",
        help="Skip automatic HPO after --all (HPO runs by default)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of Optuna HPO trials (default 10, ~5 min for classical)",
    )
    args = parser.parse_args()

    from backend.training.classical.trainer import ClassicalMLTrainer

    trainer = ClassicalMLTrainer()

    if args.all:
        results = trainer.train_all_combos(args.dataset)
        if not args.no_hpo and results:
            run_auto_hpo(args.dataset, results, args.n_trials)

    elif args.vectorizer and args.classifier:
        trainer.train_experiment(args.dataset, args.vectorizer, args.classifier)

    else:
        parser.error("Provide --all, or both --vectorizer and --classifier")


if __name__ == "__main__":
    main()
