"""
Phase 5 — Classical ML Training.

Trains all (vectorizer, classifier) combos in ACTIVE_COMBOS.
All training warnings are suppressed.

Usage:
    python phases/phase_5_train_classical.py --all
    python phases/phase_5_train_classical.py --vectorizer tfidf_combined --classifier svm
    python phases/phase_5_train_classical.py --all --hpo --n-trials 20
    python phases/phase_5_train_classical.py --dataset v2 --all
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

# Suppress before any sklearn/mlflow/lightgbm imports
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="v1")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--vectorizer", default=None)
    parser.add_argument("--classifier", default=None)
    parser.add_argument("--hpo", action="store_true")
    parser.add_argument("--n-trials", type=int, default=10)
    args = parser.parse_args()

    from backend.training.classical.trainer import ClassicalMLTrainer

    trainer = ClassicalMLTrainer()

    if args.all:
        trainer.train_all_combos(args.dataset)

    elif args.vectorizer and args.classifier:
        if args.hpo:
            from backend.training.classical.hpo import ClassicalHPORunner

            # HPOResult has .best_params (dict) and .best_value (float)
            hpo_result = ClassicalHPORunner().run(
                args.dataset,
                args.vectorizer,
                args.classifier,
                n_trials=args.n_trials,
            )
            trainer.train_experiment(
                args.dataset,
                args.vectorizer,
                args.classifier,
                classifier_params=hpo_result.best_params,
            )
        else:
            trainer.train_experiment(args.dataset, args.vectorizer, args.classifier)

    else:
        parser.error("Provide --all or both --vectorizer and --classifier")


if __name__ == "__main__":
    main()
