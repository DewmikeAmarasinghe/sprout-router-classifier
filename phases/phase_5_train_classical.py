"""
Phase 5 — Classical ML Training.

Trains all (vectorizer, classifier) combinations and logs results to MLflow.
val.csv used for evaluation. test.csv NOT accessed here.

Usage:
    python phases/phase_5_train_classical.py                          # all combos
    python phases/phase_5_train_classical.py --vec tfidf_combined --clf lightgbm
    python phases/phase_5_train_classical.py --all --dataset v1

Available vectorizers:  tfidf_char, tfidf_word, tfidf_combined, word2vec, spacy
Available classifiers:  logistic_regression, svm, lightgbm, xgboost, catboost

Recommended order:
    1. Run --all (TF-IDF combos are fast, ~2 min each)
    2. Check mlflow ui to see which combo has best recall_1 + MCC
    3. Optionally run individual combos with custom params

For spaCy vectorizer:
    python -m spacy download en_core_web_md   (run once)

After this, run:
    python phases/phase_6_train_transformers.py

Key metric: recall_1 >= 0.97 is required for production deployment.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="v1", help="Dataset version")
    parser.add_argument("--vec", default=None, help="Vectorizer key")
    parser.add_argument("--clf", default=None, help="Classifier key")
    parser.add_argument("--all", action="store_true", help="Run all ACTIVE_COMBOS")
    parser.add_argument(
        "--list", action="store_true", help="List available vectorizers and classifiers"
    )
    args = parser.parse_args()

    from backend.training.classical.config import (
        ACTIVE_COMBOS,
        CLASSIFIER_REGISTRY,
        VECTORIZER_REGISTRY,
    )
    from backend.training.classical.trainer import ClassicalMLTrainer

    if args.list:
        print("\nVectorizers:")
        for k, v in VECTORIZER_REGISTRY.items():
            print(f"  {k:<20} {v.display_name}")
        print("\nClassifiers:")
        for k, v in CLASSIFIER_REGISTRY.items():
            print(f"  {k:<25} {v.display_name}")
        print(f"\nActive combos ({len(ACTIVE_COMBOS)}):")
        for vec, clf in ACTIVE_COMBOS:
            print(f"  {vec} + {clf}")
        return

    trainer = ClassicalMLTrainer()

    if args.all or (not args.vec and not args.clf):
        log.info(f"Running all {len(ACTIVE_COMBOS)} active combos on dataset '{args.dataset}'")
        trainer.train_all_combos(args.dataset)

    elif args.vec and args.clf:
        log.info(f"Running: {args.vec} + {args.clf} on dataset '{args.dataset}'")
        result = trainer.train_experiment(args.dataset, args.vec, args.clf)
        print(f"\nResult: {result.metrics.summary_line()}")
        print(f"Model:  {result.model_path}")

    else:
        parser.error("Provide both --vec and --clf, or use --all, or use --list")


if __name__ == "__main__":
    main()
