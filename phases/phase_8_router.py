"""
Phase 8 — Router Threshold Tuning.

Sweeps confidence thresholds [0.30–0.90] on val.csv using the best trained model
and finds the highest threshold that keeps recall_1 >= 0.95.

The optimal threshold is saved to experiments/{dataset}/router/threshold_curve.json.
Copy the optimal_threshold value to CONFIDENCE_THRESHOLD in config/settings.py.

Usage:
    python phases/phase_8_router.py --dataset v1
    python phases/phase_8_router.py --dataset v1 --model xlmr-base
    python phases/phase_8_router.py --dataset v1 --test "nearest branch to me"
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


def load_predictor(dataset: str, model_key: str | None) -> tuple:
    """Load the specified or best available RouterPredictor.

    Returns (predictor: RouterPredictor, experiment_id: str).
    Typed as tuple to avoid circular import at module level — both elements
    are well-typed inside the function body.

    Classical models are saved as a directory:
        experiments/{dataset}/classical/models/{experiment_id}/model.pkl
    RouterPredictor.from_pkl() accepts the directory path and finds model.pkl inside.
    """
    from backend.evaluation.comparator import ModelComparator
    from backend.router.predictor import RouterPredictor
    from backend.shared.path_resolver import get_experiment_path

    comparator = ModelComparator()
    best = comparator.best_model(dataset)

    if not best:
        raise RuntimeError("No trained models found. Run phase_5 or phase_6 first.")

    experiment_id = model_key or best.experiment_id
    approach = best.approach if not model_key else infer_approach(dataset, experiment_id)

    if approach == "classical":
        # Pass the directory — from_pkl resolves model.pkl inside it.
        model_path = get_experiment_path(dataset, "classical") / "models" / experiment_id
        predictor = RouterPredictor.from_pkl(model_path)
    else:
        checkpoint_dir = get_experiment_path(dataset, "transformers") / "models" / experiment_id
        predictor = RouterPredictor.from_hf_checkpoint(checkpoint_dir)

    return predictor, experiment_id


def infer_approach(dataset: str, experiment_id: str) -> str:
    from backend.evaluation.comparator import ModelComparator

    rows = ModelComparator().compare(dataset)
    for row in rows:
        if row.experiment_id == experiment_id:
            return row.approach
    return "classical"


def run_threshold_tuning(dataset: str, predictor: object, experiment_id: str) -> float:
    from backend.router.threshold_tuner import ThresholdTuner

    print(f"\nSweeping thresholds on val.csv with {experiment_id}...")
    result = ThresholdTuner().find_optimal_threshold(dataset, predictor)

    print("\n" + "═" * 70)
    print("  THRESHOLD TUNING RESULTS")
    print("═" * 70)
    print(f"  Model:             {experiment_id}")
    print(f"  Optimal threshold: {result.optimal_threshold}")
    print(f"  recall_1:          {result.optimal_recall_1:.4f}")
    print(f"  precision_1:       {result.optimal_precision_1:.4f}")
    print(f"  accuracy:          {result.optimal_accuracy:.4f}")
    print(f"  Target:            recall_1 >= {result.min_recall_1_target:.2f}")
    print("═" * 70)
    print(f"\n  ► Set CONFIDENCE_THRESHOLD = {result.optimal_threshold} in config/settings.py")
    print(f"  Saved: {result.output_path}")
    return result.optimal_threshold


def run_test_prediction(predictor: object, message: str, threshold: float) -> None:
    from backend.router.predictor import RouterPredictor

    assert isinstance(predictor, RouterPredictor)
    predictor.set_threshold(threshold)
    result = predictor.predict(message)

    print(f'\n  Test message: "{message}"')
    print(f"  Label:        {result.label}")
    print(f"  Routes to:    {result.routed_to}")
    print(f"  Confidence:   {result.confidence:.4f}")
    print(f"  Reason:       {result.routing_reason}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="v1")
    parser.add_argument("--model", default=None, help="Model key (default: best from comparator)")
    parser.add_argument("--test", default=None, help="Test one message after tuning")
    args = parser.parse_args()

    predictor, experiment_id = load_predictor(args.dataset, args.model)
    optimal_threshold = run_threshold_tuning(args.dataset, predictor, experiment_id)

    if args.test:
        run_test_prediction(predictor, args.test, optimal_threshold)


if __name__ == "__main__":
    main()
