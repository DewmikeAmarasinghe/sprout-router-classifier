"""
Phase 6 — Transformer Fine-Tuning.

Fine-tunes multilingual transformer models on the generated dataset.
Run this on Kaggle/Colab with GPU — CPU training is too slow.

MODELS (in recommended training order):
    xlmr-base   → START HERE. ~50 min on Kaggle T4.
    papluca     → Pre-tuned on language detection. ~50 min.
    muril       → Best for Tamil/Tanglish. ~50 min.
    xlmr-large  → Only if xlmr-base plateaus. ~100 min.

WHY KAGGLE/COLAB:
    T4 GPU (16GB VRAM) is 4–8× faster than M-series Mac for transformer training.
    Notebooks share checkpoints via Google Drive or Kaggle outputs.
    See docs/08_WORKFLOW.md for the Kaggle workflow.

PRODUCTION THRESHOLD:
    recall_1 >= 0.97 on val set. False negatives (label=1 predicted as 0)
    send sensitive/complex messages to gpt-4o-mini — bad UX.

Usage:
    python phases/phase_6_train_transformers.py --dataset v1 --model xlmr-base
    python phases/phase_6_train_transformers.py --dataset v1 --all
    python phases/phase_6_train_transformers.py --dataset v1 --model xlmr-base --export-onnx
    python phases/phase_6_train_transformers.py --dataset v1 --model xlmr-base --hpo --n-trials 10
    python phases/phase_6_train_transformers.py --dataset v1 --all --export-onnx

After completion:
    python phases/phase_7_evaluate.py --dataset v1
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

from backend.training.transformers.config import TRANSFORMER_REGISTRY
from backend.training.transformers.trainer import TransformerTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="v1", help="Dataset version (e.g. v1)")
    parser.add_argument(
        "--model",
        default=None,
        choices=list(TRANSFORMER_REGISTRY),
        help="Model key to train. Required unless --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all models in TRANSFORMER_REGISTRY sequentially.",
    )
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="Run HPO with Optuna before final training.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of Optuna trials for --hpo. Default 10.",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export trained checkpoint(s) to ONNX after training.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 quantization during ONNX export (requires --export-onnx).",
    )
    args = parser.parse_args()

    if not args.all and not args.model:
        parser.error("Provide --model <key> or --all")

    trainer = TransformerTrainer()

    if args.all:
        results = trainer.train_all_models(args.dataset)
    else:
        if args.hpo:
            from backend.training.transformers.hpo import TransformerHPORunner

            hpo_result = TransformerHPORunner().run(
                dataset_name=args.dataset,
                model_key=args.model,
                n_trials=args.n_trials,
            )
            param_overrides = hpo_result.best_params
            logging.info(f"HPO complete. Best params: {param_overrides}")
        else:
            param_overrides = None

        result = trainer.train_experiment(
            dataset_name=args.dataset,
            model_key=args.model,
            param_overrides=param_overrides,
        )
        results = [result]

    if args.export_onnx:
        from backend.training.transformers.onnx_exporter import OnnxExporter

        exporter = OnnxExporter()
        for r in results:
            exporter.export(
                dataset_name=r.dataset_name,
                model_key=r.experiment_id,
                quantize=args.quantize,
            )


if __name__ == "__main__":
    main()
