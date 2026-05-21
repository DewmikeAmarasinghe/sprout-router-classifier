"""
Phase 6 — Transformer Fine-Tuning.

What exactly happens during training:
  We do FULL fine-tuning of all weights — not just the head.
  AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
  loads the pre-trained XLM-R encoder (125M params, trained on 100 languages via MLM)
  and adds a new randomly-initialised linear head (768 → 2).
  All parameters are trained with AdamW (lr=2e-5, weight_decay=0.01).
  The low lr prevents catastrophic forgetting of pre-trained language knowledge
  while adapting the model to our routing classification task.
  3 epochs with load_best_model_at_end=True acts as implicit early stopping.

Auto-HPO:
  After each training run, if recall_1 < 0.97, Optuna HPO runs automatically
  (default 5 trials). Each trial trains with different hyperparameters and evaluates
  on val.csv. MedianPruner stops bad trials after epoch 1, saving GPU time.
  If recall_1 >= 0.97, HPO is skipped.
  Use --no-hpo to disable auto-HPO entirely.
  Use --hpo to force-run HPO regardless of initial recall_1.

Usage:
    python phases/phase_6_train_transformers.py --dataset v1 --model xlmr-base
    python phases/phase_6_train_transformers.py --dataset v1 --model xlmr-base --no-hpo
    python phases/phase_6_train_transformers.py --dataset v1 --all
    python phases/phase_6_train_transformers.py --dataset v1 --model xlmr-base --export-onnx
    python phases/phase_6_train_transformers.py --dataset v1 --model xlmr-base --hpo --n-trials 10
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

from backend.training.transformers.config import TRANSFORMER_REGISTRY
from backend.training.transformers.trainer import TransformerTrainer


def run_auto_hpo(dataset: str, model_key: str, n_trials: int) -> None:
    """Run Optuna HPO on a transformer model and retrain with best params.

    HOW TRANSFORMER HPO WORKS:
      Each Optuna trial:
        1. Samples hyperparameters from HPO_SEARCH_SPACE (lr, batch_size,
           num_epochs, warmup_ratio, weight_decay, max_length).
        2. Runs a full training run with those params on train.csv.
        3. Evaluates on val.csv → returns recall_1 as objective.
        4. MedianPruner: after epoch 1 of a trial, if recall_1 is below the
           median of completed trials, the trial is pruned (stopped early).
           This saves ~50% of GPU time vs running all trials to completion.

      With n_trials=5 and MedianPruner:
        ~2-3 trials complete fully (~150 min), ~2-3 pruned after epoch 1 (~50 min).
        Total extra time: ~200-250 min on T4. Worthwhile if initial recall < 0.97.

      NEVER uses test.csv — only val.csv for evaluation.
    """
    from backend.training.transformers.hpo import TransformerHPORunner

    log.info(
        f"Auto-HPO: {model_key}  n_trials={n_trials}  "
        f"(each trial = full training run, MedianPruner stops bad trials early)"
    )

    hpo_result = TransformerHPORunner().run(
        dataset_name=dataset,
        model_key=model_key,
        n_trials=n_trials,
        optimize_metric="recall_1",
    )

    log.info(f"HPO best params: {hpo_result.best_params}")
    log.info("Retraining with tuned hyperparameters...")

    trainer = TransformerTrainer()
    result = trainer.train_experiment(dataset, model_key, param_overrides=hpo_result.best_params)
    flag = "✅ PASSES" if result.metrics.passes_production_threshold else "⚠ still below 0.97"
    log.info(
        f"HPO retrain: recall_1={result.metrics.recall_1:.4f}  MCC={result.metrics.mcc:.4f}  {flag}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="v1")
    parser.add_argument(
        "--model",
        default=None,
        choices=list(TRANSFORMER_REGISTRY),
        help="Model key to train. Required unless --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all models in TRANSFORMER_REGISTRY sequentially.",
    )
    parser.add_argument(
        "--no-hpo",
        action="store_true",
        help="Skip automatic HPO even if recall_1 < 0.97.",
    )
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="Force-run HPO regardless of initial recall_1.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Optuna HPO trials (default 5). Each trial = full training run.",
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
        if not args.no_hpo:
            for r in results:
                if not r.metrics.passes_production_threshold or args.hpo:
                    run_auto_hpo(args.dataset, r.experiment_id, args.n_trials)
    else:
        if args.hpo:
            run_auto_hpo(args.dataset, args.model, args.n_trials)
        else:
            result = trainer.train_experiment(args.dataset, args.model)
            results = [result]

            if not args.no_hpo and (not result.metrics.passes_production_threshold or args.hpo):
                if not result.metrics.passes_production_threshold:
                    log.info(
                        f"recall_1={result.metrics.recall_1:.4f} < 0.97. "
                        f"Running auto-HPO ({args.n_trials} trials)..."
                    )
                run_auto_hpo(args.dataset, args.model, args.n_trials)

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
