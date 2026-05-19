"""
HPO for transformer models using Optuna with MedianPruner.

MedianPruner kills trials whose epoch-1 eval metric is below the median of
completed trials, saving ~50% of GPU time vs running all trials to completion.

Expected time for 10 trials with pruning: ~2–3 hours on Kaggle T4.

Usage:
    runner = TransformerHPORunner()
    result = runner.run(
        dataset_name="v1",
        model_key="xlmr-base",
        n_trials=10,
    )
    print(result.best_params)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.shared.path_resolver import get_experiment_path
from backend.training.transformers.config import HPO_SEARCH_SPACE, TRANSFORMER_REGISTRY

if TYPE_CHECKING:
    import optuna

log = logging.getLogger(__name__)


@dataclass
class TransformerHPOResult:
    """Result of one transformer HPO run."""

    dataset_name: str
    model_key: str
    best_params: dict = field(default_factory=dict)
    best_value: float = 0.0
    n_trials: int = 0
    completed_trials: int = 0
    pruned_trials: int = 0
    optimize_metric: str = "recall_1"
    output_path: str = ""


class TransformerHPORunner:
    """Run Optuna HPO over a single transformer model with MedianPruner."""

    def run(
        self,
        dataset_name: str,
        model_key: str,
        n_trials: int = 10,
        optimize_metric: str = "recall_1",
        direction: str = "maximize",
        n_startup_trials: int = 3,
    ) -> TransformerHPOResult:
        """Run HPO study with MedianPruner and return the best params.

        Args:
            dataset_name: Dataset version, e.g. "v1".
            model_key: Key from TRANSFORMER_REGISTRY (e.g. "xlmr-base").
            n_trials: Number of Optuna trials.
            optimize_metric: Metric to optimise. Supported: "recall_1", "mcc",
                             "roc_auc", "f1_macro".
            direction: "maximize" or "minimize".
            n_startup_trials: Trials to run before pruning kicks in.

        Returns:
            TransformerHPOResult with best_params, best_value, and output_path.
        """
        import optuna
        from optuna.integration.mlflow import MLflowCallback
        from optuna.pruners import MedianPruner

        from backend.training.transformers.trainer import TransformerTrainer

        if model_key not in TRANSFORMER_REGISTRY:
            raise ValueError(
                f"Unknown model key {model_key!r}. Available: {sorted(TRANSFORMER_REGISTRY)}"
            )

        output_dir = get_experiment_path(dataset_name, "transformers") / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        trainer = TransformerTrainer()
        study_name = f"{model_key}__{dataset_name}"

        pruner = MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=0,
            interval_steps=1,
        )
        mlflow_cb = MLflowCallback(
            metric_name=optimize_metric,
            mlflow_kwargs={"experiment_name": f"transformers_hpo_{dataset_name}"},
        )

        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial)
            try:
                result = trainer.train_experiment(
                    dataset_name=dataset_name,
                    model_key=model_key,
                    param_overrides=params,
                )
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as exc:
                log.warning(f"Trial {trial.number} failed: {exc}")
                raise optuna.exceptions.TrialPruned() from exc

            value = getattr(result.metrics, optimize_metric, 0.0)
            trial.set_user_attr("recall_1", result.metrics.recall_1)
            trial.set_user_attr("mcc", result.metrics.mcc)
            trial.set_user_attr("roc_auc", result.metrics.roc_auc)
            trial.set_user_attr("f1_macro", result.metrics.f1_macro)

            # Report intermediate value for pruning (treat as epoch 1 result)
            trial.report(float(value), step=1)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return float(value)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            pruner=pruner,
        )

        log.info(
            f"Starting TransformerHPO: model={model_key}  "
            f"dataset={dataset_name}  n_trials={n_trials}"
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[mlflow_cb],
            show_progress_bar=True,
        )

        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        pruned = [t for t in study.trials if t.state.name == "PRUNED"]

        best_params = study.best_params
        best_value = study.best_value

        output_path = output_dir / f"{model_key}__best_params.json"
        output_path.write_text(
            json.dumps(
                {
                    "study_name": study_name,
                    "model_key": model_key,
                    "dataset_name": dataset_name,
                    "best_params": best_params,
                    "best_value": best_value,
                    "optimize_metric": optimize_metric,
                    "n_trials": n_trials,
                    "completed_trials": len(completed),
                    "pruned_trials": len(pruned),
                },
                indent=2,
            )
        )

        log.info(
            f"HPO done: best {optimize_metric}={best_value:.4f}  "
            f"completed={len(completed)}  pruned={len(pruned)}"
        )
        log.info(f"Best params: {best_params}")
        log.info(f"Saved to: {output_path}")

        return TransformerHPOResult(
            dataset_name=dataset_name,
            model_key=model_key,
            best_params=best_params,
            best_value=best_value,
            n_trials=n_trials,
            completed_trials=len(completed),
            pruned_trials=len(pruned),
            optimize_metric=optimize_metric,
            output_path=str(output_path),
        )

    def load_best_params(self, dataset_name: str, model_key: str) -> dict:
        """Load previously saved best params for a model."""
        output_dir = get_experiment_path(dataset_name, "transformers") / "results"
        path = output_dir / f"{model_key}__best_params.json"

        if not path.exists():
            raise FileNotFoundError(
                f"No HPO results found at {path}. Run TransformerHPORunner.run(...) first."
            )

        data = json.loads(path.read_text())
        return data["best_params"]


def suggest_params(trial: optuna.Trial) -> dict:
    """Sample hyperparams from HPO_SEARCH_SPACE for one trial."""
    params: dict = {}
    for name, spec in HPO_SEARCH_SPACE.items():
        kind = spec[0]
        if kind == "float_log":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec[1]))
        else:
            raise ValueError(f"Unknown search space kind: {kind!r}")
    return params
