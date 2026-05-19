"""
HPO for classical ML using Optuna.

Wraps ClassicalMLTrainer.train_experiment() in an Optuna study.
The search space is per-classifier — only hyperparams that meaningfully affect
performance are tuned (regularisation, tree depth, learning rate).

Usage:
    runner = ClassicalHPORunner()
    best = runner.run(
        dataset_name="v1",
        vectorizer_key="tfidf_combined",
        classifier_key="logistic_regression",
        n_trials=20,
    )
    print(best.best_params)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.shared.path_resolver import get_experiment_path

if TYPE_CHECKING:
    import optuna

log = logging.getLogger(__name__)

# Per-classifier search spaces.
# Format: param_name → ("float_log" | "float" | "int" | "categorical", *args)
SEARCH_SPACES: dict[str, dict[str, tuple]] = {
    "logistic_regression": {
        "C": ("float_log", 1e-3, 1e2),
        "max_iter": ("categorical", [500, 1000, 2000]),
    },
    "svm": {
        "C": ("float_log", 1e-3, 1e2),
        "max_iter": ("categorical", [1000, 2000, 5000]),
    },
    "lightgbm": {
        "n_estimators": ("int", 100, 600),
        "learning_rate": ("float_log", 1e-3, 3e-1),
        "num_leaves": ("int", 15, 127),
    },
    "xgboost": {
        "n_estimators": ("int", 100, 600),
        "learning_rate": ("float_log", 1e-3, 3e-1),
        "max_depth": ("int", 3, 10),
    },
    "catboost": {
        "iterations": ("int", 100, 600),
        "learning_rate": ("float_log", 1e-3, 3e-1),
        "depth": ("int", 4, 10),
    },
}


@dataclass
class HPOResult:
    """Result of one HPO run."""

    dataset_name: str
    vectorizer_key: str
    classifier_key: str
    best_params: dict = field(default_factory=dict)
    best_value: float = 0.0
    n_trials: int = 0
    output_path: str = ""


class ClassicalHPORunner:
    """Run Optuna HPO over a (vectorizer, classifier) pair."""

    def run(
        self,
        dataset_name: str,
        vectorizer_key: str,
        classifier_key: str,
        n_trials: int = 20,
        direction: str = "maximize",
        optimize_metric: str = "mcc",
        n_jobs: int = 1,
    ) -> HPOResult:
        """Run HPO study and return the best params.

        Args:
            dataset_name: Dataset version, e.g. "v1".
            vectorizer_key: Key from VECTORIZER_REGISTRY.
            classifier_key: Key from CLASSIFIER_REGISTRY.
            n_trials: Number of Optuna trials.
            direction: "maximize" or "minimize".
            optimize_metric: Which ExperimentResult metric to optimise.
                             Supported: "mcc", "recall_1", "roc_auc", "f1_macro".
            n_jobs: Parallel workers for study.optimize().

        Returns:
            HPOResult with best_params and best_value.
        """
        import optuna
        from optuna.integration.mlflow import MLflowCallback

        from backend.training.classical.trainer import ClassicalMLTrainer

        if classifier_key not in SEARCH_SPACES:
            raise ValueError(
                f"No HPO search space defined for classifier {classifier_key!r}. "
                f"Available: {sorted(SEARCH_SPACES)}"
            )

        output_dir = get_experiment_path(dataset_name, "classical") / "hpo"
        output_dir.mkdir(parents=True, exist_ok=True)

        trainer = ClassicalMLTrainer()
        study_name = f"{vectorizer_key}__{classifier_key}"

        mlflow_cb = MLflowCallback(
            metric_name=optimize_metric,
            mlflow_kwargs={"experiment_name": f"classical_hpo_{dataset_name}"},
        )

        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial, classifier_key)
            result = trainer.train_experiment(
                dataset_name=dataset_name,
                vectorizer_key=vectorizer_key,
                classifier_key=classifier_key,
                classifier_params=params,
            )
            value = getattr(result.metrics, optimize_metric, 0.0)
            trial.set_user_attr("recall_1", result.metrics.recall_1)
            trial.set_user_attr("mcc", result.metrics.mcc)
            trial.set_user_attr("roc_auc", result.metrics.roc_auc)
            return float(value)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=[mlflow_cb],
            show_progress_bar=True,
        )

        best_params = study.best_params
        best_value = study.best_value

        output_path = output_dir / f"{study_name}__best_params.json"
        output_path.write_text(
            json.dumps(
                {
                    "study_name": study_name,
                    "best_params": best_params,
                    "best_value": best_value,
                    "optimize_metric": optimize_metric,
                    "n_trials": n_trials,
                },
                indent=2,
            )
        )
        log.info(f"HPO complete: best {optimize_metric}={best_value:.4f}  params={best_params}")
        log.info(f"Saved best params to: {output_path}")

        return HPOResult(
            dataset_name=dataset_name,
            vectorizer_key=vectorizer_key,
            classifier_key=classifier_key,
            best_params=best_params,
            best_value=best_value,
            n_trials=n_trials,
            output_path=str(output_path),
        )

    def load_best_params(
        self,
        dataset_name: str,
        vectorizer_key: str,
        classifier_key: str,
    ) -> dict:
        """Load previously saved best params for a (vectorizer, classifier) pair."""
        output_dir = get_experiment_path(dataset_name, "classical") / "hpo"
        study_name = f"{vectorizer_key}__{classifier_key}"
        path = output_dir / f"{study_name}__best_params.json"

        if not path.exists():
            raise FileNotFoundError(
                f"No HPO results found at {path}. Run ClassicalHPORunner.run(...) first."
            )

        data = json.loads(path.read_text())
        return data["best_params"]


def suggest_params(trial: optuna.Trial, classifier_key: str) -> dict:
    """Sample hyperparams for one trial from the classifier's search space."""
    space = SEARCH_SPACES[classifier_key]
    params: dict = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "float_log":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f"Unknown search space kind: {kind!r}")
    return params
