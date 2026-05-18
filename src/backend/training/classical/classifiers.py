"""
Classifier factories for classical ML experiments.

All classifiers output probabilities (required for the confidence threshold
in the router). Classifiers that don't support predict_proba natively
(LinearSVC) are wrapped with CalibratedClassifierCV.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def build_classifier(key: str, **param_overrides: Any) -> Any:
    """Factory: return an unfitted sklearn-compatible classifier.

    All returned classifiers support predict_proba().
    SVM is wrapped with CalibratedClassifierCV automatically.

    Args:
        key: One of the keys in CLASSIFIER_REGISTRY.
        **param_overrides: Override default params from the registry.

    Returns:
        Unfitted sklearn-compatible classifier.
    """
    from backend.training.classical.config import CLASSIFIER_REGISTRY

    if key not in CLASSIFIER_REGISTRY:
        raise ValueError(f"Unknown classifier: {key!r}. Available: {sorted(CLASSIFIER_REGISTRY)}")

    spec = CLASSIFIER_REGISTRY[key]
    params = {**spec.default_params, **param_overrides}

    if key == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            solver=params["solver"],
            class_weight=params.get("class_weight"),
            n_jobs=params.get("n_jobs"),
        )

    if key == "svm":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC

        base = LinearSVC(
            C=params["C"],
            max_iter=params["max_iter"],
            class_weight=params.get("class_weight"),
        )
        return CalibratedClassifierCV(base, cv=3)

    if key == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            class_weight=params.get("class_weight"),
            n_jobs=params.get("n_jobs", -1),
            verbose=params.get("verbose", -1),
        )

    if key == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            eval_metric=params.get("eval_metric", "logloss"),
            n_jobs=params.get("n_jobs", -1),
            tree_method=params.get("tree_method", "hist"),
            use_label_encoder=False,
            verbosity=0,
        )

    if key == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=params["iterations"],
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            verbose=params.get("verbose", 0),
            auto_class_weights=params.get("auto_class_weights", "Balanced"),
        )

    raise ValueError(f"No implementation for classifier: {key!r}")
