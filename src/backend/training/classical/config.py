"""
Classical ML experiment configuration.

Defines which vectorizers and classifiers to run and their default params.
All combos in ACTIVE_COMBOS are run by phase_5_train_classical.py --all.

WHY THESE VECTORIZERS:
  tfidf_char:     Best for romanized Sinhala/Tamil — character ngrams (2-5grams)
                  catch unique patterns like "kohomada", "theriyum" that TF-IDF
                  word misses when OOV. Most important for code-mixed detection.
  tfidf_word:     Word unigrams+bigrams. Strong baseline for pure English.
  tfidf_combined: Union of char + word. Usually best overall.
  word2vec:       Mean pooling of W2V vectors trained on the corpus. Dense.
  spacy:          Pre-trained en_core_web_md sentence vectors. Dense.

WHY THESE CLASSIFIERS:
  logistic_regression: Fast, calibrated, excellent baseline for sparse TF-IDF.
  svm:                 LinearSVC + Platt calibration. Strong for high-dim sparse.
  lightgbm:            Gradient boosting, handles both sparse and dense well.
  xgboost:             Similar to LGBM, different implementation.
  catboost:            Good for text features, handles overfitting well.

ACTIVE_COMBOS priority order (fast first, dense last):
  TF-IDF combos run in minutes. W2V/spaCy combos take longer (train W2V on corpus).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VectorizerSpec:
    key: str
    display_name: str
    description: str
    default_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ClassifierSpec:
    key: str
    display_name: str
    description: str
    needs_calibration: bool  # True if classifier doesn't output probabilities
    default_params: dict = field(default_factory=dict)


VECTORIZER_REGISTRY: dict[str, VectorizerSpec] = {
    "tfidf_char": VectorizerSpec(
        key="tfidf_char",
        display_name="TF-IDF Char ngrams",
        description="Character 2-5gram TF-IDF. Best for romanized code-mixed text.",
        default_params={
            "analyzer": "char_wb",
            "ngram_range": (2, 5),
            "max_features": 100_000,
            "sublinear_tf": True,
            "min_df": 2,
        },
    ),
    "tfidf_word": VectorizerSpec(
        key="tfidf_word",
        display_name="TF-IDF Word ngrams",
        description="Word unigram + bigram TF-IDF. Baseline.",
        default_params={
            "analyzer": "word",
            "ngram_range": (1, 2),
            "max_features": 50_000,
            "sublinear_tf": True,
            "min_df": 2,
        },
    ),
    "tfidf_combined": VectorizerSpec(
        key="tfidf_combined",
        display_name="TF-IDF Combined (char + word)",
        description="FeatureUnion of char ngrams + word ngrams. Usually best.",
        default_params={
            "char_ngram_range": (2, 5),
            "word_ngram_range": (1, 2),
            "max_features_char": 80_000,
            "max_features_word": 40_000,
            "sublinear_tf": True,
            "min_df": 2,
        },
    ),
    "word2vec": VectorizerSpec(
        key="word2vec",
        display_name="Word2Vec Mean Pooling",
        description="Train W2V on corpus, use mean word vector. Dense 100-dim.",
        default_params={
            "vector_size": 100,
            "window": 5,
            "min_count": 2,
            "workers": 4,
            "epochs": 10,
        },
    ),
    "spacy": VectorizerSpec(
        key="spacy",
        display_name="spaCy en_core_web_md",
        description="Pre-trained 300-dim spaCy vectors via nlp.pipe(). Dense.",
        default_params={
            "model": "en_core_web_md",
            "batch_size": 512,
        },
    ),
}

CLASSIFIER_REGISTRY: dict[str, ClassifierSpec] = {
    "logistic_regression": ClassifierSpec(
        key="logistic_regression",
        display_name="Logistic Regression",
        description="L2-regularised LR. Fast, calibrated, strong TF-IDF baseline.",
        needs_calibration=False,
        default_params={
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "n_jobs": -1,
        },
    ),
    "svm": ClassifierSpec(
        key="svm",
        display_name="LinearSVC (calibrated)",
        description="LinearSVC wrapped with CalibratedClassifierCV for probabilities.",
        needs_calibration=True,
        default_params={
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
        },
    ),
    "lightgbm": ClassifierSpec(
        key="lightgbm",
        display_name="LightGBM",
        description="Gradient boosting. Fast, handles sparse and dense well.",
        needs_calibration=False,
        default_params={
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "class_weight": "balanced",
            "n_jobs": -1,
            "verbose": -1,
        },
    ),
    "xgboost": ClassifierSpec(
        key="xgboost",
        display_name="XGBoost",
        description="Gradient boosting. Dense features only — pair with word2vec or spacy.",
        needs_calibration=False,
        default_params={
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "n_jobs": -1,
            "tree_method": "hist",
        },
    ),
    "catboost": ClassifierSpec(
        key="catboost",
        display_name="CatBoost",
        description="Gradient boosting. Handles class imbalance and text well.",
        needs_calibration=False,
        default_params={
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
            "verbose": 0,
            "auto_class_weights": "Balanced",
        },
    ),
}

# Ordered by expected performance / training speed.
# Run --all to execute all; run specific combo with --vectorizer X --classifier Y.
#
# XGBoost is intentionally paired with dense vectorizers only (word2vec, spacy).
# Sparse TF-IDF + XGBoost can produce inverted predictions (mcc < 0) especially
# on GPU and is unreliable in practice.
ACTIVE_COMBOS: list[tuple[str, str]] = [
    # TF-IDF + fast classifiers (run first, ~2 min each)
    ("tfidf_combined", "logistic_regression"),
    ("tfidf_combined", "lightgbm"),
    ("tfidf_char", "logistic_regression"),
    ("tfidf_char", "lightgbm"),
    ("tfidf_word", "logistic_regression"),
    # SVM (needs calibration, slightly slower but best recall on TF-IDF)
    ("tfidf_combined", "svm"),
    ("tfidf_char", "svm"),
    # CatBoost on TF-IDF combined
    ("tfidf_combined", "catboost"),
    # Dense vector approaches (W2V trains on corpus, ~5 min)
    ("word2vec", "logistic_regression"),
    ("word2vec", "lightgbm"),
    ("word2vec", "xgboost"),
    # spaCy (requires en_core_web_md — auto-downloaded on first run)
    ("spacy", "logistic_regression"),
    ("spacy", "lightgbm"),
    ("spacy", "xgboost"),
]
