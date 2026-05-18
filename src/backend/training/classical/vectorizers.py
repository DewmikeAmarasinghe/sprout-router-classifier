"""Vectorizer factories for classical ML experiments."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

log = logging.getLogger(__name__)


def build_vectorizer(key: str, **param_overrides: Any) -> Any:
    """Factory: return an unfitted sklearn-compatible vectorizer."""
    from backend.training.classical.config import VECTORIZER_REGISTRY

    if key not in VECTORIZER_REGISTRY:
        raise ValueError(f"Unknown vectorizer: {key!r}. Available: {sorted(VECTORIZER_REGISTRY)}")

    spec = VECTORIZER_REGISTRY[key]
    params = {**spec.default_params, **param_overrides}

    if key == "tfidf_char":
        return TfidfVectorizer(
            analyzer=params["analyzer"],
            ngram_range=params["ngram_range"],
            max_features=params["max_features"],
            sublinear_tf=params["sublinear_tf"],
            min_df=params["min_df"],
        )

    if key == "tfidf_word":
        return TfidfVectorizer(
            analyzer=params["analyzer"],
            ngram_range=params["ngram_range"],
            max_features=params["max_features"],
            sublinear_tf=params["sublinear_tf"],
            min_df=params["min_df"],
        )

    if key == "tfidf_combined":
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=params["char_ngram_range"],
            max_features=params["max_features_char"],
            sublinear_tf=params["sublinear_tf"],
            min_df=params["min_df"],
        )
        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=params["word_ngram_range"],
            max_features=params["max_features_word"],
            sublinear_tf=params["sublinear_tf"],
            min_df=params["min_df"],
        )
        return FeatureUnion([("char", char_vec), ("word", word_vec)])

    if key == "word2vec":
        return Word2VecVectorizer(
            vector_size=params["vector_size"],
            window=params["window"],
            min_count=params["min_count"],
            workers=params["workers"],
            epochs=params["epochs"],
        )

    if key == "spacy":
        return SpacyVectorizer(model=params["model"], batch_size=params["batch_size"])

    raise ValueError(f"No implementation for vectorizer: {key!r}")


class Word2VecVectorizer:
    """Train Word2Vec on the corpus, return mean word vector per text.

    Implements sklearn's fit/transform interface for Pipeline compatibility.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model_: Any = None

    def fit(self, X: list[str], y: Any = None) -> Word2VecVectorizer:
        from gensim.models import Word2Vec

        sentences = [text.lower().split() for text in X]
        log.info(f"Training Word2Vec on {len(sentences):,} sentences...")
        self.model_ = Word2Vec(
            sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
        )
        return self

    def transform(self, X: list[str]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Word2VecVectorizer not fitted. Call fit() first.")
        wv = self.model_.wv
        vectors = []
        for text in X:
            words = [w for w in text.lower().split() if w in wv]
            vector = (
                np.mean([wv[w] for w in words], axis=0) if words else np.zeros(self.vector_size)
            )
            vectors.append(vector)
        return np.array(vectors)

    def fit_transform(self, X: list[str], y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "vector_size": self.vector_size,
            "window": self.window,
            "min_count": self.min_count,
            "workers": self.workers,
            "epochs": self.epochs,
        }

    def set_params(self, **params: Any) -> Word2VecVectorizer:
        for key, value in params.items():
            setattr(self, key, value)
        return self


class SpacyVectorizer:
    """Pre-trained spaCy vectors (en_core_web_md). Returns 300-dim dense vectors.

    Requires: python -m spacy download en_core_web_md
    """

    def __init__(self, model: str = "en_core_web_md", batch_size: int = 512) -> None:
        self.model = model
        self.batch_size = batch_size
        self.nlp_: Any = None

    def fit(self, X: list[str], y: Any = None) -> SpacyVectorizer:
        try:
            import spacy

            log.info(f"Loading spaCy model: {self.model}")
            self.nlp_ = spacy.load(self.model, disable=["ner", "parser"])
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{self.model}' not found. Run: python -m spacy download {self.model}"
            ) from exc
        return self

    def transform(self, X: list[str]) -> np.ndarray:
        if self.nlp_ is None:
            raise RuntimeError("SpacyVectorizer not fitted. Call fit() first.")
        return np.array([doc.vector for doc in self.nlp_.pipe(X, batch_size=self.batch_size)])

    def fit_transform(self, X: list[str], y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_params(self, deep: bool = True) -> dict:
        return {"model": self.model, "batch_size": self.batch_size}

    def set_params(self, **params: Any) -> SpacyVectorizer:
        for key, value in params.items():
            setattr(self, key, value)
        return self
