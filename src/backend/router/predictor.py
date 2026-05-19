"""
RouterPredictor — three-layer routing pipeline.

    Layer 1: is_pure_script(text) → label=1 immediately  (< 0.1ms)
    Layer 2: ML model → P(label=1)                        (1–30ms)
    Layer 3: confidence < threshold → SAFE_DEFAULT_LABEL  (0ms)

The predictor wraps a trained sklearn pipeline (.pkl) or HuggingFace checkpoint.

Usage:
    predictor = RouterPredictor.from_pkl(
        "experiments/v1/classical/models/tfidf_combined__lightgbm.pkl"
    )
    result = predictor.predict("nearest branch to me")
    print(result.label, result.routed_to)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from backend.router.pymodels import RouterPrediction, ThresholdConfig
from backend.shared.script_detector import is_pure_script
from backend.shared.settings_manager import settings_manager

log = logging.getLogger(__name__)


class RouterPredictor:
    """Routes one message through the three-layer decision pipeline."""

    def __init__(
        self,
        model: Any,
        threshold_config: ThresholdConfig | None = None,
    ) -> None:
        self._model: Any = model
        self._config = threshold_config or ThresholdConfig(
            threshold=settings_manager.get("CONFIDENCE_THRESHOLD"),
            safe_default_label=settings_manager.get("SAFE_DEFAULT_LABEL"),
        )

    @classmethod
    def from_pkl(
        cls,
        path: str | Path,
        threshold_config: ThresholdConfig | None = None,
    ) -> RouterPredictor:
        """Load a trained sklearn pipeline from a .pkl file."""
        import pickle

        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with model_path.open("rb") as f:
            model: Any = pickle.load(f)  # noqa: S301

        log.info(f"Loaded model from {model_path}")
        return cls(model=model, threshold_config=threshold_config)

    @classmethod
    def from_hf_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        threshold_config: ThresholdConfig | None = None,
    ) -> RouterPredictor:
        """Load a fine-tuned HuggingFace model from a checkpoint directory."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        tokenizer: Any = AutoTokenizer.from_pretrained(str(checkpoint_path))
        model: Any = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_path))

        wrapped = HuggingFaceWrapper(model=model, tokenizer=tokenizer)
        log.info(f"Loaded HuggingFace checkpoint from {checkpoint_path}")
        return cls(model=wrapped, threshold_config=threshold_config)

    def predict(self, text: str) -> RouterPrediction:
        """Route one message through the three-layer pipeline.

        Layer 1: Pure Sinhala/Tamil script → always label=1 (< 0.1ms)
        Layer 2: ML model → P(label=1)
        Layer 3: confidence < threshold → safe default label
        """
        if is_pure_script(text):
            return RouterPrediction(
                label=1,
                confidence=1.0,
                routed_to="gpt-4o",
                routing_reason="script_detector",
            )

        confidence = self._get_confidence(text)
        threshold = self._config.threshold
        safe_default = self._config.safe_default_label

        if confidence >= threshold:
            label = 1 if confidence >= 0.5 else 0
            reason = "model"
        else:
            label = safe_default
            reason = "below_threshold_default"

        return RouterPrediction(
            label=label,
            confidence=confidence,
            routed_to="gpt-4o" if label == 1 else "gpt-4o-mini",
            routing_reason=reason,
        )

    def predict_batch(self, texts: list[str]) -> list[RouterPrediction]:
        """Route a batch of messages."""
        return [self.predict(text) for text in texts]

    def _get_confidence(self, text: str) -> float:
        """Return P(label=1) from the underlying ML model."""
        if isinstance(self._model, HuggingFaceWrapper):
            return self._model.predict_proba(text)

        # sklearn pipeline: predict_proba returns shape (n_samples, 2)
        proba: Any = self._model.predict_proba([text])
        return float(np.array(proba)[0, 1])


class HuggingFaceWrapper:
    """Thin wrapper around a HuggingFace model for uniform predict_proba interface."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def predict_proba(self, text: str) -> float:
        """Return P(label=1) for one text string."""
        import torch
        from scipy.special import softmax

        encoded: Any = self._tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )

        self._model.eval()
        with torch.no_grad():
            logits: Any = self._model(**encoded).logits

        proba: Any = softmax(logits.numpy(), axis=-1)
        return float(proba[0, 1])
