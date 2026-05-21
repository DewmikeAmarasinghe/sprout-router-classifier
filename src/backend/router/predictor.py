"""
RouterPredictor — uniform inference interface for classical and transformer models.

INTERFACE HISTORY:
  predict(text) now returns RoutingResult (not int).
  Callers that used `if predictor.predict(t) == 1` should use `result.label == 1`.
  The old pattern `RouterPredictor(model=p._model, threshold_config=ThresholdConfig(...))`
  is now replaced by `predictor.set_threshold(t); predictor.predict(text)`.
  ThresholdConfig is still accepted in __init__ for backward-compatible call sites.

BUGS FIXED:
  - from_pkl loaded {"vectorizer": v, "classifier": c} dict as self._model, then
    called self._model.predict_proba() → AttributeError. Fixed: unpack dict.
  - from_pkl path resolution tried .pkl extension before checking if path is a
    directory → FileNotFoundError. Fixed: tries candidates in priority order.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from backend.router.pymodels import RoutingResult, ThresholdConfig
from backend.shared.settings_manager import settings_manager

log = logging.getLogger(__name__)


class RouterPredictor:
    """Uniform predict interface for any trained router model.

    Classical:   wraps (vectorizer, classifier) loaded from model.pkl dict.
    Transformer: wraps HuggingFace model + tokenizer from checkpoint directory.
    """

    def __init__(
        self,
        model: Any,
        vectorizer: Any = None,
        threshold: float | None = None,
        threshold_config: ThresholdConfig | None = None,
    ) -> None:
        self._model = model
        self._vectorizer = vectorizer
        if threshold_config is not None:
            self._threshold = threshold_config.threshold
        elif threshold is not None:
            self._threshold = threshold
        else:
            self._threshold = float(settings_manager.get("CONFIDENCE_THRESHOLD"))

    @classmethod
    def from_pkl(cls, path: str | Path) -> RouterPredictor:
        """Load a classical model from pickle.

        Tries candidates in priority order so it handles all layouts:
          path/model.pkl  ← trainer default (directory layout)
          path            ← direct file path
          path.pkl        ← bare name

        The saved format is {"vectorizer": v, "classifier": c}.
        Unpacked automatically — classifier stored as self._model.
        """
        model_path = Path(path)

        candidates = [
            model_path / "model.pkl",
            model_path,
            model_path.with_suffix(".pkl"),
        ]

        pkl_path: Path | None = None
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                pkl_path = candidate
                break

        if pkl_path is None:
            tried = "\n  ".join(str(c) for c in candidates)
            raise FileNotFoundError(
                f"Model not found for path: {model_path}\n"
                f"  Tried:\n  {tried}\n"
                f"  Ensure trainer saved model.pkl inside the experiment directory."
            )

        log.info(f"Loaded model from {pkl_path}")

        with pkl_path.open("rb") as f:
            bundle = pickle.load(f)

        if isinstance(bundle, dict) and "vectorizer" in bundle and "classifier" in bundle:
            return cls(model=bundle["classifier"], vectorizer=bundle["vectorizer"])

        return cls(model=bundle)

    @classmethod
    def from_hf_checkpoint(cls, checkpoint_dir: str | Path) -> RouterPredictor:
        """Load a fine-tuned HuggingFace model from a checkpoint directory."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        ckpt = Path(checkpoint_dir)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
        model.eval()
        log.info(f"Loaded HF checkpoint from {ckpt}")
        return cls(model=model, vectorizer=tokenizer)

    def set_threshold(self, threshold: float) -> None:
        """Update the decision threshold."""
        self._threshold = threshold

    def predict(self, text: str) -> RoutingResult:
        """Route one message. Returns RoutingResult with label, model, confidence, reason."""
        confidence = self._get_confidence(text)
        label = 1 if confidence >= self._threshold else 0
        routed_to = "gpt-4o" if label == 1 else "gpt-4o-mini"
        reason = (
            f"confidence {confidence:.3f} ≥ threshold {self._threshold:.2f} → {routed_to}"
            if label == 1
            else f"confidence {confidence:.3f} < threshold {self._threshold:.2f} → {routed_to}"
        )
        return RoutingResult(
            label=label,
            routed_to=routed_to,
            confidence=confidence,
            routing_reason=reason,
        )

    def predict_batch(self, texts: list[str]) -> list[RoutingResult]:
        """Route a list of messages."""
        return [self.predict(t) for t in texts]

    def predict_proba(self, text: str) -> float:
        """Return P(label=1). Used by threshold tuner and comparator."""
        return self._get_confidence(text)

    def _get_confidence(self, text: str) -> float:
        """Return P(label=1).

        Classical path: vectorizer.transform([text]) → classifier.predict_proba.
        Transformer path: tokenizer → model forward → softmax[:, 1].
        """
        if self._vectorizer is not None and hasattr(self._vectorizer, "transform"):
            vec: Any = self._vectorizer.transform([text])
            proba: Any = self._model.predict_proba(vec)
            return float(proba[0][1])

        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba([text])
            return float(proba[0][1])

        return self._hf_confidence(text)

    def _hf_confidence(self, text: str) -> float:
        """HuggingFace forward pass → P(label=1)."""
        import torch
        from scipy.special import softmax as sp_softmax

        tokenizer = self._vectorizer
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits.cpu().numpy()
        proba = sp_softmax(logits, axis=-1)
        return float(proba[0][1])
