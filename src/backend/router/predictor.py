"""
RouterPredictor — uniform inference interface for classical and transformer models.

ROUTING PIPELINE (in predict()):
  1. Script detection (script_detector.is_pure_script):
       Pure Sinhala/Tamil Unicode → mandatory label=1, confidence=1.0, skip ML.
  2. ML model confidence:
       Classical: vectorizer → predict_proba[:, 1]
       Transformer: tokenizer → forward → softmax[:, 1]
  3. Threshold check:
       confidence ≥ threshold → label=1 (gpt-4o)
       confidence <  threshold → label=0 (gpt-4o-mini)

All callers (routes_router.py, panel_router.py, ablation.py, phase_8_router.py)
call predict(text) → RoutingResult. Script detection is automatic and transparent.

INTERFACE:
  from_pkl(path)               → loads classical model (directory or .pkl file)
  from_hf_checkpoint(path)     → loads HuggingFace fine-tuned model
  predict(text) → RoutingResult
  predict_batch(texts) → list[RoutingResult]
  set_threshold(float)

BUGS FIXED vs earlier versions:
  - from_pkl unpacks {"vectorizer": v, "classifier": c} dict correctly.
  - threshold_config accepted in __init__ for backward-compatible callers.
"""

from __future__ import annotations

import logging
import pickle
import warnings
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
            raise FileNotFoundError(f"Model not found for path: {model_path}\n  Tried:\n  {tried}")

        log.info(f"Loaded model from {pkl_path}")

        with pkl_path.open("rb") as f, warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
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

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt), fix_mistral_regex=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
        model.eval()
        log.info(f"Loaded HF checkpoint from {ckpt}")
        return cls(model=model, vectorizer=tokenizer)

    def set_threshold(self, threshold: float) -> None:
        """Update the decision threshold."""
        self._threshold = threshold

    def predict(self, text: str) -> RoutingResult:
        """Route one message through the three-layer decision pipeline.

        Layer 1 — Script detection:
            If the text contains pure Sinhala or Tamil Unicode script (not romanized),
            it is mandatory gpt-4o. No ML inference needed.

        Layer 2 — ML confidence:
            P(label=1) from the trained classifier.

        Layer 3 — Threshold:
            confidence >= threshold → gpt-4o (label=1)
            confidence <  threshold → gpt-4o-mini (label=0)
        """
        # Layer 1: pure Unicode script check
        try:
            from backend.shared.script_detector import is_pure_script

            if is_pure_script(text):
                return RoutingResult(
                    label=1,
                    routed_to="gpt-4o",
                    confidence=1.0,
                    routing_reason="Pure Sinhala/Tamil Unicode script detected → mandatory gpt-4o",
                )
        except Exception as exc:  # noqa: BLE001
            log.debug(f"Script detector error (skipping): {exc}")

        # Layer 2+3: ML confidence + threshold
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
        """Return P(label=1). Used by threshold tuner and comparator.

        Note: does NOT apply script detection — intentional, for evaluation purposes
        where we want to measure the ML model's raw output.
        """
        return self._get_confidence(text)

    def _get_confidence(self, text: str) -> float:
        """Return P(label=1) from the ML model only (no script detection here)."""
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
