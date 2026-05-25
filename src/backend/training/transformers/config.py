"""
TransformerSpec registry and training configuration.

MODEL SELECTION RATIONALE:
  xlmr-base:  XLM-RoBERTa is empirically the best multilingual model for Sinhala
              classification (University of Moratuwa, 2022 — arxiv.org/abs/2208.07864).
              START HERE. 125M params, ~50 min on Kaggle T4.

  xlmr-large: Same architecture, higher ceiling (560M params). 4× slower.
              Only if xlmr-base plateaus below recall_1=PRODUCTION_RECALL_THRESHOLD.

  papluca:    XLM-RoBERTa pre-finetuned on 20-language detection. May need fewer
              epochs due to existing language signal knowledge.

  muril:      Google MuRIL — trained on 17 Indian languages including transliterated
              Tamil. Best for Tanglish. Complementary to xlmr-base (ensemble potential).

  mbert:      mBERT as baseline — trained on multilingual Wikipedia (104 languages).
              Weaker than XLM-R on Sinhala, weaker than MuRIL on Tamil. Keep for
              comparison purposes so the improvement over baseline is documented.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerSpec:
    """Registry entry for one pre-trained transformer model."""

    key: str
    hf_name: str
    description: str
    notes: str = ""


# MODEL HARDWARE REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
# key           params      size (MB)   min VRAM   notes
# xlmr-base     125M–278M   ~1,000      8 GB+      Standard 12-layer. Start here.
# xlmr-large    550M–560M   ~2,200      12 GB+     24-layer; 4× compute of base.
# papluca       ~278M       ~1,000      8 GB+      XLM-R base pre-tuned on lang-detect.
# muril         ~278M       ~950        8 GB+      South Asian scripts specialist.
# mbert         178M        ~700        6 GB+      Older cased multilingual baseline.
# ─────────────────────────────────────────────────────────────────────────────
TRANSFORMER_REGISTRY: dict[str, TransformerSpec] = {
    "xlmr-base": TransformerSpec(
        key="xlmr-base",
        hf_name="xlm-roberta-base",
        description="XLM-RoBERTa base. Best for Sinhala — empirically verified.",
        notes="Handles Singlish, Tanglish, English. ~50 min on Kaggle T4.",
    ),
    "xlmr-large": TransformerSpec(
        key="xlmr-large",
        hf_name="xlm-roberta-large",
        description="XLM-RoBERTa large (560M params). Higher ceiling.",
        notes="4× slower than base. Only if xlmr-base plateaus. ~100 min on Kaggle T4.",
    ),
    "papluca": TransformerSpec(
        key="papluca",
        hf_name="papluca/xlm-roberta-base-language-detection",
        description="XLM-RoBERTa pre-finetuned on 20-language detection.",
        notes="Warm-started on language detection. May need 1–2 fewer epochs.",
    ),
    "muril": TransformerSpec(
        key="muril",
        hf_name="google/muril-base-cased",
        description="MuRIL — trained on transliterated South Asian languages.",
        notes="Best for Tamil/Tanglish. Ensemble partner with xlmr-base.",
    ),
    "mbert": TransformerSpec(
        key="mbert",
        hf_name="bert-base-multilingual-cased",
        description="mBERT — older multilingual BERT baseline (104 languages, Wikipedia).",
        notes="Weaker than XLM-R on Sinhala, weaker than MuRIL on Tamil. Baseline only.",
    ),
    # To add new models (encoder-only only):
    # "indic-bert": TransformerSpec("indic-bert", "ai4bharat/indic-bert", "...", params_millions=..., ...),
}

TRAIN_CONFIG: dict[str, int | float | bool | str] = {
    "max_length": 64,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.06,
    "weight_decay": 0.01,
    "fp16": True,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "recall_1",
    "greater_is_better": True,
    "logging_steps": 50,
    "dataloader_num_workers": 2,
    # Disk-space guards (critical on Kaggle /kaggle/working — 20 GB limit).
    # save_total_limit=2  → keep only the latest checkpoint + the best checkpoint;
    #                       prevents 3 full checkpoints (3 epochs × ~1.5 GB) accumulating.
    # save_only_model=True → skip optimizer/scheduler state in checkpoints (~1 GB each);
    #                        safe because we never resume mid-training on Kaggle.
    "save_total_limit": 2,
    "save_only_model": True,
}

HPO_SEARCH_SPACE: dict[str, tuple] = {
    "learning_rate": ("float_log", 1e-5, 5e-5),
    "per_device_train_batch_size": ("categorical", [16, 32, 64]),
    "num_train_epochs": ("categorical", [2, 3, 5]),
    "warmup_ratio": ("float", 0.0, 0.10),
    "weight_decay": ("categorical", [0.0, 0.01, 0.1]),
    "max_length": ("categorical", [64, 128]),
}
