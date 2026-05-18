"""
Global constants — single source of truth for all configuration.

Do NOT import from this file directly in application code.
Use: from backend.shared.settings_manager import settings_manager
     value = settings_manager.get("GENERATION_LLM")

RATE LIMITING:
    API_CONCURRENCY_LIMIT controls the global threading.Semaphore that caps
    concurrent in-flight API calls across all worker threads.
    This is the primary defence against 429 burst errors.

    Rule of thumb:
        Tier 1 (< $50 spend):   API_CONCURRENCY_LIMIT = 10
        Tier 2 ($50–$500):      API_CONCURRENCY_LIMIT = 15
        Tier 3+ ($500+):        API_CONCURRENCY_LIMIT = 20–30

    Keep API_CONCURRENCY_LIMIT <= max_workers in phase_2_generate.py.
    Having more workers than the limit is fine — extra workers wait on the semaphore.
"""

from __future__ import annotations

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_VERSION: str = "v1"

# ── Generation ────────────────────────────────────────────────────────────────
GENERATION_LLM: str = "gpt-5-nano"
GENERATION_BATCH_SIZE: int = 50  # rows per API call
CHECKPOINT_EVERY: int = 500  # write checkpoint after this many total rows

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Max concurrent in-flight OpenAI API calls across all worker threads.
# Lower this if you are on Tier 1 (< $50 spend). Raise it for Tier 3+.
# Setting this lower than max_workers in phase_2_generate.py is safe —
# excess workers just wait for a semaphore slot.
API_CONCURRENCY_LIMIT: int = 15

# ── Routing ───────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.6  # below this → SAFE_DEFAULT_LABEL
SAFE_DEFAULT_LABEL: int = 1  # always route unclear messages to gpt-4o

# ── OpenAI pricing (per 1M tokens) ───────────────────────────────────────────
GPT4O_INPUT_PER_1M: float = 2.50
GPT4O_CACHED_INPUT_PER_1M: float = 1.25
GPT4O_OUTPUT_PER_1M: float = 10.00
GPT4O_MINI_INPUT_PER_1M: float = 0.15
GPT4O_MINI_CACHED_INPUT_PER_1M: float = 0.075
GPT4O_MINI_OUTPUT_PER_1M: float = 0.60

# ── Transformer training defaults ─────────────────────────────────────────────
TRANSFORMER_MAX_LENGTH: int = 64  # 95th pct of Sprout messages < 50 tokens
TRANSFORMER_NUM_EPOCHS: int = 3
TRANSFORMER_BATCH_SIZE: int = 32  # use 16 for xlmr-large (OOM risk at 32)
TRANSFORMER_LEARNING_RATE: float = 2e-5
TRANSFORMER_WARMUP_RATIO: float = 0.06
TRANSFORMER_WEIGHT_DECAY: float = 0.01
TRANSFORMER_FP16: bool = True  # set False for CPU runs
