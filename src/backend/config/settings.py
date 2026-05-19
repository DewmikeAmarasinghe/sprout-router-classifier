"""
Application settings. Edit here to tune behavior across all phases and the Gradio UI.
All values are accessed via settings_manager.get("KEY") — never import this file directly.

RATE LIMIT MANAGEMENT:
    gpt-5-nano TPM limit: 200,000 tokens/min.
    At 10 workers and batch_size=50, peak usage can hit 10 × 28,950 = 289,500 TPM.
    PAUSE_AFTER_N_CELLS: pause all threads after every N completed cells.
    CHECKPOINT_PAUSE_SECONDS: how long to pause (seconds).
    Default: pause 60s after every 10 cells — keeps peak TPM well under 200k.
"""

# ── Dataset & generation ──────────────────────────────────────────────────────
DATASET_VERSION: str = "v1"
GENERATION_LLM: str = "gpt-5-nano"
GENERATION_BATCH_SIZE: int = 50
CHECKPOINT_EVERY: int = 500
MAX_GENERATION_WORKERS: int = 10  # hard cap — higher values hit TPM rate limits

# Rate limit protection
PAUSE_AFTER_N_CELLS: int = 10  # pause every N completed cells
CHECKPOINT_PAUSE_SECONDS: int = 60  # seconds to pause

# ── Router ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.6
SAFE_DEFAULT_LABEL: int = 1

# ── Training: transformers ────────────────────────────────────────────────────
TRANSFORMER_MAX_LENGTH: int = 64
TRANSFORMER_NUM_EPOCHS: int = 3
TRANSFORMER_LEARNING_RATE: float = 2e-5
TRANSFORMER_WARMUP_RATIO: float = 0.06
TRANSFORMER_WEIGHT_DECAY: float = 0.01
TRANSFORMER_BATCH_SIZE: int = 32
TRANSFORMER_FP16: bool = True

# ── API pricing (USD per 1M tokens) ──────────────────────────────────────────
GPT4O_INPUT_PER_1M: float = 2.50
GPT4O_OUTPUT_PER_1M: float = 10.00
GPT4O_CACHED_INPUT_PER_1M: float = 1.25
GPT4O_MINI_INPUT_PER_1M: float = 0.15
GPT4O_MINI_OUTPUT_PER_1M: float = 0.60
GPT4O_MINI_CACHED_INPUT_PER_1M: float = 0.075
