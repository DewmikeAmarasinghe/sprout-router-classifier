"""
Flat configuration constants for the Sprout router classifier.

All values are read via settings_manager.get("KEY").
Sensitive values (API keys, DB passwords) go in .env — never here.

PAUSE TUNING:
    PAUSE_AFTER_N_TURNS=3: all worker threads block before their next API
    call after every 3 API turns total (across all workers combined).
    Each turn generates ~75 rows, so a pause fires roughly every 225 rows.
    This creates a 100s cooldown, preventing TPM storms from large batches.
"""

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_VERSION = "v1"

# ── Generation ────────────────────────────────────────────────────────────────
GENERATION_LLM = "gpt-5-nano"  # model used to generate training data
GENERATION_BATCH_SIZE = 50  # messages requested per API call / turn

# ── Generation concurrency and rate-limit protection ──────────────────────────
MAX_GENERATION_WORKERS = 10  # hard cap — ThreadPoolExecutor max_workers
PAUSE_AFTER_N_TURNS = 3  # pause all workers after every N API turns (batches of ~75 rows)
CHECKPOINT_PAUSE_SECONDS = 90  # seconds to pause (all workers blocked) per cycle
CHECKPOINT_EVERY = 500  # write checkpoint.csv every ~N new rows

# ── Router ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70  # default routing threshold (tuned by phase_8)
SAFE_DEFAULT_LABEL = 1  # label to use when model confidence < threshold
# 1 = gpt-4o (safe default — never under-routes)

# ── Evaluation ────────────────────────────────────────────────────────────────
PRODUCTION_RECALL_THRESHOLD = 0.95  # minimum recall_1 to pass production gate
DAILY_MESSAGES_ESTIMATE = 25_000  # used in cost simulation (~$50/day at gpt-4o pricing)

# ── API pricing (USD per 1M tokens, as of mid-2025) ──────────────────────────
GPT4O_INPUT_PER_1M = 2.50
GPT4O_OUTPUT_PER_1M = 10.00
GPT4O_MINI_INPUT_PER_1M = 0.15
GPT4O_MINI_OUTPUT_PER_1M = 0.60
AVG_INPUT_TOKENS = 200
AVG_OUTPUT_TOKENS = 150
