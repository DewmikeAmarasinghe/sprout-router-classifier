"""
Flat configuration constants for the Sprout router classifier.

All values are read via settings_manager.get("KEY").
Sensitive values (API keys, DB passwords) go in .env — never here.

PAUSE TUNING:
    PAUSE_AFTER_N_CELLS=2 means all worker threads block before their next API
    call after every 2 cells complete. In-flight calls finish normally (responses
    already in transit arrive); only NEW outgoing calls are gated.

    At 7 workers, 2 cells typically complete every ~20-40s. Combined with a 90s
    pause this gives ~2 min of breathing room per cycle — enough for the 200k
    TPM window to reset between bursts of concurrent multi-turn calls.

    The 429 storms visible in logs happen when large cells (20+ turns) run
    simultaneously and accumulate token usage. Pausing every 2 cells catches
    pressure before it snowballs. The while-loop generator recovers failed turns
    automatically — occasional 429s do not cause permanent row loss.
"""

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_VERSION = "v1"

# ── Generation ────────────────────────────────────────────────────────────────
GENERATION_LLM = "gpt-5-nano"  # model used to generate training data
GENERATION_BATCH_SIZE = 50  # messages requested per API call / turn

# ── Generation concurrency and rate-limit protection ──────────────────────────
MAX_GENERATION_WORKERS = 10  # hard cap — ThreadPoolExecutor max_workers
PAUSE_AFTER_N_CELLS = 2  # pause all workers after every N cells complete
CHECKPOINT_PAUSE_SECONDS = 90  # seconds to pause (all workers blocked) per cycle
CHECKPOINT_EVERY = 500  # write checkpoint.csv every ~N new rows

# ── Router ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70  # default routing threshold (tuned by phase_8)
SAFE_DEFAULT_LABEL = 1  # label to use when model confidence < threshold
# 1 = gpt-4o (safe default — never under-routes)

# ── Evaluation ────────────────────────────────────────────────────────────────
PRODUCTION_RECALL_THRESHOLD = 0.97  # minimum recall_1 to pass production gate
DAILY_MESSAGES_ESTIMATE = 10_000  # used in cost simulation

# ── API pricing (USD per 1M tokens, as of mid-2025) ──────────────────────────
GPT4O_INPUT_PER_1M = 2.50
GPT4O_OUTPUT_PER_1M = 10.00
GPT4O_MINI_INPUT_PER_1M = 0.15
GPT4O_MINI_OUTPUT_PER_1M = 0.60
AVG_INPUT_TOKENS = 200
AVG_OUTPUT_TOKENS = 150
