"""
Phase 2 — Training Data Generation.

Generates ~60,000 training rows in parallel across 352 cells.
Each cell (language × industry × scenario) runs its own multi-turn
OpenAI conversation in a separate thread.

Rate limit protection:
    A global semaphore (API_CONCURRENCY_LIMIT in config/settings.py) caps
    concurrent in-flight API calls regardless of worker count.
    Default: 15 concurrent calls. Adjust based on your OpenAI tier:
        Tier 1 (< $50):   set to 10
        Tier 2 ($50–$500): set to 15 (default)
        Tier 3+ ($500+):  set to 20–30

    DO NOT run phase_1_grounding.py at the same time — they share the
    same API key and rate limit budget.

Usage:
    python phases/phase_2_generate.py                  # all cells, 20 workers
    python phases/phase_2_generate.py --workers 10     # fewer workers (safer rate limits)
    python phases/phase_2_generate.py --workers 40     # max workers
    python phases/phase_2_generate.py --resume         # continue from checkpoint
    python phases/phase_2_generate.py --language singlish_light  # one language only

Speed estimates (60k rows / ~4 turns per cell):
    10 workers  →  ~36 min
    20 workers  →  ~18 min  (default)
    40 workers  →  ~10 min  (only safe on Tier 3+)

After completion:
    python phases/phase_3_split.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

from backend.config.keys import IndustryKey, LanguageKey, ScenarioKey
from backend.generation.generator import GeneratorService


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="v1")
    parser.add_argument("--language", default=None, choices=list(LanguageKey))
    parser.add_argument("--industry", default=None, choices=list(IndustryKey))
    parser.add_argument("--scenario", default=None, choices=list(ScenarioKey))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed cells from checkpoint.csv (use only after interrupted run)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Worker threads (parallel cells). Default 20, max 40.",
    )
    args = parser.parse_args()

    if args.workers > 40:
        print(f"Warning: workers capped at 40 (requested {args.workers})")
        args.workers = 40

    GeneratorService().run(
        dataset_name=args.dataset,
        language=LanguageKey(args.language) if args.language else None,
        industry=IndustryKey(args.industry) if args.industry else None,
        scenario=ScenarioKey(args.scenario) if args.scenario else None,
        resume=args.resume,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
