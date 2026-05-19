"""
Phase 2 — Training Data Generation.

Generates ~60,000 training rows across 352 cells.
Each cell (language × industry × scenario) runs its own multi-turn
OpenAI conversation in a separate thread.

Workers: default 10, max 10.
    10 workers is the tested safe limit on Tier 1/2 API access.
    Higher values trigger rate limit errors (429). The OpenAI SDK handles
    transient 429s internally, but sustained overload wastes time.
    DO NOT run with more than 10 workers.

DO NOT run phase_1_grounding.py at the same time.
    They share the same API key and rate limit budget.

DO NOT start uvicorn with --reload during generation.
    File watcher restarts the server when checkpoint.csv changes.

Speed estimate (60k rows / ~4 turns per cell / 352 cells):
    10 workers → ~18 min

After completion:
    python phases/phase_3_split.py

Usage:
    python phases/phase_2_generate.py                   # all cells, 10 workers
    python phases/phase_2_generate.py --workers 5       # fewer workers if needed
    python phases/phase_2_generate.py --resume          # continue from checkpoint
    python phases/phase_2_generate.py --language singlish_light  # one language only
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
from backend.generation.generator import MAX_WORKERS, GeneratorService


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
        default=MAX_WORKERS,
        help=f"Worker threads (parallel cells). Default {MAX_WORKERS}, max {MAX_WORKERS}.",
    )
    args = parser.parse_args()

    if args.workers > MAX_WORKERS:
        print(f"Warning: workers capped at {MAX_WORKERS} (requested {args.workers})")
        args.workers = MAX_WORKERS

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
