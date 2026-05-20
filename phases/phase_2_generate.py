"""
Phase 2 — Training Data Generation.

Generates ~60,000 training rows across 352 cells using multi-turn OpenAI calls.
Each cell = one (language × industry × scenario) combination.

RESUME vs FILL-GAPS:
    --resume:     Skip ALL cells in checkpoint.csv even if underfilled.
                  Use only after Ctrl+C to restart from where you stopped.

    --fill-gaps:  Detect cells with fewer rows than their target and top them up.
                  Use after any run that hit rate limits or produced partial output.
                  Works after Ctrl+C too.

PAUSE MECHANISM:
    All worker threads pause every PAUSE_AFTER_N_CELLS cells.
    In-flight API calls complete during the pause (can't cancel them once sent).
    The HTTP responses you see after a pause message are from already-sent calls
    — the pause IS working; it prevents NEW calls, not in-flight ones.

RATE LIMITS:
    Recommend --workers 6 for --fill-gaps runs (safer TPM budget than 10).
    10 workers is fine for an initial run — the while loop will retry any failed turns.

Usage:
    python phases/phase_2_generate.py                      # fresh run
    python phases/phase_2_generate.py --resume             # skip completed cells
    python phases/phase_2_generate.py --fill-gaps          # top up underfilled cells
    python phases/phase_2_generate.py --fill-gaps --workers 6
    python phases/phase_2_generate.py --language singlish_light
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
        help="Skip ALL cells in checkpoint.csv (even underfilled). Use only after Ctrl+C.",
    )
    parser.add_argument(
        "--fill-gaps",
        action="store_true",
        dest="fill_gaps",
        help="Detect underfilled cells and top them up. Works after Ctrl+C or any partial run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Worker threads (default {MAX_WORKERS}). Use 6 for --fill-gaps runs.",
    )
    args = parser.parse_args()

    if args.workers > MAX_WORKERS:
        print(f"Warning: workers capped at {MAX_WORKERS}")
        args.workers = MAX_WORKERS

    language = LanguageKey(args.language) if args.language else None
    industry = IndustryKey(args.industry) if args.industry else None
    scenario = ScenarioKey(args.scenario) if args.scenario else None

    svc = GeneratorService()

    if args.fill_gaps:
        svc.fill_gaps(
            dataset_name=args.dataset,
            language=language,
            industry=industry,
            scenario=scenario,
            max_workers=args.workers,
        )
    else:
        svc.run(
            dataset_name=args.dataset,
            language=language,
            industry=industry,
            scenario=scenario,
            resume=args.resume,
            max_workers=args.workers,
        )


if __name__ == "__main__":
    main()
