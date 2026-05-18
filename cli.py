#!/usr/bin/env python
"""
cli.py — Development and testing tool.

Generation and split have their own phase files:
    python phases/phase_2_generate.py    ← main generation
    python phases/phase_4_split.py       ← split into train/val/test

This file is for interactive development and testing:
    python cli.py distribution
    python cli.py preview  --language singlish_light --industry banking --scenario continuation
    python cli.py dryrun   --language singlish_light --industry banking --scenario continuation --n 5
    python cli.py examples --language singlish_light --industry banking --scenario location_proximity
    python cli.py examples-all             # generate examples for all 352 cells (parallel)
    python cli.py examples-all --workers 5 # slower, lighter on API usage
    python cli.py examples-all --force     # regenerate all, even if cached
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.config.keys import IndustryKey, LanguageKey, ScenarioKey


def cmd_preview(args: argparse.Namespace) -> None:
    from backend.generation.prompt_factory import PROMPT_FACTORY

    prompt = PROMPT_FACTORY.build_preview(
        LanguageKey(args.language),
        IndustryKey(args.industry),
        ScenarioKey(args.scenario),
    )
    print("=" * 70)
    print(f"SYSTEM PROMPT: {args.language} × {args.industry} × {args.scenario}")
    print("=" * 70)
    print(prompt)


def cmd_dryrun(args: argparse.Namespace) -> None:
    from backend.generation.generator import GeneratorService
    from backend.generation.pymodels import GenerationCell

    cell = GenerationCell(
        language=LanguageKey(args.language),
        industry=IndustryKey(args.industry),
        scenario=ScenarioKey(args.scenario),
        target_count=args.n,
    )
    print(f"\nDry run: {cell.cell_id} | n={args.n}")
    print("-" * 60)

    rows = GeneratorService().generate_cell(cell)
    if not rows:
        print("No rows generated. Check OPENAI_API_KEY in .env")
        return

    print(f"Generated {len(rows)} rows  label={rows[0]['label']}\n")
    for i, row in enumerate(rows, 1):
        print(f"{i:2}. [{row['word_count']}w] {row['text']}")


def cmd_examples(args: argparse.Namespace) -> None:
    from backend.generation.example_store import example_store

    cached = example_store.is_cached(
        LanguageKey(args.language),
        IndustryKey(args.industry),
        ScenarioKey(args.scenario),
    )
    examples = example_store.get(
        LanguageKey(args.language),
        IndustryKey(args.industry),
        ScenarioKey(args.scenario),
    )
    source = "cell-specific (examples.json)" if cached else "LengthRange fallback"
    print(f"\nExamples: {args.language} × {args.industry} × {args.scenario}  [{source}]")
    print("-" * 60)
    for i, ex in enumerate(examples, 1):
        print(f"{i}. {ex}")


def cmd_examples_all(args: argparse.Namespace) -> None:
    """Generate examples for all active cells in parallel.

    Sequential time at 40s/call: ~4 hours.
    With 20 workers: ~12 minutes.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from backend.config.distribution import DISTRIBUTION
    from backend.generation.example_store import example_store

    cells = [c for c in DISTRIBUTION.all_cells_including_zero() if c.target_count > 0]
    to_gen = [
        c
        for c in cells
        if args.force or not example_store.is_cached(c.language, c.industry, c.scenario)
    ]
    skipped = len(cells) - len(to_gen)
    workers = args.workers
    est_min = max(1, len(to_gen) * 40 // workers // 60 + 1)

    print(f"\nGenerating examples for {len(to_gen)} cells ({skipped} already cached)")
    print(f"Workers: {workers}  |  Estimated time: ~{est_min} min\n")

    done = failed = 0

    def generate_one(cell) -> tuple[str, str]:
        try:
            example_store.get(cell.language, cell.industry, cell.scenario, force_regenerate=True)
            return cell.cell_id, "✓"
        except Exception as exc:  # noqa: BLE001
            return cell.cell_id, f"✗ {exc}"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(generate_one, c): c for c in to_gen}
        for i, future in enumerate(as_completed(futures), 1):
            cell_id, status = future.result()
            if status == "✓":
                done += 1
            else:
                failed += 1
            print(f"[{i:3}/{len(to_gen)}] {status} {cell_id}")

    print(f"\nDone. Generated: {done}, Failed: {failed}, Skipped (cached): {skipped}")


def cmd_distribution(args: argparse.Namespace) -> None:  # noqa: ARG001
    from backend.config.distribution import DISTRIBUTION

    summary = DISTRIBUTION.summary()
    total = sum(summary.values())
    print("\nDISTRIBUTION SUMMARY")
    print("=" * 55)
    for lang, count in sorted(summary.items(), key=lambda x: -x[1]):
        bar = "█" * int(count / total * 40)
        print(f"  {lang:<22} {count:>6,}  {bar}")
    print(f"\n  TOTAL raw target: {DISTRIBUTION.global_total:,}")
    print(f"  Active cells:     {len(DISTRIBUTION.to_cells())}")
    print(f"  All cells:        {len(DISTRIBUTION.all_cells_including_zero())}")

    print("\nTOP 10 CELLS:")
    for cell in sorted(DISTRIBUTION.to_cells(), key=lambda c: -c.target_count)[:10]:
        print(f"  {cell.cell_id:<55} {cell.target_count:>5}")

    print("\nPER-LANGUAGE BREAKDOWN:")
    for lang_b in DISTRIBUTION.languages:
        print(f"\n  {lang_b.language} ({lang_b.fraction:.1%} = {lang_b.computed_count:,} rows):")
        for ind_b in sorted(lang_b.industries, key=lambda x: -x.computed_count)[:4]:
            print(
                f"    {ind_b.industry:<14} {ind_b.fraction:.1%} = {ind_b.computed_count:>5,} rows"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sprout Router dev CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def cell_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--language", required=True, choices=list(LanguageKey))
        p.add_argument("--industry", required=True, choices=list(IndustryKey))
        p.add_argument("--scenario", required=True, choices=list(ScenarioKey))

    p = sub.add_parser("preview", help="Print system prompt (no API call)")
    cell_args(p)

    p = sub.add_parser("dryrun", help="Generate N sentences (no save)")
    cell_args(p)
    p.add_argument("--n", type=int, default=5)

    p = sub.add_parser("examples", help="Show examples for one cell")
    cell_args(p)

    p = sub.add_parser("examples-all", help="Generate examples for all cells in parallel")
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=20)

    sub.add_parser("distribution", help="Print full distribution breakdown")

    args = parser.parse_args()
    {
        "preview": cmd_preview,
        "dryrun": cmd_dryrun,
        "examples": cmd_examples,
        "examples-all": cmd_examples_all,
        "distribution": cmd_distribution,
    }[args.command](args)


if __name__ == "__main__":
    main()
