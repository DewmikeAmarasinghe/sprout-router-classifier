"""
Phase 4 — Stratified Train/Val/Test Split.

Reads data/datasets/{dataset}/raw/generated_raw.csv and creates:
    data/datasets/{dataset}/train.csv   (80%)
    data/datasets/{dataset}/val.csv     (10%)
    data/datasets/{dataset}/test.csv    (10%)

Stratified by label × scenario — every scenario appears proportionally
in every split.

Usage:
    python phases/phase_4_split.py
    python phases/phase_4_split.py --dataset v2

TEST SET POLICY:
    test.csv is accessed ONCE — only in final evaluation (Phase 8).
    All development uses train.csv and val.csv only.
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

from backend.generation.splitter import DataSplitter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="v1")
    args = parser.parse_args()

    DataSplitter().run(args.dataset)


if __name__ == "__main__":
    main()
