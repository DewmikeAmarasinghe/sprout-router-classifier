"""
Phase 4 — Exploratory Data Analysis.

Run AFTER phase_3_split.py to see actual generated data distribution.
Run with --planned BEFORE generation to see the expected planned distribution.

Usage:
    python phases/phase_4_eda.py                   # actual EDA on generated train.csv
    python phases/phase_4_eda.py --planned          # planned EDA from distribution.py
    python phases/phase_4_eda.py --dataset v2       # specify dataset version
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def run_actual_eda(dataset: str) -> None:
    """EDA on generated train.csv — run after phase_3_split.py."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from backend.shared.path_resolver import get_dataset_path, get_experiment_path

    dataset_dir = get_dataset_path(dataset)
    train_path = dataset_dir / "train.csv"

    if not train_path.exists():
        log.error(f"{train_path} not found. Run phase_3_split.py first.")
        sys.exit(1)

    df = pd.read_csv(train_path)
    val_df = (
        pd.read_csv(dataset_dir / "val.csv")
        if (dataset_dir / "val.csv").exists()
        else pd.DataFrame()
    )
    test_df = (
        pd.read_csv(dataset_dir / "test.csv")
        if (dataset_dir / "test.csv").exists()
        else pd.DataFrame()
    )

    log.info(f"Loaded train: {len(df):,} rows")

    plots_dir = get_experiment_path(dataset, "eda_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_name": dataset,
        "train_rows": len(df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "label_0_count": int((df["label"] == 0).sum()),
        "label_1_count": int((df["label"] == 1).sum()),
        "label_0_pct": round(float((df["label"] == 0).mean() * 100), 2),
        "label_1_pct": round(float((df["label"] == 1).mean() * 100), 2),
        "mean_word_count": round(float(df["word_count"].mean()), 2),
        "median_word_count": float(df["word_count"].median()),
        "languages": df["language"].value_counts().to_dict(),
        "scenarios": df["scenario"].value_counts().to_dict(),
    }

    exp_dir = plots_dir.parent
    (exp_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
    plt.style.use("seaborn-v0_8-darkgrid")

    plot_label_distribution(df, dataset, plots_dir, COLORS)
    plot_language_distribution(df, dataset, plots_dir, COLORS)
    plot_scenario_distribution(df, dataset, plots_dir)
    plot_word_length_by_label(df, dataset, plots_dir, COLORS)
    plot_word_length_by_language(df, dataset, plots_dir)

    print("\n" + "═" * 55)
    print(f"  EDA — {dataset} (train set)")
    print("═" * 55)
    print(f"  Train rows:  {len(df):,}")
    print(f"  Label=0:     {summary['label_0_count']:,}  ({summary['label_0_pct']}%)")
    print(f"  Label=1:     {summary['label_1_count']:,}  ({summary['label_1_pct']}%)")
    print(f"  Mean words:  {summary['mean_word_count']}")
    print("  Language distribution:")
    for lang, cnt in sorted(summary["languages"].items(), key=lambda x: -x[1]):
        print(f"    {lang:<22} {cnt:>6,}  ({cnt / len(df) * 100:.1f}%)")
    print("  Top scenarios:")
    for sc, cnt in list(sorted(summary["scenarios"].items(), key=lambda x: -x[1]))[:5]:
        print(f"    {sc:<25} {cnt:>6,}  ({cnt / len(df) * 100:.1f}%)")
    print("═" * 55)
    log.info("EDA complete.")
    log.info(f"  Plots:   {plots_dir}")
    log.info(f"  Summary: {plots_dir.parent / 'eda_summary.json'}")


def run_planned_eda(dataset: str) -> None:
    """Planned EDA from distribution.py — run BEFORE generation.

    Shows the expected label, language, and scenario distributions based
    on the fractions configured in distribution.py. No CSV files needed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from backend.config.distribution import DISTRIBUTION
    from backend.config.keys import LABEL_0_SCENARIOS, LanguageKey
    from backend.shared.path_resolver import get_experiment_path

    plots_dir = get_experiment_path(dataset, "eda_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    cells = DISTRIBUTION.to_cells()
    total_rows = DISTRIBUTION.global_total

    label_0 = sum(
        c.target_count
        for c in cells
        if c.language == LanguageKey.PURE_ENGLISH and c.scenario in LABEL_0_SCENARIOS
    )
    label_1 = total_rows - label_0

    lang_counts = DISTRIBUTION.summary()
    scenario_counts: dict[str, int] = {}
    for c in cells:
        scenario_counts[c.scenario] = scenario_counts.get(c.scenario, 0) + c.target_count

    COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
    plt.style.use("seaborn-v0_8-darkgrid")

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = [label_0, label_1]
    bars = ax.bar(["label=0\n(gpt-4o-mini)", "label=1\n(gpt-4o)"], counts, color=COLORS[:2])
    for bar, v in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 50,
            f"{v:,}\n({v / total_rows * 100:.1f}%)",
            ha="center",
            fontsize=10,
        )
    ax.set_title(f"Planned Label Distribution — {dataset} (from distribution.py)")
    ax.set_ylabel("Target rows")
    fig.tight_layout()
    fig.savefig(plots_dir / "planned_label_distribution.png", dpi=150)
    plt.close(fig)

    lang_sorted = sorted(lang_counts.items(), key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        [k for k, _ in lang_sorted], [v for _, v in lang_sorted], color=COLORS[: len(lang_sorted)]
    )
    for bar, (_, v) in zip(bars, lang_sorted, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 50, f"{v:,}", ha="center", fontsize=9)
    ax.set_title(f"Planned Language Distribution — {dataset}")
    ax.set_ylabel("Target rows")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plots_dir / "planned_language_distribution.png", dpi=150)
    plt.close(fig)

    sc_sorted = sorted(scenario_counts.items(), key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar([k for k, _ in sc_sorted], [v for _, v in sc_sorted], color="#3F51B5")
    for bar, (_, v) in zip(bars, sc_sorted, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 20, f"{v:,}", ha="center", fontsize=8)
    ax.set_title(f"Planned Scenario Distribution — {dataset}")
    ax.set_ylabel("Target rows")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plots_dir / "planned_scenario_distribution.png", dpi=150)
    plt.close(fig)

    print("\n" + "═" * 60)
    print(f"  PLANNED EDA — {dataset} (from distribution.py)")
    print("═" * 60)
    print(f"  Global target:  {total_rows:,} rows")
    print(f"  Label=0 (mini): {label_0:,}  ({label_0 / total_rows * 100:.1f}%)")
    print(f"  Label=1 (4o):   {label_1:,}  ({label_1 / total_rows * 100:.1f}%)")
    print("\n  Planned language split:")
    for lang, cnt in lang_sorted:
        print(f"    {lang:<22} {cnt:>6,}  ({cnt / total_rows * 100:.1f}%)")
    print("\n  Planned scenario split (all languages combined):")
    for sc, cnt in sc_sorted:
        print(f"    {sc:<28} {cnt:>6,}  ({cnt / total_rows * 100:.1f}%)")
    print("═" * 60)
    log.info(f"Planned EDA plots saved to: {plots_dir}")


# ── Plot helpers ──────────────────────────────────────────────────────────────


def plot_label_distribution(df, dataset, plots_dir, COLORS):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["label"].value_counts().sort_index()
    bars = ax.bar(["label=0\n(gpt-4o-mini)", "label=1\n(gpt-4o)"], counts.values, color=COLORS[:2])
    for bar, v in zip(bars, counts.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 50,
            f"{v:,}\n({v / len(df) * 100:.1f}%)",
            ha="center",
            fontsize=10,
        )
    ax.set_title(f"Label Distribution — {dataset} train")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(plots_dir / "label_distribution.png", dpi=150)
    plt.close(fig)


def plot_language_distribution(df, dataset, plots_dir, COLORS):
    import matplotlib.pyplot as plt

    lang_counts = df["language"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(lang_counts.index, lang_counts.values, color=COLORS[: len(lang_counts)])
    ax.set_title(f"Language Distribution — {dataset} train")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    for bar, v in zip(bars, lang_counts.values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 30, f"{v:,}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "language_distribution.png", dpi=150)
    plt.close(fig)


def plot_scenario_distribution(df, dataset, plots_dir):
    import matplotlib.pyplot as plt

    sc_counts = df["scenario"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(sc_counts.index, sc_counts.values, color="#3F51B5")
    ax.set_title(f"Scenario Distribution — {dataset} train")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    for bar, v in zip(bars, sc_counts.values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 20, f"{v:,}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "scenario_distribution.png", dpi=150)
    plt.close(fig)


def plot_word_length_by_label(df, dataset, plots_dir, COLORS):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for label_val, color in [(0, COLORS[0]), (1, COLORS[1])]:
        ax.hist(
            df[df["label"] == label_val]["word_count"],
            bins=40,
            alpha=0.6,
            color=color,
            label=f"label={label_val}",
        )
    ax.set_title(f"Word Count by Label — {dataset} train")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "word_length_by_label.png", dpi=150)
    plt.close(fig)


def plot_word_length_by_language(df, dataset, plots_dir):
    import matplotlib.pyplot as plt

    lang_order = df["language"].value_counts().index.tolist()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(
        [df[df["language"] == lang]["word_count"].values for lang in lang_order],
        tick_labels=lang_order,
        patch_artist=True,
        boxprops={"facecolor": "#E3F2FD"},
    )
    ax.set_title(f"Word Count by Language — {dataset} train")
    ax.set_ylabel("Word count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plots_dir / "word_length_by_language.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="v1")
    parser.add_argument(
        "--planned",
        action="store_true",
        help="Show planned distribution from distribution.py (no CSV needed)",
    )
    args = parser.parse_args()

    if args.planned:
        run_planned_eda(args.dataset)
    else:
        run_actual_eda(args.dataset)


if __name__ == "__main__":
    main()
