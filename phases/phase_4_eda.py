"""
Phase 4 — EDA (Exploratory Data Analysis).

Sanity-check that generated data matches the designed distribution.
Run after phase_3_split.py.

    python phases/phase_4_eda.py
    python phases/phase_4_eda.py --dataset v1

Output:
    experiments/{dataset}/eda_plots/label_distribution.png
    experiments/{dataset}/eda_plots/language_distribution.png
    experiments/{dataset}/eda_plots/scenario_distribution.png
    experiments/{dataset}/eda_plots/word_length_by_label.png
    experiments/{dataset}/eda_plots/word_length_by_language.png
    experiments/{dataset}/eda_summary.json
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


def run_eda(dataset_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from backend.shared.path_resolver import get_dataset_path, get_experiment_path

    dataset_dir = get_dataset_path(dataset_name)
    plots_dir = get_experiment_path(dataset_name, "eda_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_path = dataset_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found. Run phase_3_split.py first.")

    dfs: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        path = dataset_dir / f"{split}.csv"
        if path.exists():
            dfs[split] = pd.read_csv(path)

    df = dfs["train"]
    log.info(f"Loaded train: {len(df):,} rows")

    # ── Summary stats ─────────────────────────────────────────────────────────
    summary = {
        "dataset_name": dataset_name,
        "train_rows": len(df),
        "val_rows": len(dfs.get("val", pd.DataFrame())),
        "test_rows": len(dfs.get("test", pd.DataFrame())),
        "label_0_count": int((df["label"] == 0).sum()),
        "label_1_count": int((df["label"] == 1).sum()),
        "label_0_pct": round(float((df["label"] == 0).mean() * 100), 2),
        "label_1_pct": round(float((df["label"] == 1).mean() * 100), 2),
        "mean_word_count": round(float(df["word_count"].mean()), 2),
        "median_word_count": float(df["word_count"].median()),
        "languages": df["language"].value_counts().to_dict(),
        "scenarios": df["scenario"].value_counts().to_dict(),
    }

    print(f"\n{'═' * 55}")
    print(f"  EDA — {dataset_name} (train set)")
    print(f"{'═' * 55}")
    print(f"  Train rows:  {summary['train_rows']:,}")
    print(f"  Label=0:     {summary['label_0_count']:,}  ({summary['label_0_pct']}%)")
    print(f"  Label=1:     {summary['label_1_count']:,}  ({summary['label_1_pct']}%)")
    print(f"  Mean words:  {summary['mean_word_count']:.1f}")
    print("\n  Language distribution:")
    for lang, cnt in sorted(summary["languages"].items(), key=lambda x: -x[1]):
        print(f"    {lang:<22} {cnt:>6,}  ({cnt / summary['train_rows'] * 100:.1f}%)")
    print("\n  Top scenarios:")
    for sc, cnt in sorted(summary["scenarios"].items(), key=lambda x: -x[1])[:5]:
        print(f"    {sc:<25} {cnt:>6,}  ({cnt / summary['train_rows'] * 100:.1f}%)")
    print(f"{'═' * 55}\n")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plt.style.use("seaborn-v0_8-darkgrid")
    COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    # 1. Label distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["label"].value_counts().sort_index()
    bars = ax.bar(
        ["label=0\n(gpt-4o-mini)", "label=1\n(gpt-4o)"],
        counts.values,
        color=COLORS[:2],
    )
    for bar, v in zip(bars, counts.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 50,
            f"{v:,}\n({v / len(df) * 100:.1f}%)",
            ha="center",
            fontsize=10,
        )
    ax.set_title(f"Label Distribution — {dataset_name} train")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(plots_dir / "label_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Language distribution
    lang_counts = df["language"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(lang_counts.index, lang_counts.values, color=COLORS[: len(lang_counts)])
    ax.set_title(f"Language Distribution — {dataset_name} train")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    for bar, v in zip(bars, lang_counts.values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 30, f"{v:,}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "language_distribution.png", dpi=150)
    plt.close(fig)

    # 3. Scenario distribution
    sc_counts = df["scenario"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(sc_counts.index, sc_counts.values, color="#3F51B5")
    ax.set_title(f"Scenario Distribution — {dataset_name} train")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    for bar, v in zip(bars, sc_counts.values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 20, f"{v:,}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "scenario_distribution.png", dpi=150)
    plt.close(fig)

    # 4. Word length by label
    fig, ax = plt.subplots(figsize=(8, 4))
    for label_val, color in [(0, COLORS[0]), (1, COLORS[1])]:
        subset = df[df["label"] == label_val]["word_count"]
        ax.hist(subset, bins=40, alpha=0.6, color=color, label=f"label={label_val}")
    ax.set_title(f"Word Count by Label — {dataset_name} train")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "word_length_by_label.png", dpi=150)
    plt.close(fig)

    # 5. Word length by language (box plot)
    # tick_labels is the correct parameter in matplotlib 3.9+
    lang_order = df["language"].value_counts().index.tolist()
    data = [df[df["language"] == lang]["word_count"].values for lang in lang_order]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(
        data,
        tick_labels=lang_order,  # was `labels` — renamed in matplotlib 3.9
        patch_artist=True,
        boxprops={"facecolor": "#E3F2FD"},
    )
    ax.set_title(f"Word Count by Language — {dataset_name} train")
    ax.set_ylabel("Word count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plots_dir / "word_length_by_language.png", dpi=150)
    plt.close(fig)

    # ── Save summary ──────────────────────────────────────────────────────────
    exp_dir = get_experiment_path(dataset_name, "eda_plots").parent
    summary_path = exp_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    log.info("EDA complete.")
    log.info(f"  Plots:   {plots_dir}")
    log.info(f"  Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="v1")
    args = parser.parse_args()
    run_eda(args.dataset)


if __name__ == "__main__":
    main()
