"""
EDA tab panel.

Runs the same analysis as phase_4_eda.py but directly in the UI.
EDA logic lives here — no import from phases/ needed.

Shows: 5 distribution plots + summary stats.
"""

from __future__ import annotations

import json
import logging

import gradio as gr

from backend.shared.path_resolver import get_dataset_path, get_experiment_path
from backend.shared.settings_manager import settings_manager

log = logging.getLogger(__name__)

PLOT_NAMES = [
    "label_distribution.png",
    "language_distribution.png",
    "scenario_distribution.png",
    "word_length_by_label.png",
    "word_length_by_language.png",
]

PlotTuple = tuple[str | None, str | None, str | None, str | None, str | None]
ResultTuple = tuple[str, str | None, str | None, str | None, str | None, str | None]


def compute_and_save_eda(dataset: str) -> None:
    """Run EDA on train.csv and save plots + summary JSON."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    dataset_dir = get_dataset_path(dataset)
    plots_dir = get_experiment_path(dataset, "eda_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_path = dataset_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found. Run phase_3_split.py first.")

    df = pd.read_csv(train_path)

    summary = {
        "dataset_name": dataset,
        "train_rows": len(df),
        "val_rows": len(pd.read_csv(dataset_dir / "val.csv"))
        if (dataset_dir / "val.csv").exists()
        else 0,
        "test_rows": len(pd.read_csv(dataset_dir / "test.csv"))
        if (dataset_dir / "test.csv").exists()
        else 0,
        "label_0_count": int((df["label"] == 0).sum()),
        "label_1_count": int((df["label"] == 1).sum()),
        "label_0_pct": round(float((df["label"] == 0).mean() * 100), 2),
        "label_1_pct": round(float((df["label"] == 1).mean() * 100), 2),
        "mean_word_count": round(float(df["word_count"].mean()), 2),
        "median_word_count": float(df["word_count"].median()),
        "languages": df["language"].value_counts().to_dict(),
        "scenarios": df["scenario"].value_counts().to_dict(),
    }

    exp_dir = get_experiment_path(dataset, "eda_plots").parent
    (exp_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
    plt.style.use("seaborn-v0_8-darkgrid")

    # 1. Label distribution
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

    # 2. Language distribution
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

    # 3. Scenario distribution
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

    # 4. Word length by label
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

    # 5. Word length by language (box plot)
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

    log.info(f"EDA complete for '{dataset}'. Plots: {plots_dir}")


def load_summary(dataset: str) -> str:
    """Format eda_summary.json as readable text."""
    summary_path = get_experiment_path(dataset, "eda_plots").parent / "eda_summary.json"
    if not summary_path.exists():
        return "No EDA results yet. Click 'Run EDA'."

    data = json.loads(summary_path.read_text())
    n = data["train_rows"]
    lines = [
        f"Dataset: {data['dataset_name']}",
        f"Train: {n:,}  Val: {data.get('val_rows', 0):,}  Test: {data.get('test_rows', 0):,}",
        "",
        f"Label=0 (gpt-4o-mini): {data['label_0_count']:,}  ({data['label_0_pct']}%)",
        f"Label=1 (gpt-4o):      {data['label_1_count']:,}  ({data['label_1_pct']}%)",
        f"Mean word count: {data['mean_word_count']}",
        "",
        "Language distribution:",
    ]
    for lang, cnt in sorted(data.get("languages", {}).items(), key=lambda x: -x[1]):
        lines.append(f"  {lang:<22} {cnt:>6,}  ({cnt / n * 100:.1f}%)")
    lines += ["", "Scenario distribution:"]
    for sc, cnt in sorted(data.get("scenarios", {}).items(), key=lambda x: -x[1]):
        lines.append(f"  {sc:<25} {cnt:>6,}  ({cnt / n * 100:.1f}%)")
    return "\n".join(lines)


def load_plots(dataset: str) -> PlotTuple:
    """Return file paths for the 5 EDA plots (None if not yet generated)."""
    plots_dir = get_experiment_path(dataset, "eda_plots")
    paths = [str(plots_dir / name) if (plots_dir / name).exists() else None for name in PLOT_NAMES]
    return paths[0], paths[1], paths[2], paths[3], paths[4]


def run_eda(dataset: str) -> ResultTuple:
    """Run EDA and return summary + 5 plot paths."""
    compute_and_save_eda(dataset)
    p1, p2, p3, p4, p5 = load_plots(dataset)
    return load_summary(dataset), p1, p2, p3, p4, p5


def refresh_display(dataset: str) -> ResultTuple:
    """Load existing EDA results without re-running."""
    p1, p2, p3, p4, p5 = load_plots(dataset)
    return load_summary(dataset), p1, p2, p3, p4, p5


def build() -> None:
    """Build the EDA tab."""

    gr.Markdown(
        "### Exploratory Data Analysis\n"
        "Sanity-check that generated data matches the designed distribution. "
        "Run **after** `phase_3_split.py` completes. Results also saved to "
        "`experiments/{dataset}/eda_plots/`."
    )

    dataset_box = gr.Textbox(
        value=settings_manager.get("DATASET_VERSION"),
        label="Dataset",
        max_lines=1,
    )

    with gr.Row():
        run_btn = gr.Button("▶ Run EDA", variant="primary")
        refresh_btn = gr.Button("🔄 Load Existing Results", variant="secondary")

    summary_box = gr.Textbox(
        label="Summary Statistics",
        lines=20,
        interactive=False,
        placeholder="Click 'Load Existing Results' if EDA was already run, or 'Run EDA' to generate.",
    )

    with gr.Row():
        plot_label = gr.Image(label="Label Distribution", type="filepath")
        plot_lang = gr.Image(label="Language Distribution", type="filepath")

    with gr.Row():
        plot_sc = gr.Image(label="Scenario Distribution", type="filepath")
        plot_label_wc = gr.Image(label="Word Count by Label", type="filepath")

    plot_lang_wc = gr.Image(label="Word Count by Language", type="filepath")

    all_outputs = [summary_box, plot_label, plot_lang, plot_sc, plot_label_wc, plot_lang_wc]

    run_btn.click(fn=run_eda, inputs=dataset_box, outputs=all_outputs)
    refresh_btn.click(fn=refresh_display, inputs=dataset_box, outputs=all_outputs)
