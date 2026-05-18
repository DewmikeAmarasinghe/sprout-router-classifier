"""
Training tab panel — Classical ML experiments.

Runs the same experiments as phase_5_train_classical.py but from the UI.
Shows existing results from experiments/{dataset}/classical/models/ and
allows running new experiments.

Key metric: recall_1 >= 0.97 is required for production deployment.
"""

from __future__ import annotations

import json
import logging

import gradio as gr

from backend.shared.path_resolver import get_experiment_path
from backend.shared.settings_manager import settings_manager

log = logging.getLogger(__name__)


def vectorizer_choices() -> list[str]:
    from backend.training.classical.config import VECTORIZER_REGISTRY

    return list(VECTORIZER_REGISTRY.keys())


def classifier_choices() -> list[str]:
    from backend.training.classical.config import CLASSIFIER_REGISTRY

    return list(CLASSIFIER_REGISTRY.keys())


def load_existing_results(dataset: str) -> list[list]:
    """Load saved result.json files from all previous experiments."""
    models_dir = get_experiment_path(dataset, "classical") / "models"
    rows: list[list] = []

    if not models_dir.exists():
        return rows

    for model_dir in sorted(models_dir.iterdir()):
        result_path = model_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            r = json.loads(result_path.read_text())
            rows.append(
                [
                    r.get("vectorizer_key", ""),
                    r.get("classifier_key", ""),
                    f"{r.get('recall_1', 0):.4f}",
                    f"{r.get('precision_1', 0):.4f}",
                    f"{r.get('mcc', 0):.4f}",
                    f"{r.get('roc_auc', 0):.4f}",
                    f"{r.get('latency_p95_ms', 0):.1f}ms",
                    "✅" if r.get("passes", False) else "❌",
                ]
            )
        except Exception:  # noqa: BLE001
            continue

    return sorted(rows, key=lambda row: -float(row[2]))  # sort by recall_1 desc


def train_one_experiment(dataset: str, vectorizer_key: str, classifier_key: str) -> str:
    """Train one (vectorizer, classifier) experiment. Returns formatted result."""
    if not vectorizer_key or not classifier_key:
        return "Select both a vectorizer and a classifier first."

    dataset_dir = get_experiment_path(dataset, "classical").parent.parent
    if not (dataset_dir / "train.csv").exists():
        return f"train.csv not found for dataset '{dataset}'.\nRun phase_3_split.py first."

    try:
        from backend.training.classical.trainer import ClassicalMLTrainer

        log.info(f"Training: {vectorizer_key} + {classifier_key} on {dataset}")
        result = ClassicalMLTrainer().train_experiment(dataset, vectorizer_key, classifier_key)
        m = result.metrics
        flag = (
            "✅ PASSES (recall_1 >= 0.97)"
            if m.passes_production_threshold
            else "❌ FAILS (recall_1 < 0.97)"
        )
        return (
            f"Experiment: {result.experiment_id}\n"
            f"Dataset:    {dataset}\n\n"
            f"KEY METRICS:\n"
            f"  recall_1    = {m.recall_1:.4f}   ← {flag}\n"
            f"  precision_1 = {m.precision_1:.4f}\n"
            f"  recall_0    = {m.recall_0:.4f}\n"
            f"  MCC         = {m.mcc:.4f}\n"
            f"  ROC-AUC     = {m.roc_auc:.4f}\n"
            f"  PR-AUC      = {m.pr_auc:.4f}\n"
            f"  log_loss    = {m.log_loss:.4f}\n"
            f"  ECE         = {m.ece:.4f}\n\n"
            f"LATENCY (val set, n=500):\n"
            f"  p50 = {m.latency_p50_ms:.1f}ms  p95 = {m.latency_p95_ms:.1f}ms\n\n"
            f"Model saved: {result.model_path}\n"
            f"MLflow run:  {result.mlflow_run_id}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def train_all_experiments(dataset: str) -> str:
    """Run all ACTIVE_COMBOS experiments. Returns summary table."""
    dataset_dir = get_experiment_path(dataset, "classical").parent.parent
    if not (dataset_dir / "train.csv").exists():
        return f"train.csv not found for dataset '{dataset}'. Run phase_3_split.py first."

    try:
        from backend.training.classical.config import ACTIVE_COMBOS
        from backend.training.classical.trainer import ClassicalMLTrainer

        log.info(f"Training all {len(ACTIVE_COMBOS)} active combos on {dataset}")
        results = ClassicalMLTrainer().train_all_combos(dataset)

        lines = [f"Completed {len(results)} experiments:\n"]
        lines.append(f"  {'Experiment':<40} {'recall_1':>8} {'prec_1':>7} {'MCC':>7} {'pass':>5}")
        lines.append("  " + "─" * 68)
        for r in results:
            flag = "✅" if r.metrics.passes_production_threshold else "❌"
            lines.append(
                f"  {r.experiment_id:<40} "
                f"{r.metrics.recall_1:>8.4f} "
                f"{r.metrics.precision_1:>7.4f} "
                f"{r.metrics.mcc:>7.4f} "
                f"{flag:>5}"
            )

        passing = [r for r in results if r.metrics.passes_production_threshold]
        if passing:
            best = max(passing, key=lambda r: r.metrics.mcc)
            lines += [
                "",
                f"Best passing model: {best.experiment_id}",
                f"  recall_1={best.metrics.recall_1:.4f}  MCC={best.metrics.mcc:.4f}",
                f"  → {best.model_path}",
            ]
        else:
            lines.append("\n⚠️  No model passed recall_1 >= 0.97 threshold. Consider HPO.")

        return "\n".join(lines)

    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def build() -> None:
    """Build the Classical ML Training tab."""

    gr.Markdown(
        "### Classical ML Training\n"
        "Train and evaluate (vectorizer, classifier) combinations on `train.csv`.\n"
        "**Key metric: `recall_1 >= 0.97` is required for production deployment.**\n\n"
        "Results are saved to `experiments/{dataset}/classical/models/` and logged to MLflow."
    )

    dataset_box = gr.Textbox(
        value=settings_manager.get("DATASET_VERSION"),
        label="Dataset",
        max_lines=1,
    )

    gr.Markdown("---")

    # ── Existing results ──────────────────────────────────────────────────────
    gr.Markdown("#### Previous Results")
    gr.Markdown("Loaded from saved `result.json` files. Refresh after running new experiments.")

    results_table = gr.Dataframe(
        headers=[
            "Vectorizer",
            "Classifier",
            "recall_1",
            "prec_1",
            "MCC",
            "ROC-AUC",
            "p95 latency",
            "Passes",
        ],
        datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
        interactive=False,
        label="Experiment Results (sorted by recall_1)",
    )

    refresh_results_btn = gr.Button("🔄 Load Results", size="sm")
    refresh_results_btn.click(fn=load_existing_results, inputs=dataset_box, outputs=results_table)

    gr.Markdown("---")

    # ── Run one experiment ────────────────────────────────────────────────────
    gr.Markdown(
        "#### Run One Experiment\n"
        "Select a vectorizer and classifier, then click **Train**. "
        "Training takes 1–10 minutes depending on the combo."
    )

    with gr.Row():
        vec_dd = gr.Dropdown(
            choices=vectorizer_choices(),
            value="tfidf_combined",
            label="Vectorizer",
        )
        clf_dd = gr.Dropdown(
            choices=classifier_choices(),
            value="lightgbm",
            label="Classifier",
        )

    gr.Markdown(
        "Vectorizer guide: `tfidf_combined` is usually best for romanized code-mixed text.  \n"
        "Classifier guide: Start with `logistic_regression` (fast) then `lightgbm` (usually best)."
    )

    train_one_btn = gr.Button("▶ Train One Experiment", variant="secondary")
    train_one_out = gr.Textbox(label="Result", lines=18, interactive=False)
    train_one_btn.click(
        fn=train_one_experiment,
        inputs=[dataset_box, vec_dd, clf_dd],
        outputs=train_one_out,
    )

    gr.Markdown("---")

    # ── Run all experiments ───────────────────────────────────────────────────
    gr.Markdown(
        "#### Run All Combos\n"
        f"Runs all `{len(_get_active_combos())}` combinations defined in `ACTIVE_COMBOS` in `training/classical/config.py`. "
        "TF-IDF combos first (~2 min each), dense vectorizers last (~5–10 min each). "
        "Total: ~30–60 minutes. Results logged to MLflow."
    )

    train_all_btn = gr.Button("🚀 Train All Active Combos", variant="primary")
    train_all_out = gr.Textbox(label="Results", lines=25, interactive=False)
    train_all_btn.click(fn=train_all_experiments, inputs=dataset_box, outputs=train_all_out)

    gr.Markdown(
        "---\n**After training:** Run `mlflow ui` to compare all experiments interactively."
    )


def _get_active_combos() -> list:
    try:
        from backend.training.classical.config import ACTIVE_COMBOS

        return ACTIVE_COMBOS
    except Exception:  # noqa: BLE001
        return []
