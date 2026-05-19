"""
Training tab panel.

Two sections:
  Classical ML  — fast, runs locally, full streaming results
  Transformers  — GPU-intensive, runs locally or on Kaggle

All training logic lives in backend/training/ — this panel is a thin display layer.
Key metric: recall_1 >= 0.97 required for production deployment.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import Generator

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


def transformer_model_choices() -> list[str]:
    from backend.training.transformers.config import TRANSFORMER_REGISTRY

    return list(TRANSFORMER_REGISTRY.keys())


def load_classical_results(dataset: str) -> list[list]:
    """Load saved result.json files from all classical experiments."""
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

    return sorted(rows, key=lambda row: -float(row[2]))


def load_transformer_results(dataset: str) -> list[list]:
    """Load saved result.json files from all transformer experiments."""
    models_dir = get_experiment_path(dataset, "transformers") / "models"
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
                    r.get("model_name", ""),
                    f"{r.get('recall_1', 0):.4f}",
                    f"{r.get('precision_1', 0):.4f}",
                    f"{r.get('mcc', 0):.4f}",
                    f"{r.get('roc_auc', 0):.4f}",
                    f"{r.get('latency_p50_ms', 0):.1f}ms",
                    "✅" if r.get("passes", False) else "❌",
                ]
            )
        except Exception:  # noqa: BLE001
            continue

    return sorted(rows, key=lambda row: -float(row[1]))


def train_one_classical(dataset: str, vectorizer_key: str, classifier_key: str) -> str:
    if not vectorizer_key or not classifier_key:
        return "Select both a vectorizer and a classifier first."

    dataset_dir = get_experiment_path(dataset, "classical").parent.parent
    if not (dataset_dir / "train.csv").exists():
        return f"train.csv not found for dataset '{dataset}'. Run phase_3_split.py first."

    try:
        from backend.training.classical.trainer import ClassicalMLTrainer

        result = ClassicalMLTrainer().train_experiment(dataset, vectorizer_key, classifier_key)
        m = result.metrics
        flag = "✅ PASSES" if m.passes_production_threshold else "❌ FAILS"
        return (
            f"Experiment: {result.experiment_id}\n\n"
            f"  recall_1    = {m.recall_1:.4f}   ← {flag} (threshold: 0.97)\n"
            f"  precision_1 = {m.precision_1:.4f}\n"
            f"  recall_0    = {m.recall_0:.4f}\n"
            f"  MCC         = {m.mcc:.4f}\n"
            f"  ROC-AUC     = {m.roc_auc:.4f}\n"
            f"  log_loss    = {m.log_loss:.4f}\n"
            f"  latency p50 = {m.latency_p50_ms:.1f}ms  p95 = {m.latency_p95_ms:.1f}ms\n\n"
            f"Model saved: {result.model_path}\n"
            f"MLflow run:  {result.mlflow_run_id}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def train_all_classical(dataset: str) -> str:
    dataset_dir = get_experiment_path(dataset, "classical").parent.parent
    if not (dataset_dir / "train.csv").exists():
        return f"train.csv not found for dataset '{dataset}'. Run phase_3_split.py first."

    try:
        from backend.training.classical.config import ACTIVE_COMBOS
        from backend.training.classical.trainer import ClassicalMLTrainer

        results = ClassicalMLTrainer().train_all_combos(dataset)
        lines = [f"Completed {len(results)}/{len(ACTIVE_COMBOS)} experiments:\n"]
        lines.append(f"  {'Experiment':<40} {'recall_1':>8} {'MCC':>7} {'pass':>5}")
        lines.append("  " + "─" * 60)

        for r in results:
            flag = "✅" if r.metrics.passes_production_threshold else "❌"
            lines.append(
                f"  {r.experiment_id:<40} {r.metrics.recall_1:>8.4f} {r.metrics.mcc:>7.4f} {flag:>5}"
            )

        passing = [r for r in results if r.metrics.passes_production_threshold]
        if passing:
            best = max(passing, key=lambda r: r.metrics.mcc)
            lines += [
                "",
                f"Best: {best.experiment_id}  recall_1={best.metrics.recall_1:.4f}  MCC={best.metrics.mcc:.4f}",
            ]
        else:
            lines.append("\n⚠ No model passed recall_1 >= 0.97. Consider HPO.")

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def stream_train_transformer(dataset: str, model_key: str) -> Generator[str, None, None]:
    """Stream transformer training in a background thread.

    Transformer training takes 30–100 min. Without a GPU, this is extremely slow.
    For GPU training, use Kaggle (see CLI Reference tab).
    """
    if not model_key:
        yield "Select a model first."
        return

    updates: queue.SimpleQueue[str] = queue.SimpleQueue()

    def thread_target() -> None:
        try:
            from backend.training.transformers.trainer import TransformerTrainer

            result = TransformerTrainer().train_experiment(dataset, model_key)
            m = result.metrics
            flag = "✅ PASSES" if m.passes_production_threshold else "❌ FAILS"
            updates.put(
                f"__DONE__"
                f"Model:       {result.experiment_id}\n"
                f"recall_1:    {m.recall_1:.4f}   ← {flag}\n"
                f"precision_1: {m.precision_1:.4f}\n"
                f"MCC:         {m.mcc:.4f}\n"
                f"ROC-AUC:     {m.roc_auc:.4f}\n"
                f"Latency p50: {m.latency_p50_ms:.1f}ms\n"
                f"Checkpoint:  {result.model_path}"
            )
        except Exception as exc:  # noqa: BLE001
            updates.put(f"__ERROR__{exc}")

    threading.Thread(target=thread_target, daemon=True).start()

    yield (
        f"Training {model_key} on dataset '{dataset}'...\n"
        "HuggingFace training progress is printed to the terminal.\n"
        "Without a GPU, this may take several hours."
    )

    while True:
        try:
            msg = updates.get(timeout=5.0)
        except queue.Empty:
            yield (
                f"Training {model_key}... still running.\n"
                "Check terminal for HuggingFace epoch/step progress."
            )
            continue

        if msg.startswith("__DONE__"):
            yield f"✅ Training complete!\n\n{msg[8:]}"
        elif msg.startswith("__ERROR__"):
            yield f"❌ Training failed:\n{msg[9:]}"
        break


def build() -> None:
    """Build the Training tab with Classical ML and Transformer sections."""

    dataset_box = gr.Textbox(
        value=settings_manager.get("DATASET_VERSION"),
        label="Dataset version",
        max_lines=1,
    )

    # ── Classical ML ──────────────────────────────────────────────────────────
    with gr.Accordion("⚡ Classical ML", open=True):
        gr.Markdown(
            "Fast sklearn models: TF-IDF + classifier combinations. "
            "Runs locally in minutes. **Start here before running transformers.**\n\n"
            "**Production threshold: recall_1 ≥ 0.97**"
        )

        with gr.Accordion("📋 Previous Results", open=True):
            classical_table = gr.Dataframe(
                headers=[
                    "Vectorizer",
                    "Classifier",
                    "recall_1",
                    "prec_1",
                    "MCC",
                    "ROC-AUC",
                    "p95ms",
                    "Pass",
                ],
                datatype=["str"] * 8,
                interactive=False,
            )
            refresh_classical_btn = gr.Button("🔄 Load Results", size="sm")
            refresh_classical_btn.click(
                fn=load_classical_results,
                inputs=dataset_box,
                outputs=classical_table,
            )

        with gr.Accordion("▶ Run One Experiment", open=True):
            with gr.Row():
                vec_dd = gr.Dropdown(
                    choices=vectorizer_choices(), value="tfidf_combined", label="Vectorizer"
                )
                clf_dd = gr.Dropdown(
                    choices=classifier_choices(), value="lightgbm", label="Classifier"
                )
            gr.Markdown(
                "Start with `tfidf_combined + logistic_regression` (fast baseline), "
                "then `tfidf_combined + lightgbm` (usually best)."
            )
            train_one_btn = gr.Button("▶ Train", variant="secondary")
            train_one_out = gr.Textbox(label="Result", lines=14, interactive=False)
            train_one_btn.click(
                fn=train_one_classical,
                inputs=[dataset_box, vec_dd, clf_dd],
                outputs=train_one_out,
            )

        with gr.Accordion("🚀 Run All Active Combos", open=False):
            gr.Markdown(
                "Runs all combinations defined in `ACTIVE_COMBOS` in "
                "`training/classical/config.py`. ~30–60 min total. Results logged to MLflow."
            )
            train_all_btn = gr.Button("🚀 Train All", variant="primary")
            train_all_out = gr.Textbox(label="Summary", lines=20, interactive=False)
            train_all_btn.click(fn=train_all_classical, inputs=dataset_box, outputs=train_all_out)

    # ── Transformers ──────────────────────────────────────────────────────────
    with gr.Accordion("🤗 Transformers", open=False):
        gr.Markdown(
            "Fine-tuned multilingual transformers (XLM-R, MuRIL, etc.).\n\n"
            "⚠ **GPU strongly recommended.** On CPU, training takes several hours per model. "
            "For fast training, use the Kaggle notebook (`kaggle_train_transformers.py`). "
            "The 'Train Locally' button works but is slow without a GPU."
        )

        with gr.Accordion("📋 Previous Results", open=True):
            transformer_table = gr.Dataframe(
                headers=["Model", "recall_1", "prec_1", "MCC", "ROC-AUC", "p50ms", "Pass"],
                datatype=["str"] * 7,
                interactive=False,
            )
            refresh_transformer_btn = gr.Button("🔄 Load Results", size="sm")
            refresh_transformer_btn.click(
                fn=load_transformer_results,
                inputs=dataset_box,
                outputs=transformer_table,
            )

        with gr.Accordion("▶ Train Locally (GPU recommended)", open=True):
            transformer_dd = gr.Dropdown(
                choices=transformer_model_choices(),
                value="xlmr-base",
                label="Model (train xlmr-base first — empirically best for Sinhala)",
            )
            train_transformer_btn = gr.Button(
                "▶ Train Locally",
                variant="secondary",
            )
            transformer_out = gr.Textbox(
                label="Training output (progress in terminal)",
                lines=14,
                interactive=False,
            )
            train_transformer_btn.click(
                fn=stream_train_transformer,
                inputs=[dataset_box, transformer_dd],
                outputs=transformer_out,
            )

        with gr.Accordion("☁ Kaggle Instructions (recommended for GPU)", open=False):
            gr.Markdown("""
**Recommended order for Kaggle sessions:**

| Session | Models | Estimated time |
|---------|--------|----------------|
| 1 | xlmr-base + papluca | ~95 min |
| 2 | muril + mbert | ~105 min |
| 3 | xlmr-large (only if needed) | ~100 min |

**Setup:**
1. Upload `kaggle_train_transformers.py` to Kaggle as a new notebook
2. Add your training data dataset and code dataset as inputs
3. Set accelerator → **GPU T4 x2**
4. Change `MODEL_TO_TRAIN` in Cell 6 and run all cells
5. Download checkpoint from Outputs tab → copy to `experiments/v1/transformers/models/`
""")
            gr.Code(
                value=(
                    "# Upload data to Kaggle\n"
                    "kaggle datasets create -p data/datasets/v1  # run locally\n\n"
                    "# After downloading checkpoint from Kaggle:\n"
                    "cp -r ~/Downloads/xlmr-base/ experiments/v1/transformers/models/xlmr-base/\n\n"
                    "# Then evaluate\n"
                    "python phases/phase_7_evaluate.py --dataset v1"
                ),
                language="shell",
                interactive=False,
            )
