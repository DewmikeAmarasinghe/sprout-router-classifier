"""
Training tab panel.

Two sections:
  Classical ML  — fast, runs locally, full streaming results
  Transformers  — GPU-intensive, runs locally or on Kaggle

All training logic lives in backend/training/ — this panel is a thin display layer.
Key metric: recall_1 >= PRODUCTION_RECALL_THRESHOLD required for production deployment.
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
from backend.training.pymodels import (
    CLASSICAL_TABLE_HEADERS,
    TRANSFORMER_TABLE_HEADERS,
    MetricsResult,
)

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
    """Load all metrics from saved result.json files for classical experiments."""
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
            row: list = [r.get("vectorizer_key", ""), r.get("classifier_key", "")]
            for field in MetricsResult.field_names():
                row.append(MetricsResult.format_cell(field, float(r.get(field, 0))))
            row.append("✅" if r.get("passes", False) else "❌")
            rows.append(row)
        except Exception:  # noqa: BLE001
            continue

    recall_col = 2 + MetricsResult.field_names().index("recall_1")
    return sorted(rows, key=lambda row: -float(row[recall_col].rstrip("ms")))


def load_transformer_results(dataset: str) -> list[list]:
    """Load all metrics from saved result.json files for transformer experiments."""
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
            row: list = [r.get("model_name", "") or r.get("experiment_id", "")]
            for field in MetricsResult.field_names():
                row.append(MetricsResult.format_cell(field, float(r.get(field, 0))))
            row.append("✅" if r.get("passes", False) else "❌")
            rows.append(row)
        except Exception:  # noqa: BLE001
            continue

    recall_col = 1 + MetricsResult.field_names().index("recall_1")
    return sorted(rows, key=lambda row: -float(row[recall_col].rstrip("ms")))


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
            f"  recall_1    = {m.recall_1:.4f}   ← {flag} (threshold: {float(settings_manager.get('PRODUCTION_RECALL_THRESHOLD'))})\n"
            f"  recall_0    = {m.recall_0:.4f}\n"
            f"  precision_1 = {m.precision_1:.4f}\n"
            f"  precision_0 = {m.precision_0:.4f}\n"
            f"  MCC         = {m.mcc:.4f}\n"
            f"  ROC-AUC     = {m.roc_auc:.4f}\n"
            f"  PR-AUC      = {m.pr_auc:.4f}\n"
            f"  f1_macro    = {m.f1_macro:.4f}\n"
            f"  log_loss    = {m.log_loss:.4f}\n"
            f"  ECE         = {m.ece:.4f}\n"
            f"  latency p50 = {m.latency_p50_ms:.1f}ms  p95 = {m.latency_p95_ms:.1f}ms\n\n"
            f"Model saved: {result.model_path}\n"
            f"MLflow run:  {result.mlflow_run_id}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def run_hpo_classical(dataset: str, vectorizer_key: str, classifier_key: str) -> str:
    if not vectorizer_key or not classifier_key:
        return "Select both a vectorizer and a classifier first."

    try:
        from backend.training.classical.hpo import ClassicalHPORunner
        from backend.training.classical.trainer import ClassicalMLTrainer

        hpo_result = ClassicalHPORunner().run(
            dataset_name=dataset,
            vectorizer_key=vectorizer_key,
            classifier_key=classifier_key,
            n_trials=10,
            optimize_metric="mcc",
        )
        tuned = ClassicalMLTrainer().train_experiment(
            dataset,
            vectorizer_key,
            classifier_key,
            classifier_params=hpo_result.best_params,
        )
        m = tuned.metrics
        flag = "✅ PASSES" if m.passes_production_threshold else "❌ FAILS"
        return (
            f"HPO complete — {tuned.experiment_id}\n\n"
            f"  Best params: {hpo_result.best_params}\n\n"
            f"  recall_1    = {m.recall_1:.4f}   ← {flag}\n"
            f"  MCC         = {m.mcc:.4f}\n"
            f"  ROC-AUC     = {m.roc_auc:.4f}\n"
            f"  latency p95 = {m.latency_p95_ms:.1f}ms\n\n"
            f"Model saved: {tuned.model_path}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def train_all_classical(dataset: str) -> str:
    dataset_dir = get_experiment_path(dataset, "classical").parent.parent
    if not (dataset_dir / "train.csv").exists():
        return f"train.csv not found for dataset '{dataset}'. Run phase_3_split.py first."

    try:
        from backend.training.classical.config import CLASSIFIER_REGISTRY, VECTORIZER_REGISTRY
        from backend.training.classical.trainer import ClassicalMLTrainer

        all_combos = [(vec, clf) for vec in VECTORIZER_REGISTRY for clf in CLASSIFIER_REGISTRY]
        results = ClassicalMLTrainer().train_combos(dataset, combos=all_combos)
        lines = [f"Completed {len(results)}/{len(all_combos)} experiments:\n"]
        lines.append(f"  {'Experiment':<40} {'recall_1':>8} {'MCC':>7} {'pass':>5}")
        lines.append("  " + "─" * 60)

        for r in results:
            flag = "✅" if r.metrics.passes_production_threshold else "❌"
            lines.append(
                f"  {r.experiment_id:<40} {r.metrics.recall_1:>8.4f} {r.metrics.mcc:>7.4f} {flag:>5}"
            )

        passing = [r for r in results if r.metrics.passes_production_threshold]
        threshold = float(settings_manager.get("PRODUCTION_RECALL_THRESHOLD"))
        if passing:
            best = max(passing, key=lambda r: r.metrics.mcc)
            lines += [
                "",
                f"Best: {best.experiment_id}  recall_1={best.metrics.recall_1:.4f}  MCC={best.metrics.mcc:.4f}",
                f"({len(passing)} of {len(results)} passed recall_1 ≥ {threshold})",
            ]
        else:
            lines.append(
                f"\n⚠ No model passed recall_1 >= {threshold}. Run HPO or train transformers."
            )

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def stream_train_transformer(dataset: str, model_key: str) -> Generator[str, None, None]:
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
                f"recall_0:    {m.recall_0:.4f}\n"
                f"precision_1: {m.precision_1:.4f}\n"
                f"precision_0: {m.precision_0:.4f}\n"
                f"MCC:         {m.mcc:.4f}\n"
                f"ROC-AUC:     {m.roc_auc:.4f}\n"
                f"PR-AUC:      {m.pr_auc:.4f}\n"
                f"f1_macro:    {m.f1_macro:.4f}\n"
                f"log_loss:    {m.log_loss:.4f}\n"
                f"ECE:         {m.ece:.4f}\n"
                f"Latency p50: {m.latency_p50_ms:.1f}ms  p95: {m.latency_p95_ms:.1f}ms\n"
                f"Checkpoint:  {result.model_path}"
            )
        except Exception as exc:  # noqa: BLE001
            updates.put(f"__ERROR__{exc}")

    threading.Thread(target=thread_target, daemon=True).start()

    yield (
        f"Training {model_key} on dataset '{dataset}'...\n"
        "HuggingFace progress is printed to the terminal.\n"
        "Without a GPU, this may take several hours."
    )

    while True:
        try:
            msg = updates.get(timeout=5.0)
        except queue.Empty:
            yield f"Training {model_key}... still running. Check terminal for progress."
            continue

        if msg.startswith("__DONE__"):
            yield f"✅ Training complete!\n\n{msg[8:]}"
        elif msg.startswith("__ERROR__"):
            yield f"❌ Training failed:\n{msg[9:]}"
        break


def stream_hpo_transformer(dataset: str, model_key: str) -> Generator[str, None, None]:
    if not model_key:
        yield "Select a model first."
        return

    updates: queue.SimpleQueue[str] = queue.SimpleQueue()

    def thread_target() -> None:
        try:
            from backend.training.transformers.hpo import TransformerHPORunner
            from backend.training.transformers.trainer import TransformerTrainer

            hpo_result = TransformerHPORunner().run(
                dataset_name=dataset,
                model_key=model_key,
                n_trials=5,
            )
            result = TransformerTrainer().train_experiment(
                dataset, model_key, param_overrides=hpo_result.best_params
            )
            m = result.metrics
            flag = "✅ PASSES" if m.passes_production_threshold else "❌ FAILS"
            updates.put(
                f"__DONE__"
                f"HPO complete — {result.experiment_id}\n"
                f"Best params: {hpo_result.best_params}\n\n"
                f"recall_1:    {m.recall_1:.4f}   ← {flag}\n"
                f"MCC:         {m.mcc:.4f}\n"
                f"ROC-AUC:     {m.roc_auc:.4f}\n"
                f"Latency p95: {m.latency_p95_ms:.1f}ms\n"
                f"Checkpoint:  {result.model_path}"
            )
        except Exception as exc:  # noqa: BLE001
            updates.put(f"__ERROR__{exc}")

    threading.Thread(target=thread_target, daemon=True).start()

    yield (
        f"Running HPO for {model_key} — 5 trials × 3 epochs each.\n"
        "⚠ This can take several hours on GPU. Check terminal for Optuna progress."
    )

    while True:
        try:
            msg = updates.get(timeout=5.0)
        except queue.Empty:
            yield f"HPO {model_key}... still running. Check terminal for progress."
            continue

        if msg.startswith("__DONE__"):
            yield f"✅ HPO complete!\n\n{msg[8:]}"
        elif msg.startswith("__ERROR__"):
            yield f"❌ HPO failed:\n{msg[9:]}"
        break


def build() -> None:
    dataset_box = gr.Textbox(
        value=settings_manager.get("DATASET_VERSION"),
        label="Dataset version",
        max_lines=1,
    )

    with gr.Accordion("⚡ Classical ML", open=True):
        gr.Markdown(
            "Fast sklearn models: TF-IDF + classifier combinations. Runs locally in minutes.\n\n"
            f"**Production threshold: recall_1 ≥ {float(settings_manager.get('PRODUCTION_RECALL_THRESHOLD'))}**\n\n"
            "Classical training: sklearn fits in a single pass — no epochs. "
            "HPO (Optuna, 10 trials) runs automatically after `--all` / `--all-active` and picks the best hyperparameters."
        )

        with gr.Accordion("📋 Previous Results (all metrics)", open=True):
            classical_table = gr.Dataframe(
                headers=CLASSICAL_TABLE_HEADERS,
                datatype=["str"] * len(CLASSICAL_TABLE_HEADERS),
                interactive=False,
            )
            refresh_classical_btn = gr.Button("🔄 Load Results", size="sm")
            refresh_classical_btn.click(
                fn=load_classical_results, inputs=dataset_box, outputs=classical_table
            )

        with gr.Accordion("▶ Run One Experiment", open=True):
            with gr.Row():
                vec_dd = gr.Dropdown(
                    choices=vectorizer_choices(), value="tfidf_combined", label="Vectorizer"
                )
                clf_dd = gr.Dropdown(choices=classifier_choices(), value="svm", label="Classifier")
            gr.Markdown(
                "Any vectorizer × classifier combination is supported. "
                "`ACTIVE_COMBOS` is the curated subset used by `--all-active`."
            )
            with gr.Row():
                train_one_btn = gr.Button("▶ Train", variant="secondary")
                hpo_one_btn = gr.Button("🔬 Run HPO", variant="secondary")
            train_one_out = gr.Textbox(label="Result", lines=16, interactive=False)
            train_one_btn.click(
                fn=train_one_classical, inputs=[dataset_box, vec_dd, clf_dd], outputs=train_one_out
            )
            hpo_one_btn.click(
                fn=run_hpo_classical, inputs=[dataset_box, vec_dd, clf_dd], outputs=train_one_out
            )

        with gr.Accordion("🚀 Run All Combos", open=True):
            gr.Markdown(
                "Runs all 25 combinations (5 vectorizers × 5 classifiers) sequentially. ~60–90 min total.\n\n"
                "For the curated subset only: `python phases/phase_5_train_classical.py --all-active`"
            )
            train_all_btn = gr.Button("🚀 Train All", variant="primary")
            train_all_out = gr.Textbox(label="Summary", lines=20, interactive=False)
            train_all_btn.click(fn=train_all_classical, inputs=dataset_box, outputs=train_all_out)

    with gr.Accordion("🤗 Transformers", open=True):
        gr.Markdown(
            "Fine-tuned multilingual transformers (XLM-R, MuRIL, etc.).\n\n"
            "⚠ **GPU strongly recommended.** Training runs for **3 epochs** per model per `TRAIN_CONFIG`.\n\n"
            "For GPU training, use the Kaggle notebook (`kaggle_train_transformers.py`).\n\n"
            "**`--all` flag**: `python phases/phase_6_train_transformers.py --all --dataset v1` trains all 5 models "
            "sequentially (~5 hours on T4). On Kaggle, prefer training 1–2 per session to stay within the 12-hour limit."
        )

        with gr.Accordion("📋 Previous Results (all metrics)", open=True):
            transformer_table = gr.Dataframe(
                headers=TRANSFORMER_TABLE_HEADERS,
                datatype=["str"] * len(TRANSFORMER_TABLE_HEADERS),
                interactive=False,
            )
            refresh_transformer_btn = gr.Button("🔄 Load Results", size="sm")
            refresh_transformer_btn.click(
                fn=load_transformer_results, inputs=dataset_box, outputs=transformer_table
            )

        with gr.Accordion("▶ Train (GPU recommended)", open=True):
            transformer_dd = gr.Dropdown(
                choices=transformer_model_choices(),
                value="xlmr-base",
                label="Model (train xlmr-base first — 3 epochs, ~50 min on T4)",
            )
            with gr.Row():
                train_transformer_btn = gr.Button("▶ Train", variant="secondary")
                hpo_transformer_btn = gr.Button("🔬 Run HPO", variant="secondary")
            transformer_out = gr.Textbox(
                label="Training output (progress in terminal)", lines=16, interactive=False
            )
            train_transformer_btn.click(
                fn=stream_train_transformer,
                inputs=[dataset_box, transformer_dd],
                outputs=transformer_out,
            )
            hpo_transformer_btn.click(
                fn=stream_hpo_transformer,
                inputs=[dataset_box, transformer_dd],
                outputs=transformer_out,
            )

        with gr.Accordion("☁ Kaggle Instructions", open=True):
            gr.Markdown("""
**Recommended Kaggle sessions:**

| Session | Models | Time |
|---------|--------|------|
| 1 | Classical (`--all-active`) + xlmr-base | ~70 min |
| 2 | papluca + muril | ~100 min |
| 3 | mbert (+ xlmr-large if needed) | ~150 min |

**Setup:** Upload `kaggle_train_transformers.py`, set GPU T4 x2, run all cells.

**HPO** runs automatically after `--all-active` (classical) and after each transformer if recall_1 < threshold.
Use `--no-hpo` to skip. Use `--hpo` to force HPO even when recall already passes.
HPO uses the **val set** only — it never touches test.csv.
""")
            gr.Code(
                value=(
                    "# After downloading checkpoint from Kaggle:\n"
                    "cp -r ~/Downloads/experiments/v1/ ./experiments/v1/\n\n"
                    "# Then evaluate locally\n"
                    "python phases/phase_7_evaluate.py --dataset v1\n"
                    "python phases/phase_8_router.py --dataset v1"
                ),
                language="shell",
                interactive=False,
            )
