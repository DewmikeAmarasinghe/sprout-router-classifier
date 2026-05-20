"""
Router tab panel.

Live routing prediction for any message using a trained model.
Shows: label, routed model, confidence, routing reason, and threshold curve.

All prediction logic lives in backend/router/ — this panel is a thin display layer.
"""

from __future__ import annotations

import gradio as gr

from backend.shared.settings_manager import settings_manager


def get_available_models(dataset: str) -> list[str]:
    """Return trained model keys available for routing."""
    try:
        from backend.evaluation.comparator import ModelComparator

        rows = ModelComparator().compare(dataset)
        return [r.experiment_id for r in rows] if rows else []
    except Exception:
        return []


def predict_message(
    dataset: str,
    model_key: str,
    message: str,
    threshold: float,
) -> tuple[str, str, str, str]:
    """Route one message. Returns (label_str, routed_to, confidence, reason)."""
    if not message.strip():
        return "—", "—", "—", "Enter a message above"
    if not model_key:
        return "—", "—", "—", "No model selected. Train a model first."

    try:
        from backend.router.predictor import RouterPredictor
        from backend.router.pymodels import ThresholdConfig
        from backend.shared.path_resolver import get_experiment_path

        approach = infer_approach(dataset, model_key)

        if approach == "classical":
            path = get_experiment_path(dataset, "classical") / "models" / f"{model_key}.pkl"
            predictor = RouterPredictor.from_pkl(path)
        else:
            path = get_experiment_path(dataset, "transformers") / "models" / model_key
            predictor = RouterPredictor.from_hf_checkpoint(path)

        result = RouterPredictor(
            model=predictor._model,  # type: ignore[attr-defined]
            threshold_config=ThresholdConfig(threshold=threshold),
        ).predict(message)

        label_str = f"label={result.label}  ({'gpt-4o' if result.label == 1 else 'gpt-4o-mini'})"
        return label_str, result.routed_to, f"{result.confidence:.4f}", result.routing_reason

    except FileNotFoundError:
        return "—", "—", "—", f"Model '{model_key}' not found. Re-run training."
    except Exception as exc:
        return "—", "—", "—", f"Error: {exc}"


def infer_approach(dataset: str, experiment_id: str) -> str:
    try:
        from backend.evaluation.comparator import ModelComparator

        rows = ModelComparator().compare(dataset)
        for row in rows:
            if row.experiment_id == experiment_id:
                return row.approach
    except Exception:
        pass
    return "classical"


def load_threshold_curve(dataset: str) -> str:
    """Load saved threshold curve and format as text table."""
    import json

    from backend.shared.path_resolver import get_experiment_path

    path = get_experiment_path(dataset, "classical").parent / "router" / "threshold_curve.json"
    if not path.exists():
        return "No threshold curve found. Run: python phases/phase_8_router.py --dataset v1"

    try:
        data = json.loads(path.read_text())
        lines = [
            f"Optimal threshold: {data['optimal_threshold']}  "
            f"recall_1={data['optimal_recall_1']:.4f}  "
            f"precision_1={data['optimal_precision_1']:.4f}\n",
            f"{'Threshold':>10}  {'recall_1':>9}  {'precision_1':>12}  {'accuracy':>9}  {'TP':>5}  {'FN':>5}",
            "─" * 60,
        ]
        for row in data.get("curve", []):
            marker = " ◄ OPTIMAL" if row["threshold"] == data["optimal_threshold"] else ""
            lines.append(
                f"{row['threshold']:>10.2f}  {row['recall_1']:>9.4f}  "
                f"{row['precision_1']:>12.4f}  {row['accuracy']:>9.4f}  "
                f"{row['tp']:>5}  {row['fn']:>5}{marker}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Error loading curve: {exc}"


def refresh_models(dataset: str) -> gr.Dropdown:
    models = get_available_models(dataset)
    value = models[0] if models else None
    return gr.Dropdown(choices=models, value=value)


def build() -> None:
    """Build the Router tab. Thin Gradio wrapper over backend/router/."""

    dataset_default = settings_manager.get("DATASET_VERSION")
    threshold_default = settings_manager.get("CONFIDENCE_THRESHOLD")

    gr.Markdown(
        "**Live routing prediction.** Load a trained model and test any message.\n\n"
        "Requires trained models from **Phase 5** (classical) or **Phase 6** (transformers)."
    )

    # ── Config row ───────────────────────────────────────────────────────────
    with gr.Row():
        dataset_box = gr.Textbox(
            value=dataset_default,
            label="Dataset version",
            max_lines=1,
            scale=1,
        )
        model_dd = gr.Dropdown(
            choices=get_available_models(dataset_default),
            label="Model",
            scale=2,
        )
        threshold_slider = gr.Slider(
            minimum=0.30,
            maximum=0.90,
            value=threshold_default,
            step=0.05,
            label=f"Confidence threshold (production default: {threshold_default})",
            scale=2,
        )
        refresh_btn = gr.Button("↻ Refresh models", size="sm", scale=1)

    # ── Prediction ───────────────────────────────────────────────────────────
    with gr.Accordion("🚦 Live Prediction", open=True):
        gr.Markdown(
            "Type any customer message. The router applies the three-layer decision:\n"
            "script_detector → ML model confidence → threshold check."
        )
        message_box = gr.Textbox(
            label="Customer message",
            lines=3,
            placeholder='e.g. "nearest branch to me?" or "still shows error"',
        )
        predict_btn = gr.Button("🚦 Route this message", variant="primary")

        with gr.Row():
            label_out = gr.Textbox(label="Decision", interactive=False, scale=2)
            routed_to_out = gr.Textbox(label="Routes to", interactive=False, scale=2)
            confidence_out = gr.Textbox(label="Confidence", interactive=False, scale=1)
            reason_out = gr.Textbox(label="Routing reason", interactive=False, scale=2)

        predict_btn.click(
            fn=predict_message,
            inputs=[dataset_box, model_dd, message_box, threshold_slider],
            outputs=[label_out, routed_to_out, confidence_out, reason_out],
        )

    # ── Threshold curve ───────────────────────────────────────────────────────
    with gr.Accordion("📈 Threshold Curve", open=True):
        gr.Markdown(
            "Run `python phases/phase_8_router.py --dataset v1` to generate the curve.\n"
            "Shows how recall_1 and precision_1 change across thresholds."
        )
        curve_btn = gr.Button("Load threshold curve", variant="secondary")
        curve_out = gr.Textbox(label="Threshold sweep results", lines=20, interactive=False)
        curve_btn.click(fn=load_threshold_curve, inputs=[dataset_box], outputs=curve_out)

    # ── Batch test ────────────────────────────────────────────────────────────
    with gr.Accordion("📋 Batch Test Messages", open=True):
        gr.Markdown("Paste multiple messages (one per line) to route in batch.")
        batch_input = gr.Textbox(label="Messages (one per line)", lines=8)
        batch_btn = gr.Button("Route all", variant="secondary")
        batch_output = gr.Textbox(label="Results", lines=15, interactive=False)

        def route_batch(dataset: str, model_key: str, text: str, threshold: float) -> str:
            lines = [m.strip() for m in text.strip().splitlines() if m.strip()]
            results = []
            for msg in lines:
                _, routed, conf, reason = predict_message(dataset, model_key, msg, threshold)
                results.append(f"[{routed}  {conf}  {reason}]  {msg[:60]}")
            return "\n".join(results) if results else "No messages entered."

        batch_btn.click(
            fn=route_batch,
            inputs=[dataset_box, model_dd, batch_input, threshold_slider],
            outputs=batch_output,
        )

    refresh_btn.click(fn=refresh_models, inputs=[dataset_box], outputs=model_dd)
