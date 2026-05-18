"""Gradio application. 5 tabs: Generation, EDA, Train, Evaluate, Router."""

from __future__ import annotations

import gradio as gr


def build_app() -> gr.Blocks:
    from frontend.panels import panel_eda, panel_generation, panel_training

    with gr.Blocks(title="Sprout Router Classifier") as demo:
        gr.Markdown("# 🚦 Sprout Router Classifier — model-router-classifier")
        gr.Markdown(
            "Binary LLM router: `gpt-4o-mini` (label=0) or `gpt-4o` (label=1)  |  "
            "Edit `config/settings.py` to change defaults."
        )
        with gr.Tabs():
            with gr.Tab("📦 Generation"):
                panel_generation.build()

            with gr.Tab("📊 EDA"):
                panel_eda.build()

            with gr.Tab("🏋️ Train"):
                panel_training.build()

            with gr.Tab("📈 Evaluate"):
                gr.Markdown(
                    "### Evaluation\n"
                    "Run `python phases/phase_7_evaluate.py` after training is complete.\n\n"
                    "This tab will show the full evaluation report once implemented."
                )

            with gr.Tab("🚦 Router"):
                gr.Markdown(
                    "### Router\n"
                    "Run `python phases/phase_8_router.py` after evaluation.\n\n"
                    "This tab will show threshold tuning and live router testing once implemented."
                )

    return demo


if __name__ == "__main__":
    build_app().launch(server_port=7860, share=False)
