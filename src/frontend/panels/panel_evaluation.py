"""
Evaluate tab panel.

Shows model comparison table, cost simulation, and best model summary.
All business logic lives in backend/evaluation/ — this panel is a thin display layer.
"""

from __future__ import annotations

import gradio as gr

from backend.evaluation.comparator import ModelComparator
from backend.evaluation.cost_simulator import CostSimulator
from backend.shared.settings_manager import settings_manager

COMPARISON_HEADERS = [
    "Experiment",
    "Approach",
    "recall_1",
    "precision_1",
    "recall_0",
    "MCC",
    "ROC-AUC",
    "F1-macro",
    "p50 ms",
    "p95 ms",
    "Pass",
]

COST_HEADERS = [
    "Strategy",
    "% to mini",
    "% to gpt-4o",
    "daily cost $",
    "daily savings $",
    "monthly savings $",
    "recall_1",
]


def load_comparison(dataset: str) -> tuple[list[list], str]:
    try:
        comparator = ModelComparator()
        rows = comparator.compare(dataset)
        if not rows:
            return [], "No results found. Run phase_5 and/or phase_6 first."
        table = [format_comparison_row(r) for r in rows]
        best = comparator.best_model(dataset)
        summary = (
            (
                f"✅ Best: {best.experiment_id}  |  "
                f"recall_1={best.recall_1:.4f}  |  MCC={best.mcc:.4f}  |  "
                f"{'production-safe' if best.passes_production_threshold else '⚠ below threshold'}"
            )
            if best
            else "No best model found."
        )
        return table, summary
    except Exception as exc:  # noqa: BLE001
        return [], f"Error: {exc}"


def load_cost_sim(dataset: str, daily_messages: int) -> list[list]:
    try:
        rows = ModelComparator().compare(dataset)
        results = CostSimulator().simulate_all(dataset, rows, int(daily_messages))
        return [format_cost_row(r) for r in results]
    except Exception as exc:  # noqa: BLE001
        return [[f"Error: {exc}"]]


def format_comparison_row(r: object) -> list:
    return [
        getattr(r, "experiment_id", ""),
        getattr(r, "approach", ""),
        f"{getattr(r, 'recall_1', 0):.4f}",
        f"{getattr(r, 'precision_1', 0):.4f}",
        f"{getattr(r, 'recall_0', 0):.4f}",
        f"{getattr(r, 'mcc', 0):.4f}",
        f"{getattr(r, 'roc_auc', 0):.4f}",
        f"{getattr(r, 'f1_macro', 0):.4f}",
        f"{getattr(r, 'latency_p50_ms', 0):.1f}",
        f"{getattr(r, 'latency_p95_ms', 0):.1f}",
        "✅" if getattr(r, "passes_production_threshold", False) else "❌",
    ]


def format_cost_row(r: object) -> list:
    return [
        getattr(r, "strategy_name", ""),
        f"{getattr(r, 'pct_routed_to_mini', 0):.1%}",
        f"{getattr(r, 'pct_routed_to_4o', 0):.1%}",
        f"${getattr(r, 'daily_cost_usd', 0):.4f}",
        f"${getattr(r, 'daily_savings_usd', 0):.4f}",
        f"${getattr(r, 'monthly_savings_usd', 0):.2f}",
        f"{getattr(r, 'recall_1', 0):.4f}",
    ]


def build() -> None:
    dataset_default = settings_manager.get("DATASET_VERSION")

    gr.Markdown(
        "**Evaluate** trained models. Run **Phase 5** (classical) and/or **Phase 6** "
        "(transformers on Kaggle) before loading results here.\n\n"
        "Production threshold: **recall_1 ≥ 0.97** — "
        "at most 3% of complex/sensitive messages incorrectly routed to gpt-4o-mini."
    )

    with gr.Row():
        dataset_box = gr.Textbox(
            value=dataset_default, label="Dataset version", max_lines=1, scale=1
        )
        daily_msgs = gr.Slider(
            minimum=1_000,
            maximum=100_000,
            value=10_000,
            step=1_000,
            label="Daily message volume (for cost sim)",
            scale=3,
        )

    with gr.Accordion("📊 Model Comparison", open=True):
        gr.Markdown("Ranks all trained experiments by recall_1. **Pass ✅** = recall_1 ≥ 0.97.")
        compare_btn = gr.Button("▶ Load Comparison", variant="primary")
        best_summary = gr.Textbox(label="Best model", interactive=False, max_lines=2)
        compare_table = gr.Dataframe(
            headers=COMPARISON_HEADERS,
            datatype=["str"] * len(COMPARISON_HEADERS),
            interactive=False,
        )
        compare_btn.click(
            fn=load_comparison,
            inputs=[dataset_box],
            outputs=[compare_table, best_summary],
        )

    with gr.Accordion("💰 Cost Simulation", open=True):
        gr.Markdown("Daily API cost vs routing strategy. Pricing from `config/settings.py`.")
        cost_btn = gr.Button("▶ Run Cost Simulation", variant="secondary")
        cost_table = gr.Dataframe(
            headers=COST_HEADERS,
            datatype=["str"] * len(COST_HEADERS),
            interactive=False,
        )
        cost_btn.click(fn=load_cost_sim, inputs=[dataset_box, daily_msgs], outputs=cost_table)

    with gr.Accordion("🖥 CLI Reference", open=True):
        gr.Code(
            value=(
                "# Full evaluation\n"
                "python phases/phase_7_evaluate.py --dataset v1\n\n"
                "# With specific daily volume\n"
                "python phases/phase_7_evaluate.py --dataset v1 --daily-messages 25000\n\n"
                "# False-negative breakdown on val.csv\n"
                "python phases/phase_7_evaluate.py --dataset v1 --error-analysis\n\n"
                "# Final test-set eval (run ONCE at the end)\n"
                "python phases/phase_7_evaluate.py --dataset v1 --ablate"
            ),
            language="shell",
            interactive=False,
        )
