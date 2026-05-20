"""
Generation tab panel.

Three distinct preview sections plus full pipeline controls:
  1. System Prompt Preview   — context sent to the model (open=True)
  2. User Prompt Preview     — per-turn user messages (open=True)
  3. Dry Run                 — generate and show messages (open=True)
  4. Full Pipeline           — generate + split controls (open=True)

All accordions default to open=True per project coding standards.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Generator

import gradio as gr

from backend.config.distribution import DISTRIBUTION
from backend.config.keys import IndustryKey, LanguageKey, ScenarioKey
from backend.generation.example_store import example_store
from backend.generation.generator import CONTINUE_USER_MSG, build_first_turn_user_msg
from backend.generation.prompt_factory import PROMPT_FACTORY
from backend.shared.settings_manager import settings_manager


def language_choices() -> list[str]:
    return list(LanguageKey)


def industry_choices() -> list[str]:
    return list(IndustryKey)


def scenario_choices() -> list[str]:
    return list(ScenarioKey)


def get_cell_info(language: str, industry: str, scenario: str) -> str:
    try:
        lang_b = DISTRIBUTION.get_language(LanguageKey(language))
        if not lang_b:
            return "Language not found"
        ind_b = lang_b.get_industry(IndustryKey(industry))
        if not ind_b:
            return "Industry not found"
        sc_b = ind_b.get_scenario(ScenarioKey(scenario))
        if not sc_b:
            return "Scenario not found"
        cached = (
            "✅ cell-specific (examples.json)"
            if example_store.is_cached(
                LanguageKey(language), IndustryKey(industry), ScenarioKey(scenario)
            )
            else "⚪ LengthRange fallback"
        )
        return (
            f"Target: {sc_b.computed_count:,} rows  │  "
            f"Language {lang_b.fraction:.1%} of {DISTRIBUTION.global_total:,}  │  "
            f"Industry {ind_b.fraction:.1%} of language  │  "
            f"Scenario {sc_b.fraction:.1%} of industry  │  "
            f"Examples: {cached}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def on_selection_change(language: str, industry: str, scenario: str) -> tuple[str, str, str, str]:
    return (
        get_cell_info(language, industry, scenario),
        preview_system_prompt(language, industry, scenario),
        preview_first_user_msg(language, industry, scenario),
        preview_continuation_msg(language, industry, scenario),
    )


def preview_system_prompt(language: str, industry: str, scenario: str) -> str:
    try:
        return PROMPT_FACTORY.build_preview(
            LanguageKey(language), IndustryKey(industry), ScenarioKey(scenario)
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def preview_first_user_msg(language: str, industry: str, scenario: str) -> str:
    try:
        lang_b = DISTRIBUTION.get_language(LanguageKey(language))
        ind_b = lang_b.get_industry(IndustryKey(industry)) if lang_b else None
        sc_b = ind_b.get_scenario(ScenarioKey(scenario)) if ind_b else None
        total = sc_b.computed_count if sc_b else 50

        batch_size = settings_manager.get("GENERATION_BATCH_SIZE")
        n = min(batch_size, total)
        return build_first_turn_user_msg(n=n, total_target=total, batch_size=batch_size)
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def preview_continuation_msg(language: str, industry: str, scenario: str) -> str:
    try:
        lang_b = DISTRIBUTION.get_language(LanguageKey(language))
        ind_b = lang_b.get_industry(IndustryKey(industry)) if lang_b else None
        sc_b = ind_b.get_scenario(ScenarioKey(scenario)) if ind_b else None
        total = sc_b.computed_count if sc_b else 200

        batch_size = settings_manager.get("GENERATION_BATCH_SIZE")
        generated = batch_size * 4
        remaining = total - generated
        n = min(batch_size, remaining)

        if remaining <= 0:
            return "(Cell is small — only 1 batch needed, no continuation turns)"

        return CONTINUE_USER_MSG.format(
            turn=5,
            generated=generated,
            target=total,
            n=n,
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def dry_run(language: str, industry: str, scenario: str, n: int) -> str:
    """Generate N messages and return only the generated output with word counts.

    Nothing is saved to disk.
    """
    try:
        from backend.generation.generator import GeneratorService
        from backend.generation.pymodels import GenerationCell

        cell = GenerationCell(
            language=LanguageKey(language),
            industry=IndustryKey(industry),
            scenario=ScenarioKey(scenario),
            target_count=int(n),
        )
        rows = GeneratorService().generate_cell(cell)

        if not rows:
            return "No rows generated. Check OPENAI_API_KEY in .env"

        label = rows[0]["label"]
        lines = [
            f"Generated {len(rows)} rows  │  label={label}  "
            f"({'gpt-4o-mini' if label == 0 else 'gpt-4o'})\n",
        ]
        for i, row in enumerate(rows, 1):
            lines.append(f"  {i:2}. [{row['word_count']}w]  {row['text']}")

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def run_split() -> str:
    dataset: str = settings_manager.get("DATASET_VERSION")
    try:
        from backend.generation.splitter import DataSplitter

        stats = DataSplitter().run(dataset)
        return (
            f"Split complete for '{dataset}':\n"
            f"  train: {stats['train']['rows']:,}\n"
            f"  val:   {stats['val']['rows']:,}\n"
            f"  test:  {stats['test']['rows']:,}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def stream_generate(resume: bool, workers: int) -> Generator[str, None, None]:
    dataset: str = settings_manager.get("DATASET_VERSION")
    updates: queue.SimpleQueue[str] = queue.SimpleQueue()

    def thread_target() -> None:
        from backend.generation.generator import GeneratorService

        try:
            GeneratorService().run(
                dataset_name=dataset,
                resume=resume,
                max_workers=int(workers),
                on_progress=updates.put,
            )
            updates.put("__DONE__")
        except Exception as exc:  # noqa: BLE001
            updates.put(f"__ERROR__ {exc}")

    threading.Thread(target=thread_target, daemon=True).start()

    lines: list[str] = [f"Generation started for dataset '{dataset}'..."]
    yield "\n".join(lines)

    while True:
        msg = updates.get()
        if msg == "__DONE__":
            lines.append("✅ Generation complete. Run Split next.")
            yield "\n".join(lines[-60:])
            break
        if msg.startswith("__ERROR__"):
            lines.append(f"❌ {msg[9:]}")
            yield "\n".join(lines[-60:])
            break
        lines.append(msg)
        yield "\n".join(lines[-60:])


def stream_full_pipeline(resume: bool, workers: int) -> Generator[str, None, None]:
    dataset: str = settings_manager.get("DATASET_VERSION")
    updates: queue.SimpleQueue[str] = queue.SimpleQueue()

    def thread_target() -> None:
        from backend.generation.generator import GeneratorService

        try:
            GeneratorService().run(
                dataset_name=dataset,
                resume=resume,
                max_workers=int(workers),
                on_progress=updates.put,
            )
            updates.put("__DONE__")
        except Exception as exc:  # noqa: BLE001
            updates.put(f"__ERROR__ {exc}")

    threading.Thread(target=thread_target, daemon=True).start()

    lines: list[str] = [f"Pipeline started for dataset '{dataset}'..."]
    yield "\n".join(lines)

    while True:
        msg = updates.get()
        if msg == "__DONE__":
            lines.append("✅ Generation done. Running split...")
            yield "\n".join(lines[-60:])
            break
        if msg.startswith("__ERROR__"):
            lines.append(f"❌ Generation failed: {msg[9:]}")
            yield "\n".join(lines[-60:])
            return
        lines.append(msg)
        yield "\n".join(lines[-60:])

    try:
        from backend.generation.splitter import DataSplitter

        stats = DataSplitter().run(dataset)
        lines += [
            "✅ Split complete:",
            f"  train: {stats['train']['rows']:,}",
            f"  val:   {stats['val']['rows']:,}",
            f"  test:  {stats['test']['rows']:,}",
            "",
            "Next: EDA → Training → Evaluate.",
        ]
        yield "\n".join(lines[-60:])
    except Exception as exc:  # noqa: BLE001
        lines.append(f"❌ Split failed: {exc}")
        yield "\n".join(lines[-60:])


def build() -> None:
    """Build the Generation tab. All accordions open=True per coding standards."""

    gr.Markdown("Select **(Language × Industry × Scenario)**. All sections update automatically.")

    with gr.Row():
        lang_dd = gr.Dropdown(
            choices=language_choices(), value=language_choices()[0], label="Language"
        )
        ind_dd = gr.Dropdown(
            choices=industry_choices(), value=industry_choices()[0], label="Industry"
        )
        sc_dd = gr.Dropdown(
            choices=scenario_choices(), value=scenario_choices()[0], label="Scenario"
        )

    cell_info = gr.Textbox(
        value=get_cell_info(language_choices()[0], industry_choices()[0], scenario_choices()[0]),
        label="Cell info — target rows and fraction breakdown",
        interactive=False,
        max_lines=1,
    )

    with gr.Accordion("🗂 System Prompt Preview", open=True):
        gr.Markdown(
            "Context the model receives at the start of every session: language mandate, "
            "industry info, scenario definition, length distribution, examples, and rules.\n\n"
            "The 'Generate N messages' user message is NOT shown here — it's added per-turn by the generator.\n\n"
            "Run `python cli.py examples-all --workers 10` to populate cell-specific examples."
        )
        system_prompt_box = gr.Textbox(
            value=preview_system_prompt(
                language_choices()[0], industry_choices()[0], scenario_choices()[0]
            ),
            label="System prompt (context only)",
            lines=35,
            interactive=False,
        )

    with gr.Accordion("💬 User Prompt Preview (real per-turn requests)", open=True):
        gr.Markdown(
            "What gets sent as the USER MESSAGE each turn. The system prompt stays fixed; "
            "only user messages change per batch.\n\n"
            "**Turn 1** sets session scope (total target + batch count).\n"
            "**Turn 2+** shows progress so the model understands how much variety is still needed."
        )
        with gr.Row():
            first_turn_box = gr.Textbox(
                value=preview_first_user_msg(
                    language_choices()[0], industry_choices()[0], scenario_choices()[0]
                ),
                label="Turn 1 — First batch user message",
                lines=5,
                interactive=False,
            )
            continuation_box = gr.Textbox(
                value=preview_continuation_msg(
                    language_choices()[0], industry_choices()[0], scenario_choices()[0]
                ),
                label="Turn 5 — Continuation batch user message (example)",
                lines=14,
                interactive=False,
            )

    with gr.Accordion("🧪 Dry Run — generate and preview messages", open=True):
        gr.Markdown(
            "Generate N messages for the selected cell. **Nothing is saved.**\n\n"
            "Shows each generated message with its word count and routing label."
        )
        dry_n = gr.Slider(minimum=3, maximum=50, value=10, step=1, label="N messages to generate")
        dry_btn = gr.Button("🧪 Generate (no save)", variant="secondary")
        dry_out = gr.Textbox(
            label="Generated messages  [word count per message]",
            lines=20,
            interactive=False,
        )
        dry_btn.click(fn=dry_run, inputs=[lang_dd, ind_dd, sc_dd, dry_n], outputs=dry_out)

    with gr.Accordion("🚀 Full Pipeline", open=True):
        dataset = settings_manager.get("DATASET_VERSION")
        cp_every = settings_manager.get("CHECKPOINT_EVERY")
        max_w = settings_manager.get("MAX_GENERATION_WORKERS")

        gr.Markdown(
            f"**Dataset:** `{dataset}`  |  **Max workers:** {max_w}\n\n"
            f"Checkpoint every **{cp_every:,}** rows. Live progress updates as cells complete.\n\n"
            "⚠️ Start server **without `--reload`** — file watcher interrupts generation.\n\n"
            "**Resume** = skip completed cells (use only after Ctrl+C).\n"
            "**Fill gaps** = CLI: `python phases/phase_2_generate.py --fill-gaps --workers 6`"
        )

        workers_slider = gr.Slider(
            minimum=1,
            maximum=10,
            value=7,
            step=1,
            label="Workers (recommend 7 — 10 hits rate limits on large cells)",
        )
        resume_cb = gr.Checkbox(
            label="Resume from checkpoint (use only after an interrupted run)",
            value=False,
        )

        with gr.Row():
            gen_btn = gr.Button("▶ Generate only", variant="secondary")
            split_btn = gr.Button("✂ Split only", variant="secondary")
            full_btn = gr.Button("🚀 Generate + Split", variant="primary", scale=2)

        pipeline_out = gr.Textbox(
            label="Progress (live — updates as each cell completes)",
            lines=20,
            interactive=False,
        )

        gen_btn.click(fn=stream_generate, inputs=[resume_cb, workers_slider], outputs=pipeline_out)
        split_btn.click(fn=run_split, outputs=pipeline_out)
        full_btn.click(
            fn=stream_full_pipeline, inputs=[resume_cb, workers_slider], outputs=pipeline_out
        )

    for dd in [lang_dd, ind_dd, sc_dd]:
        dd.change(
            fn=on_selection_change,
            inputs=[lang_dd, ind_dd, sc_dd],
            outputs=[cell_info, system_prompt_box, first_turn_box, continuation_box],
        )
