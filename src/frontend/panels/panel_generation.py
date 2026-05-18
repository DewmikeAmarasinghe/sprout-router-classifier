"""
Generation tab panel.

Live progress is streamed into the UI as cells complete.
Generator functions (yield) feed [1/352] Done: ... messages
directly to the output textbox without blocking.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Generator

import gradio as gr

from backend.config.distribution import DISTRIBUTION
from backend.config.keys import IndustryKey, LanguageKey, ScenarioKey
from backend.generation.example_store import example_store
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
            "✅ cell-specific"
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


def on_selection_change(language: str, industry: str, scenario: str) -> tuple[str, str]:
    return (
        get_cell_info(language, industry, scenario),
        preview_prompt(language, industry, scenario),
    )


def preview_prompt(language: str, industry: str, scenario: str) -> str:
    try:
        return PROMPT_FACTORY.build_preview(
            LanguageKey(language),
            IndustryKey(industry),
            ScenarioKey(scenario),
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def show_examples(language: str, industry: str, scenario: str) -> str:
    try:
        examples = example_store.get(
            LanguageKey(language),
            IndustryKey(industry),
            ScenarioKey(scenario),
        )
        cached = example_store.is_cached(
            LanguageKey(language),
            IndustryKey(industry),
            ScenarioKey(scenario),
        )
        header = (
            "Source: cell-specific (examples.json)\n"
            if cached
            else "Source: LengthRange fallback — run `python cli.py examples-all`\n"
        )
        return header + "\n".join(f"{i + 1}. {ex}" for i, ex in enumerate(examples))
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def dry_run(language: str, industry: str, scenario: str, n: int) -> str:
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
        lines = [f"Generated {len(rows)} rows  label={rows[0]['label']}\n"]
        lines += [f"{i:2}. [{row['word_count']}w] {row['text']}" for i, row in enumerate(rows, 1)]
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


# ── Streaming generation ──────────────────────────────────────────────────────


def _run_in_thread_with_queue(
    dataset: str,
    resume: bool,
    workers: int,
    updates: queue.SimpleQueue[str],
) -> None:
    """Thread target: runs generation and feeds progress into the queue."""
    from backend.generation.generator import GeneratorService

    def progress(msg: str) -> None:
        updates.put(msg)

    try:
        GeneratorService().run(
            dataset_name=dataset,
            resume=resume,
            max_workers=workers,
            on_progress=progress,
        )
        updates.put("__DONE__")
    except Exception as exc:  # noqa: BLE001
        updates.put(f"__ERROR__ {exc}")


def stream_generate(resume: bool, workers: int) -> Generator[str, None, None]:
    """Generator: streams [1/352] Done: ... progress into the output textbox."""
    dataset = settings_manager.get("DATASET_VERSION")
    updates: queue.SimpleQueue[str] = queue.SimpleQueue()

    thread = threading.Thread(
        target=_run_in_thread_with_queue,
        args=(dataset, resume, int(workers), updates),
        daemon=True,
    )
    thread.start()

    lines: list[str] = [f"Generation started for dataset '{dataset}'..."]
    yield "\n".join(lines)

    while True:
        msg = updates.get()  # blocks until next message arrives

        if msg == "__DONE__":
            lines.append("✅ Generation complete. Run Split next.")
            yield "\n".join(lines[-60:])
            break

        if msg.startswith("__ERROR__"):
            lines.append(f"❌ {msg[9:]}")
            yield "\n".join(lines[-60:])
            break

        lines.append(msg)
        yield "\n".join(lines[-60:])  # show last 60 lines to avoid huge output


def stream_full_pipeline(resume: bool, workers: int) -> Generator[str, None, None]:
    """Generator: generate → split with live progress."""
    dataset = settings_manager.get("DATASET_VERSION")
    updates: queue.SimpleQueue[str] = queue.SimpleQueue()

    thread = threading.Thread(
        target=_run_in_thread_with_queue,
        args=(dataset, resume, int(workers), updates),
        daemon=True,
    )
    thread.start()

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

    # Run split after generation
    try:
        from backend.generation.splitter import DataSplitter

        stats = DataSplitter().run(dataset)
        lines += [
            "✅ Split complete:",
            f"  train: {stats['train']['rows']:,}",
            f"  val:   {stats['val']['rows']:,}",
            f"  test:  {stats['test']['rows']:,}",
            "",
            "Next: EDA tab → Training tab.",
        ]
        yield "\n".join(lines[-60:])
    except Exception as exc:  # noqa: BLE001
        lines.append(f"❌ Split failed: {exc}")
        yield "\n".join(lines[-60:])


def run_split() -> str:
    dataset = settings_manager.get("DATASET_VERSION")
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


def build() -> None:
    """Build the Generation tab. One selection, all sections react."""

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
        label="Cell info",
        interactive=False,
        max_lines=1,
    )

    with gr.Accordion("🔍 System Prompt Preview", open=True):
        gr.Markdown(
            "Auto-updates on selection change. Shows all 8 anti-scenarios. "
            "Run `python cli.py examples-all --workers 20` to populate examples."
        )
        prompt_box = gr.Textbox(
            value=preview_prompt(
                language_choices()[0], industry_choices()[0], scenario_choices()[0]
            ),
            label="System prompt (first turn)",
            lines=40,
            interactive=False,
        )

    with gr.Accordion("📋 Examples for Selected Cell", open=False):
        gr.Markdown(
            "Read-only. Run `python cli.py examples-all --workers 20` to populate all cells."
        )
        load_ex_btn = gr.Button("Show Examples", size="sm")
        examples_box = gr.Textbox(label="Examples", lines=10, interactive=False)
        load_ex_btn.click(fn=show_examples, inputs=[lang_dd, ind_dd, sc_dd], outputs=examples_box)

    with gr.Accordion("🧪 Dry Run — test one cell", open=True):
        gr.Markdown("Generate N sentences for the selected cell. **Nothing is saved.**")
        dry_n = gr.Slider(minimum=3, maximum=50, value=5, step=1, label="N sentences")
        dry_btn = gr.Button("🧪 Generate (no save)", variant="secondary")
        dry_out = gr.Textbox(label="Output", lines=20, interactive=False)
        dry_btn.click(fn=dry_run, inputs=[lang_dd, ind_dd, sc_dd, dry_n], outputs=dry_out)

    with gr.Accordion("🚀 Full Pipeline", open=True):
        dataset = settings_manager.get("DATASET_VERSION")
        cp_every = settings_manager.get("CHECKPOINT_EVERY")
        concurrency = settings_manager.get("API_CONCURRENCY_LIMIT")

        gr.Markdown(
            f"**Dataset:** `{dataset}`  |  "
            f"**API concurrency limit:** `{concurrency}` concurrent calls  "
            f"(edit `API_CONCURRENCY_LIMIT` in `config/settings.py`)\n\n"
            f"Checkpoint every **{cp_every:,}** rows. "
            "Live progress shown below as cells complete.\n\n"
            "⚠️  Start server **without `--reload`** — file watcher kills generation.\n\n"
            "**Resume** = skip completed cells. Only after an interrupted run."
        )

        workers_slider = gr.Slider(
            minimum=1,
            maximum=40,
            value=20,
            step=1,
            label="Workers (parallel cells, 1–40)",
        )
        resume_cb = gr.Checkbox(
            label="Resume from checkpoint (only if previous run was interrupted)",
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

        # Streaming outputs: use generator functions so UI updates live
        gen_btn.click(
            fn=stream_generate,
            inputs=[resume_cb, workers_slider],
            outputs=pipeline_out,
        )
        split_btn.click(
            fn=run_split,
            outputs=pipeline_out,
        )
        full_btn.click(
            fn=stream_full_pipeline,
            inputs=[resume_cb, workers_slider],
            outputs=pipeline_out,
        )

    for dd in [lang_dd, ind_dd, sc_dd]:
        dd.change(
            fn=on_selection_change, inputs=[lang_dd, ind_dd, sc_dd], outputs=[cell_info, prompt_box]
        )
