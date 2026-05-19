"""
Training data generation service.

Multi-turn conversation per cell:
    Turn 1:  Full system prompt → generate first batch
    Turn 2+: "Generate N MORE..." → diversify

RATE LIMIT PROTECTION:
    gpt-5-nano TPM limit: 200,000 tokens/min.
    A simple pause mechanism: all worker threads check a shared threading.Event
    before each API call. The main thread clears the event (pausing all threads)
    after every PAUSE_AFTER_N_CELLS completed cells, sleeps CHECKPOINT_PAUSE_SECONDS,
    then resumes. No semaphores, no token counting — just periodic pauses.

    If you still hit 429s: reduce --workers or increase CHECKPOINT_PAUSE_SECONDS.

TYPING NOTE on client:
    instructor.from_openai() dynamically patches client.chat.completions.create()
    to accept `response_model`. ty cannot resolve these overloads, so client is
    typed as Any — the standard pattern for dynamically-patched libraries.

DO NOT run uvicorn with --reload during generation.
    File watcher restarts the server when checkpoint.csv is written.
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path
from typing import Any

import instructor
from openai import OpenAI

from backend.config.distribution import DISTRIBUTION
from backend.config.keys import IndustryKey, LanguageKey, ScenarioKey, resolve_label
from backend.generation.example_store import example_store
from backend.generation.prompt_factory import PLATFORM_STYLES, PROMPT_FACTORY
from backend.generation.pymodels import GenerationBatch, GenerationCell
from backend.shared.path_resolver import get_dataset_path
from backend.shared.settings_manager import settings_manager

log = logging.getLogger(__name__)

CSV_FIELDS = ["text", "language", "industry", "scenario", "label", "word_count"]
CHECKPOINT_FILE = "checkpoint.csv"
RAW_OUTPUT_FILE = "generated_raw.csv"
MAX_WORKERS = 10  # hard cap — higher values hit TPM rate limits

# Shared pause event. The main thread clears this to pause all worker threads
# between API calls. Set = running, Clear = paused.
generation_pause = threading.Event()
generation_pause.set()

CONTINUE_USER_MSG = (
    "Generate {n} MORE customer messages for the same (language × industry × scenario) combination. "
    "Each 'text' must be the ACTUAL customer message — raw text as the customer would type it in chat. "
    "NOT an instruction, description, or prompt about what to write. "
    "WRONG output: 'Generate a 1-6 word message about checkout error.' "
    "CORRECT output: 'Still shows error at checkout.' "
    "Vary phrasing, specific products, error types, and word count across the length distribution. "
    'Return JSON: {{"prompts": [{{"text": "...", "word_count": N}}, ...]}}'
)


class GeneratorService:
    """Orchestrates training data generation across all distribution cells."""

    def run(
        self,
        dataset_name: str,
        language: LanguageKey | None = None,
        industry: IndustryKey | None = None,
        scenario: ScenarioKey | None = None,
        resume: bool = False,
        max_workers: int = MAX_WORKERS,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Generate training data for all matching distribution cells.

        Args:
            dataset_name: Output folder (e.g. "v1").
            language / industry / scenario: Optional cell filters.
            resume: Skip completed cells from checkpoint.csv.
                    Only use after an interrupted run — never on a fresh start.
            max_workers: Worker threads. Hard cap at MAX_WORKERS (10).
            on_progress: Called after each cell with a progress string.
                         Used by the Gradio streaming UI.
        """
        max_workers = min(max_workers, MAX_WORKERS)

        raw_dir = get_dataset_path(dataset_name) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = raw_dir / CHECKPOINT_FILE
        output_path = raw_dir / RAW_OUTPUT_FILE

        cells = DISTRIBUTION.to_cells()
        if language:
            cells = [c for c in cells if c.language == language]
        if industry:
            cells = [c for c in cells if c.industry == industry]
        if scenario:
            cells = [c for c in cells if c.scenario == scenario]

        existing_rows: list[dict] = []
        completed_ids: set[str] = set()

        if resume and checkpoint_path.exists():
            existing_rows, completed_ids = load_checkpoint(checkpoint_path)
            msg = f"Resumed: {len(existing_rows):,} rows, {len(completed_ids)} cells done"
            log.info(msg)
            if on_progress:
                on_progress(msg)

        cells_to_run = [c for c in cells if c.cell_id not in completed_ids]
        total = len(cells_to_run)

        if not cells_to_run:
            msg = "All cells already complete."
            log.info(msg)
            if on_progress:
                on_progress(msg)
            return

        pause_n = settings_manager.get("PAUSE_AFTER_N_CELLS")
        pause_secs = settings_manager.get("CHECKPOINT_PAUSE_SECONDS")
        cp_every = settings_manager.get("CHECKPOINT_EVERY")

        start_msg = (
            f"Starting {total} cells | workers={max_workers} | "
            f"target: {sum(c.target_count for c in cells_to_run):,} rows | "
            f"pause {pause_secs}s every {pause_n} cells"
        )
        log.info(start_msg)
        if on_progress:
            on_progress(start_msg)

        # Reset pause event before run (in case a previous run left it cleared)
        generation_pause.set()

        all_rows: list[dict] = list(existing_rows)
        rows_lock = threading.Lock()
        cells_done = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(generate_cell_in_thread, cell): cell for cell in cells_to_run
            }

            for future in as_completed(futures):
                cell = futures[future]
                try:
                    cell_id, new_rows = future.result()
                    with rows_lock:
                        all_rows.extend(new_rows)
                        completed_ids.add(cell_id)
                        cells_done += 1

                        msg = (
                            f"[{cells_done}/{total}] {cell_id} "
                            f"({len(new_rows)} rows, total={len(all_rows):,})"
                        )
                        log.info(msg)
                        if on_progress:
                            on_progress(msg)

                        if len(all_rows) % cp_every < len(new_rows) + 1:
                            write_csv(all_rows, checkpoint_path)
                            ckpt_msg = f"  ↳ Checkpoint: {len(all_rows):,} rows"
                            log.info(ckpt_msg)
                            if on_progress:
                                on_progress(ckpt_msg)

                except Exception as exc:  # noqa: BLE001
                    err = f"[{cells_done + 1}/{total}] FAILED: {cell.cell_id} — {exc}"
                    log.error(err)
                    if on_progress:
                        on_progress(err)

                # Pause all threads after every PAUSE_AFTER_N_CELLS cells
                if cells_done % pause_n == 0 and cells_done < total:
                    pause_msg = (
                        f"  ⏸ {cells_done} cells done — pausing {pause_secs}s "
                        f"(rate limit protection)"
                    )
                    log.info(pause_msg)
                    if on_progress:
                        on_progress(pause_msg)

                    generation_pause.clear()  # signal threads to wait
                    time.sleep(pause_secs)
                    generation_pause.set()  # resume

                    if on_progress:
                        on_progress("  ▶ Resuming generation...")

        # Global exact-string dedup
        seen: set[str] = set()
        deduped: list[dict] = [
            row
            for row in all_rows
            if row["text"] not in seen and not seen.add(row["text"])  # type: ignore[func-returns-value]
        ]
        if dropped := len(all_rows) - len(deduped):
            log.info(f"Removed {dropped} cross-cell exact duplicates")

        write_csv(deduped, output_path)
        done_msg = f"Done: {len(deduped):,} rows → {output_path}"
        log.info(done_msg)
        if on_progress:
            on_progress(done_msg)

    def generate_cell(
        self,
        cell: GenerationCell,
        seen_texts: set[str] | None = None,
    ) -> list[dict]:
        """Generate rows for one cell. Used for dry run and single-cell testing."""
        client: Any = instructor.from_openai(OpenAI())
        return run_cell_turns(client, cell, seen_texts or set())


def generate_cell_in_thread(cell: GenerationCell) -> tuple[str, list[dict]]:
    """Thread function: fresh client per thread, independent seen_texts set."""
    client: Any = instructor.from_openai(OpenAI())
    rows = run_cell_turns(client, cell, set())
    return cell.cell_id, rows


def run_cell_turns(
    client: Any,
    cell: GenerationCell,
    seen_texts: set[str],
) -> list[dict]:
    """Multi-turn generation loop for one cell.

    Uses one persistent conversation per cell so the model has perfect recall
    of its own outputs and avoids repetition naturally.

    TYPING: client is Any — instructor dynamically patches OpenAI's
    chat.completions.create() to accept response_model; ty cannot resolve
    these overloads without Any annotation.

    Args:
        client: instructor-patched OpenAI client.
        cell: Target cell defining language, industry, scenario, and count.
        seen_texts: Global dedup set; mutated in place.

    Returns:
        List of row dicts: text, language, industry, scenario, label, word_count.
    """
    label = resolve_label(cell.language, cell.scenario)
    batch_size = settings_manager.get("GENERATION_BATCH_SIZE")
    n_calls = ceil(cell.target_count / batch_size)
    rows: list[dict] = []

    ex = example_store.get(cell.language, cell.industry, cell.scenario)
    messages: list[dict] = []

    for call_idx in range(n_calls):
        n_this_call = min(batch_size, cell.target_count - len(rows))
        if n_this_call <= 0:
            break

        platform_style = PLATFORM_STYLES[call_idx % len(PLATFORM_STYLES)]

        user_content = (
            PROMPT_FACTORY.build(
                language=cell.language,
                industry=cell.industry,
                scenario=cell.scenario,
                examples=ex,
                n=n_this_call,
                platform_style=platform_style,
            )
            if call_idx == 0
            else CONTINUE_USER_MSG.format(n=n_this_call)
        )

        messages.append({"role": "user", "content": user_content})

        try:
            # Blocks if main thread issued a rate-limit pause
            generation_pause.wait()

            batch: GenerationBatch = client.chat.completions.create(
                model=settings_manager.get("GENERATION_LLM"),
                response_model=GenerationBatch,
                messages=messages,
                max_retries=3,
            )
            messages.append({"role": "assistant", "content": batch.model_dump_json()})

            for p in batch.prompts:
                if p.text in seen_texts:
                    continue
                rows.append(
                    {
                        "text": p.text,
                        "language": cell.language,
                        "industry": cell.industry,
                        "scenario": cell.scenario,
                        "label": label,
                        "word_count": p.word_count,
                    }
                )
                seen_texts.add(p.text)

        except Exception as exc:  # noqa: BLE001
            log.warning(f"  {cell.cell_id} call {call_idx + 1} failed: {exc}")
            if messages and messages[-1]["role"] == "user":
                messages.pop()  # keep conversation valid

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: Path) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    completed: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            completed.add(f"{row['language']}__{row['industry']}__{row['scenario']}")
    return rows, completed
