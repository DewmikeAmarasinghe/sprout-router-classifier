"""
Training data generation service.

SYSTEM vs USER MESSAGE STRUCTURE:

  Turn 0 — First batch:
    system: PROMPT_FACTORY.build()  ← all context, language/industry/scenario/rules
    user:   "━━━ Batch 1 of ~N | Session target: T messages | This batch: 50 ━━━
             Generate exactly 50 messages now."

  Turn 1+ — Continuation batches:
    user:   CONTINUE_USER_MSG (turn number, generated so far, remaining, this batch N)

  This separation means:
    - Gradio system prompt preview shows only context (no hardcoded count)
    - Each user message clearly states progress and what to do next

FAILURE RECOVERY:
  Turn 0 failure: pop user message, reset messages=[] → rebuilds full system+user next iteration
  Turn N>0 failure: pop user message → system + previous turns intact → continues with next batch

word_count: computed as len(text.split()) locally. Not asked from the LLM.
"""

from __future__ import annotations

import collections
import csv
import logging
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
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
MAX_WORKERS = 10
MAX_TURNS_PER_CELL = 100
# Keep only this many recent (user, assistant) turn pairs in the conversation
# history. The system prompt always stays. This prevents context rot — the model
# copying from its own 500+ message history instead of generating new content.
MAX_HISTORY_TURNS = 3

generation_pause = threading.Event()
generation_pause.set()

# Continuation turn user message (turn 1+).
# Progress header gives the LLM full context so it can:
#   1. Understand how much variety is still needed
#   2. Know exactly how many to generate this turn
CONTINUE_USER_MSG = (
    "━━━ Batch {turn} | Progress: {generated}/{target} messages | This batch: {n} ━━━\n\n"
    "Generate exactly {n} MORE unique customer messages for this cell.\n\n"
    "  - Raw message text only — exactly what a customer would type\n"
    "  - Every message in this batch must be unique from each other AND from previous batches\n"
    "  - Vary sentence structure — don't reuse the same grammatical template\n"
    "  - Mix lengths as shown in the system prompt (follow the length distribution guide)\n"
    "  - Stay within the language × industry × scenario in your system context\n\n"
    'Return ONLY: {{"prompts": [{{"text": "..."}}]}}'
)


def build_first_turn_user_msg(n: int, total_target: int, batch_size: int) -> str:
    """Build the user message for turn 0 (first batch).

    Tells the LLM the session total target and approximate number of batches,
    so it understands the overall scope and can plan variety accordingly.
    """
    n_batches = math.ceil(total_target / batch_size)
    return (
        f"━━━ Batch 1 of ~{n_batches} | Session target: {total_target:,} messages | "
        f"This batch: {n} ━━━\n\n"
        f"Generate exactly {n} messages now."
    )


class GeneratorService:
    def run(
        self,
        dataset_name: str,
        language: LanguageKey | None = None,
        industry: IndustryKey | None = None,
        scenario: ScenarioKey | None = None,
        resume: bool = False,
        max_workers: int = MAX_WORKERS,
        max_cells: int | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Generate training data for all matching distribution cells.

        Args:
            resume:    Skip cells already in checkpoint.csv (even if underfilled).
            max_cells: Stop after completing this many cells. Run again (or --fill-gaps)
                       to continue with the remaining ones.
        """
        max_workers = min(max_workers, MAX_WORKERS)
        raw_dir = get_dataset_path(dataset_name) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = raw_dir / CHECKPOINT_FILE
        output_path = raw_dir / RAW_OUTPUT_FILE

        cells = filter_cells(DISTRIBUTION.to_cells(), language, industry, scenario)

        existing_rows: list[dict] = []
        completed_ids: set[str] = set()

        if checkpoint_path.exists():
            existing_rows, completed_ids = load_checkpoint(checkpoint_path)
            if completed_ids:
                msg = f"Skipping {len(completed_ids)} already-completed cells"
                log.info(msg)
                if on_progress:
                    on_progress(msg)

        cells_to_run = [c for c in cells if c.cell_id not in completed_ids]
        if not cells_to_run:
            msg = "All cells already complete."
            log.info(msg)
            if on_progress:
                on_progress(msg)
            return

        if max_cells is not None:
            cells_to_run = cells_to_run[:max_cells]
            log.info(f"--cells {max_cells}: running {len(cells_to_run)} cells this session")

        run_cells(
            cells_to_run=cells_to_run,
            existing_rows=existing_rows,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            max_workers=max_workers,
            on_progress=on_progress,
        )

    def fill_gaps(
        self,
        dataset_name: str,
        language: LanguageKey | None = None,
        industry: IndustryKey | None = None,
        scenario: ScenarioKey | None = None,
        max_workers: int = MAX_WORKERS,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Detect and top up cells that have fewer rows than their target.

        Works after Ctrl+C, rate-limit partial runs, or any incomplete run.
        Recommend --workers 6 for fill runs (safer TPM budget than 10).
        """
        raw_dir = get_dataset_path(dataset_name) / "raw"
        checkpoint_path = raw_dir / CHECKPOINT_FILE

        if not checkpoint_path.exists():
            log.warning("No checkpoint.csv found. Run generation first.")
            return

        existing_rows, _ = load_checkpoint(checkpoint_path)

        cell_counts: dict[str, int] = collections.Counter(
            f"{row['language']}__{row['industry']}__{row['scenario']}" for row in existing_rows
        )

        cells = filter_cells(DISTRIBUTION.to_cells(), language, industry, scenario)
        fill_cells = [
            GenerationCell(
                language=c.language,
                industry=c.industry,
                scenario=c.scenario,
                target_count=c.target_count - cell_counts.get(c.cell_id, 0),
            )
            for c in cells
            if cell_counts.get(c.cell_id, 0) < c.target_count
        ]

        if not fill_cells:
            msg = "All cells fully generated. Nothing to fill."
            log.info(msg)
            if on_progress:
                on_progress(msg)
            return

        total_missing = sum(c.target_count for c in fill_cells)
        msg = (
            f"Found {len(fill_cells)} underfilled cells, "
            f"{total_missing:,} rows missing. Running fill with {max_workers} workers..."
        )
        log.info(msg)
        if on_progress:
            on_progress(msg)

        run_cells(
            cells_to_run=fill_cells,
            existing_rows=existing_rows,
            checkpoint_path=checkpoint_path,
            output_path=raw_dir / RAW_OUTPUT_FILE,
            max_workers=min(max_workers, MAX_WORKERS),
            on_progress=on_progress,
        )

    def generate_cell(
        self,
        cell: GenerationCell,
        seen_texts: set[str] | None = None,
    ) -> list[dict]:
        """Generate rows for a single cell. Used by cli.py and panel_generation.py."""
        client: Any = instructor.from_openai(OpenAI())
        return run_cell_turns(client, cell, seen_texts or set())


def filter_cells(
    cells: list[GenerationCell],
    language: LanguageKey | None,
    industry: IndustryKey | None,
    scenario: ScenarioKey | None,
) -> list[GenerationCell]:
    if language:
        cells = [c for c in cells if c.language == language]
    if industry:
        cells = [c for c in cells if c.industry == industry]
    if scenario:
        cells = [c for c in cells if c.scenario == scenario]
    return cells


def run_cells(
    cells_to_run: list[GenerationCell],
    existing_rows: list[dict],
    checkpoint_path: Path,
    output_path: Path,
    max_workers: int,
    on_progress: Callable[[str], None] | None,
) -> None:
    pause_n = settings_manager.get("PAUSE_AFTER_N_TURNS")
    pause_secs = settings_manager.get("CHECKPOINT_PAUSE_SECONDS")
    total = len(cells_to_run)

    start_msg = (
        f"Starting {total} cells | workers={max_workers} | "
        f"target: {sum(c.target_count for c in cells_to_run):,} rows | "
        f"pause {pause_secs}s every {pause_n} API turns"
    )
    log.info(start_msg)
    if on_progress:
        on_progress(start_msg)

    generation_pause.set()

    all_rows: list[dict] = list(existing_rows)
    rows_lock = threading.Lock()
    cells_done = 0
    cell_row_counts: dict[str, int] = {}
    # Single-element lists so closures can mutate without nonlocal.
    turns_done = [0]
    last_pause_at = [0]  # turns_done value when the last pause was claimed
    pause_lock = threading.Lock()  # only one worker executes a pause at a time

    def make_on_rows(cell_id: str) -> Callable[[list[dict]], None]:
        """Batch callback: appends rows to checkpoint every turn, pauses every pause_n turns.

        Cascading-pause prevention: each worker claims a pause slot by advancing
        last_pause_at under rows_lock, but only the first to win pause_lock actually
        sleeps. Others skip — the next pause fires pause_n turns later.
        """

        def on_rows(batch_rows: list[dict]) -> None:
            do_pause = False
            with rows_lock:
                all_rows.extend(batch_rows)
                cell_row_counts[cell_id] = cell_row_counts.get(cell_id, 0) + len(batch_rows)
                if batch_rows:
                    append_to_checkpoint(batch_rows, checkpoint_path)
                turns_done[0] += 1
                if turns_done[0] - last_pause_at[0] >= pause_n:
                    last_pause_at[0] = turns_done[0]
                    do_pause = True

            if do_pause and pause_lock.acquire(blocking=False):
                try:
                    pause_msg = f"  ⏸ {turns_done[0]} API turns — pausing {pause_secs}s"
                    log.info(pause_msg)
                    if on_progress:
                        on_progress(pause_msg)
                    generation_pause.clear()
                    time.sleep(pause_secs)
                    generation_pause.set()
                    log.info("  ▶ Resuming generation...")
                    if on_progress:
                        on_progress("  ▶ Resuming generation...")
                finally:
                    pause_lock.release()

        return on_rows

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_cell_in_thread, cell, make_on_rows(cell.cell_id)): cell
            for cell in cells_to_run
        }

        for future in as_completed(futures):
            cell = futures[future]
            try:
                cell_id = future.result()
                with rows_lock:
                    cells_done += 1
                    cell_rows = cell_row_counts.get(cell_id, 0)
                    msg = (
                        f"[{cells_done}/{total}] {cell_id} "
                        f"({cell_rows} rows, total={len(all_rows):,})"
                    )
                    log.info(msg)
                    if on_progress:
                        on_progress(msg)

            except Exception as exc:  # noqa: BLE001
                cells_done += 1
                err = f"[{cells_done}/{total}] FAILED: {cell.cell_id} — {exc}"
                log.error(err)
                if on_progress:
                    on_progress(err)

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


def generate_cell_in_thread(
    cell: GenerationCell,
    on_rows: Callable[[list[dict]], None],
) -> str:
    """Generate rows for a cell, streaming each batch via on_rows. Returns cell_id."""
    client: Any = instructor.from_openai(OpenAI())
    run_cell_turns(client, cell, set(), on_rows)
    return cell.cell_id


def run_cell_turns(
    client: Any,
    cell: GenerationCell,
    seen_texts: set[str],
    on_rows: Callable[[list[dict]], None] | None = None,
) -> list[dict]:
    """Multi-turn generation loop with clear system+user message split.

    Turn 0 structure:
      system: all context (language, industry, scenario, examples, rules)
      user:   "━━━ Batch 1 of ~N | Session target: T | This batch: n ━━━ Generate n messages."

    Turn 1+ structure:
      user: CONTINUE_USER_MSG (batch number, progress, remaining, this batch n)

    Failure recovery:
      Turn 0 fail → pop user msg, reset messages=[] → next iter rebuilds system+user
      Turn N>0 fail → pop user msg, keep system+history → next iter uses CONTINUE_USER_MSG

    word_count = len(text.split()) — computed locally, not asked from LLM.
    """
    label = resolve_label(cell.language, cell.scenario)
    batch_size = settings_manager.get("GENERATION_BATCH_SIZE")
    rows: list[dict] = []
    messages: list[dict] = []  # empty = no successful turn yet
    turn_idx = 0

    ex = example_store.get(cell.language, cell.industry, cell.scenario)

    while len(rows) < cell.target_count and turn_idx < MAX_TURNS_PER_CELL:
        n_this_turn = min(batch_size, cell.target_count - len(rows))
        platform_style = PLATFORM_STYLES[turn_idx % len(PLATFORM_STYLES)]

        if not messages:
            # Turn 0 (or retry after turn-0 failure)
            system_content = PROMPT_FACTORY.build(
                language=cell.language,
                industry=cell.industry,
                scenario=cell.scenario,
                examples=ex,
                platform_style=platform_style,
            )
            messages = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": build_first_turn_user_msg(
                        n_this_turn, cell.target_count, batch_size
                    ),
                },
            ]
        else:
            messages.append(
                {
                    "role": "user",
                    "content": CONTINUE_USER_MSG.format(
                        turn=turn_idx + 1,
                        generated=len(rows),
                        target=cell.target_count,
                        n=n_this_turn,
                    ),
                }
            )

        try:
            generation_pause.wait()

            batch: GenerationBatch = client.chat.completions.create(
                model=settings_manager.get("GENERATION_LLM"),
                response_model=GenerationBatch,
                messages=messages,
                max_retries=3,
            )
            messages.append({"role": "assistant", "content": batch.model_dump_json()})

            # Sliding window: keep system prompt + last MAX_HISTORY_TURNS (user, assistant) pairs.
            # Prevents context rot — model copying its own 500+ message history verbatim.
            max_msgs = 1 + MAX_HISTORY_TURNS * 2
            if len(messages) > max_msgs:
                messages = [messages[0]] + messages[-MAX_HISTORY_TURNS * 2 :]

            batch_rows: list[dict] = []
            n_empty = 0
            n_dupes = 0
            for p in batch.prompts:
                text = p.text.strip()
                if not text:
                    n_empty += 1
                    continue
                if text in seen_texts:
                    n_dupes += 1
                    continue
                batch_rows.append(
                    {
                        "text": text,
                        "language": cell.language,
                        "industry": cell.industry,
                        "scenario": cell.scenario,
                        "label": label,
                        "word_count": len(text.split()),
                    }
                )
                seen_texts.add(text)

            rows.extend(batch_rows)
            if on_rows:
                on_rows(batch_rows)

            pct = len(rows) / cell.target_count * 100
            log.info(
                f"  {cell.cell_id}: turn {turn_idx + 1}"
                f" | gen={len(batch.prompts)} empty={n_empty} dupes={n_dupes} +{len(batch_rows)}"
                f" | {len(rows)}/{cell.target_count} ({pct:.0f}%)"
            )

        except Exception as exc:  # noqa: BLE001
            log.warning(f"  {cell.cell_id} turn {turn_idx + 1} failed: {exc!s:.200}")
            if messages and messages[-1]["role"] == "user":
                messages.pop()
            # Only system message left → turn 0 failed entirely → reset for fresh retry
            if len(messages) == 1 and messages[0].get("role") == "system":
                messages = []

        turn_idx += 1

    if len(rows) < cell.target_count:
        log.warning(
            f"  {cell.cell_id}: MAX_TURNS ({MAX_TURNS_PER_CELL}) hit, "
            f"generated {len(rows)}/{cell.target_count}"
        )

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_to_checkpoint(batch_rows: list[dict], path: Path) -> None:
    """Append a single batch to checkpoint.csv (write header only if file is new/empty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(batch_rows)


def load_checkpoint(path: Path) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    completed: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            completed.add(f"{row['language']}__{row['industry']}__{row['scenario']}")
    return rows, completed
