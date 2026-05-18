"""
Training data generation service with parallel cell execution and proper rate limiting.

RATE LIMIT STRATEGY (two-layer):
    Layer 1 — Global Semaphore (API_CONCURRENCY_LIMIT in settings.py):
        Caps concurrent in-flight API calls regardless of worker count.
        Default 15 — prevents the burst that eats RPM budget in 1 second.
    Layer 2 — Retry-After Backoff:
        Reads the retry-after header from 429 for exact wait time.
        Falls back to exponential backoff (5s, 10s, 20s, 40s, 80s) if absent.

DO NOT run uvicorn with --reload during generation.
    File watcher restarts the server when checkpoint.csv changes.

DO NOT run phase_1_grounding.py and phase_2_generate.py simultaneously.
    They share the same API key and rate limit budget.
"""

from __future__ import annotations

import csv
import logging
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path
from typing import Any

import instructor
from openai import OpenAI, RateLimitError

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
MAX_RATE_LIMIT_RETRIES = 6

CONTINUE_USER_MSG = (
    "Generate {n} MORE unique messages for the same combination. "
    "All must be meaningfully different from what you generated above — "
    "vary phrasing, context, length, and specific details. "
    'Return JSON: {{"prompts": [{{"text": "...", "word_count": N}}, ...]}}'
)

_api_semaphore: threading.Semaphore | None = None
_semaphore_lock = threading.Lock()


def get_api_semaphore() -> threading.Semaphore:
    """Lazy-init the global API semaphore with the configured concurrency limit."""
    global _api_semaphore  # noqa: PLW0603
    with _semaphore_lock:
        if _api_semaphore is None:
            limit = settings_manager.get("API_CONCURRENCY_LIMIT")
            _api_semaphore = threading.Semaphore(limit)
            log.info(f"API concurrency semaphore: limit={limit}")
    return _api_semaphore


def parse_retry_after(exc: RateLimitError) -> float | None:
    """Extract the exact wait time from the 429 response headers."""
    try:
        headers = exc.response.headers  # type: ignore[union-attr]
        if "retry-after-ms" in headers:
            return float(headers["retry-after-ms"]) / 1000.0
        if "retry-after" in headers:
            return float(headers["retry-after"])
        match = re.search(r"try again in (\d+(?:\.\d+)?)s", str(exc), re.IGNORECASE)
        if match:
            return float(match.group(1)) + 0.5
    except Exception:  # noqa: BLE001
        pass
    return None


def call_with_rate_limit(client: Any, **kwargs: Any) -> Any:
    """Semaphore-gated API call with retry-after-aware backoff on 429."""
    semaphore = get_api_semaphore()
    base_backoff = 5.0

    with semaphore:
        for attempt in range(MAX_RATE_LIMIT_RETRIES):
            try:
                return client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                if attempt == MAX_RATE_LIMIT_RETRIES - 1:
                    log.error(f"Rate limit: gave up after {MAX_RATE_LIMIT_RETRIES} attempts")
                    raise
                wait = parse_retry_after(exc) or min(base_backoff * (2**attempt), 120.0)
                log.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES}). "
                    f"Waiting {wait:.1f}s..."
                )
                time.sleep(wait)


class GeneratorService:
    """Orchestrates parallel training data generation with progress reporting."""

    def run(
        self,
        dataset_name: str,
        language: LanguageKey | None = None,
        industry: IndustryKey | None = None,
        scenario: ScenarioKey | None = None,
        resume: bool = False,
        max_workers: int = 20,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Generate training data for all matching cells in parallel.

        Args:
            dataset_name: Output folder (e.g. "v1").
            language / industry / scenario: Optional cell filters.
            resume: Skip completed cells from checkpoint.csv.
                    Use ONLY after an interrupted run.
            max_workers: Worker threads. Default 20, capped at 40.
            on_progress: Optional callback called after each cell completes.
                         Receives a formatted progress string. Used by the
                         Gradio streaming UI to show live progress.
        """
        max_workers = min(max_workers, 40)

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

        completed_ids: set[str] = set()
        existing_rows: list[dict] = []

        if resume and checkpoint_path.exists():
            existing_rows, completed_ids = load_checkpoint(checkpoint_path)
            msg = f"Resumed: {len(existing_rows):,} rows, {len(completed_ids)} cells done"
            log.info(msg)
            if on_progress:
                on_progress(msg)
        elif resume:
            log.info("resume=True but no checkpoint.csv — starting fresh")

        cells_to_run = [c for c in cells if c.cell_id not in completed_ids]
        total = len(cells_to_run)

        if not cells_to_run:
            msg = "All cells already complete."
            log.info(msg)
            if on_progress:
                on_progress(msg)
            return

        concurrency = settings_manager.get("API_CONCURRENCY_LIMIT")
        start_msg = (
            f"Starting {total} cells | workers={max_workers} | "
            f"api_concurrency={concurrency} | "
            f"target: {sum(c.target_count for c in cells_to_run):,} rows"
        )
        log.info(start_msg)
        if on_progress:
            on_progress(start_msg)

        all_rows: list[dict] = list(existing_rows)
        rows_lock = threading.Lock()
        checkpoint_every: int = settings_manager.get("CHECKPOINT_EVERY")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(generate_cell_in_thread, cell): cell for cell in cells_to_run
            }
            for i, future in enumerate(as_completed(futures), 1):
                cell = futures[future]
                try:
                    cell_id, new_rows = future.result()
                    with rows_lock:
                        all_rows.extend(new_rows)
                        completed_ids.add(cell_id)

                        cell_msg = (
                            f"[{i}/{total}] Done: {cell_id} "
                            f"({len(new_rows)} rows, total={len(all_rows):,})"
                        )
                        log.info(cell_msg)
                        if on_progress:
                            on_progress(cell_msg)

                        if len(all_rows) % checkpoint_every < len(new_rows) + 1:
                            write_csv(all_rows, checkpoint_path)
                            ckpt_msg = f"  ↳ Checkpoint saved: {len(all_rows):,} rows"
                            log.info(ckpt_msg)
                            if on_progress:
                                on_progress(ckpt_msg)

                except Exception as exc:  # noqa: BLE001
                    err_msg = f"[{i}/{total}] FAILED: {cell.cell_id} — {exc}"
                    log.error(err_msg)
                    if on_progress:
                        on_progress(err_msg)

        seen: set[str] = set()
        deduped = [
            row
            for row in all_rows
            if row["text"] not in seen and not seen.add(row["text"])  # type: ignore[func-returns-value]
        ]
        dropped = len(all_rows) - len(deduped)
        if dropped:
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
        """Generate rows for one cell. Used for dry run and direct testing."""
        client: Any = instructor.from_openai(OpenAI())
        return run_cell_turns(client, cell, seen_texts or set())


def generate_cell_in_thread(cell: GenerationCell) -> tuple[str, list[dict]]:
    """Thread function: fresh client per thread, independent local seen_texts."""
    client: Any = instructor.from_openai(OpenAI())
    rows = run_cell_turns(client, cell, set())
    return cell.cell_id, rows


def run_cell_turns(
    client: Any,
    cell: GenerationCell,
    seen_texts: set[str],
) -> list[dict]:
    """Multi-turn generation loop for one cell.

    Target rows: n_this_call = min(batch_size, target - len(rows))
    For target=159: 50 + 50 + 50 + 9 = exactly 159.
    """
    label = resolve_label(cell.language, cell.scenario)
    batch_size = settings_manager.get("GENERATION_BATCH_SIZE")
    n_calls = ceil(cell.target_count / batch_size)
    rows: list[dict] = []

    examples = example_store.get(cell.language, cell.industry, cell.scenario)
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
                examples=examples,
                n=n_this_call,
                platform_style=platform_style,
            )
            if call_idx == 0
            else CONTINUE_USER_MSG.format(n=n_this_call)
        )

        messages.append({"role": "user", "content": user_content})

        try:
            batch: GenerationBatch = call_with_rate_limit(
                client,
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

            log.debug(
                f"  {cell.cell_id} call {call_idx + 1}/{n_calls}: {len(batch.prompts)} prompts"
            )

        except Exception as exc:  # noqa: BLE001
            log.warning(f"  {cell.cell_id} call {call_idx + 1} failed: {exc}")
            if messages and messages[-1]["role"] == "user":
                messages.pop()

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: Path) -> tuple[list[dict], set[str]]:
    """Load rows and completed cell IDs from checkpoint.csv."""
    rows: list[dict] = []
    completed: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            completed.add(f"{row['language']}__{row['industry']}__{row['scenario']}")
    return rows, completed
