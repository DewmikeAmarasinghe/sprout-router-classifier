"""
ExampleStore — cached per-cell examples.

Thread-safe for parallel examples-all generation.
Handles missing or empty examples.json gracefully.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from backend.config.industry_configs import INDUSTRY_CONFIGS
from backend.config.keys import IndustryKey, LanguageKey, ScenarioKey
from backend.config.language_configs import LANGUAGE_CONFIGS
from backend.config.scenario_configs import SCENARIO_CONFIGS
from backend.shared.settings_manager import settings_manager

log = logging.getLogger(__name__)

EXAMPLES_FILE = Path(__file__).parent / "examples.json"
EXAMPLES_PER_CELL = 8


def _scenario_fallback_examples(scenario: ScenarioKey) -> list[str]:
    """Return LengthRange.examples from ScenarioConfig — zero API calls."""
    sc_cfg = SCENARIO_CONFIGS[scenario]
    examples: list[str] = []
    for r in sc_cfg.length_dist.ranges:
        examples.extend(r.examples)
    return examples[:EXAMPLES_PER_CELL]


class ExampleStore:
    """Per-cell example cache. Thread-safe for parallel access."""

    def __init__(self) -> None:
        self._cache: dict[str, list[str]] = {}
        self._lock = threading.Lock()  # protects _cache + file writes
        self._load()

    def get(
        self,
        language: LanguageKey,
        industry: IndustryKey,
        scenario: ScenarioKey,
        force_regenerate: bool = False,
    ) -> list[str]:
        """Return examples for a cell.

        Priority:
        1. Cached cell-specific examples (from examples.json)
        2. LengthRange.examples fallback (always available, no API)

        Only calls the API when force_regenerate=True.
        """
        key = make_cell_key(language, industry, scenario)

        if not force_regenerate:
            with self._lock:
                if key in self._cache:
                    return list(self._cache[key])

        if force_regenerate:
            examples = self._generate_via_api(language, industry, scenario)
            if examples:
                with self._lock:
                    self._cache[key] = examples
                    self._save_locked()
                return examples

        return _scenario_fallback_examples(scenario)

    def get_anti_examples(
        self,
        scenario: ScenarioKey,
        language: LanguageKey,
        industry: IndustryKey,
    ) -> list[str]:
        """Return examples from anti-scenario keys as contrast examples."""
        sc_cfg = SCENARIO_CONFIGS[scenario]
        anti: list[str] = []
        for anti_key in sc_cfg.anti_scenario_keys[:2]:
            key = make_cell_key(language, industry, anti_key)
            with self._lock:
                cached = self._cache.get(key, [])
            if cached:
                anti.extend(cached[:2])
            else:
                anti.extend(_scenario_fallback_examples(anti_key)[:2])
        return anti

    def is_cached(
        self,
        language: LanguageKey,
        industry: IndustryKey,
        scenario: ScenarioKey,
    ) -> bool:
        with self._lock:
            return make_cell_key(language, industry, scenario) in self._cache

    def coverage(self) -> dict[str, int]:
        with self._lock:
            return {
                lang: sum(1 for k in self._cache if k.startswith(f"{lang}__"))
                for lang in LanguageKey
            }

    def _generate_via_api(
        self,
        language: LanguageKey,
        industry: IndustryKey,
        scenario: ScenarioKey,
    ) -> list[str]:
        from openai import OpenAI

        lang_cfg = LANGUAGE_CONFIGS[language]
        ind_cfg = INDUSTRY_CONFIGS[industry]
        sc_cfg = SCENARIO_CONFIGS[scenario]

        prompt = (
            f"Generate {EXAMPLES_PER_CELL} realistic customer service messages for:\n"
            f"  Language: {lang_cfg.display_name} — {lang_cfg.instruction}\n"
            f"  Industry: {ind_cfg.display_name} — {ind_cfg.description}\n"
            f"  Domain terms: {', '.join(ind_cfg.product_examples[:6])}\n"
            f"  Scenario: {sc_cfg.display_name} — {sc_cfg.description}\n\n"
            f"Requirements:\n"
            f"  - Each message clearly represents this scenario\n"
            f"  - Written in the specified language format\n"
            f"  - Vary lengths: {sc_cfg.length_dist.to_prompt_str()}\n"
            f"  - Realistic for {ind_cfg.typical_platform}\n\n"
            f'Return ONLY valid JSON: {{"examples": ["msg1", ..., "msg{EXAMPLES_PER_CELL}"]}}'
        )

        client = OpenAI()
        response = client.chat.completions.create(
            model=settings_manager.get("GENERATION_LLM"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if content is None:
            return []

        data: dict = json.loads(content)
        return data.get("examples", [])[:EXAMPLES_PER_CELL]

    def _load(self) -> None:
        """Load cache. Handles missing, empty, or malformed files gracefully."""
        if not EXAMPLES_FILE.exists():
            self._cache = {}
            return

        text = EXAMPLES_FILE.read_text(encoding="utf-8").strip()
        if not text:
            self._cache = {}
            return

        try:
            self._cache = json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning(f"examples.json is malformed: {exc}. Starting with empty cache.")
            self._cache = {}

    def _save_locked(self) -> None:
        """Write cache to disk. Must be called while holding self._lock."""
        EXAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
        EXAMPLES_FILE.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def make_cell_key(
    language: LanguageKey,
    industry: IndustryKey,
    scenario: ScenarioKey,
) -> str:
    return f"{language}__{industry}__{scenario}"


example_store = ExampleStore()
