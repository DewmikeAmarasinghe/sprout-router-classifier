# AGENTS.md — Coding Standards & Architecture Guide

Place this file at the project root. Cursor reads it automatically.
Read docs/00_MASTER_README.md first for project context.

---

## MANDATORY CHECKS (run before every commit)

```bash
ruff check .          # lint + import sort
ruff format .         # auto-format (line length 100)
ty check .            # type checking
pytest tests/ -v      # all tests must pass
```

`ruff check --fix .` fixes most issues automatically.
Fix all ruff and ty errors before considering any task done.

---

## PROJECT CONTEXT

Binary text classifier: routes Sprout chatbot messages to gpt-4o-mini (0) or gpt-4o (1).
Read order: docs/00_MASTER_README.md → docs/01_ARCHITECTURE.md → phase-specific plan.

---

## ARCHITECTURE RULES (never violate)

1. **backend/ owns all logic.**
   `phases/` and `frontend/panels/` and `api/routes_*` are thin wrappers.
   Never put business logic in routes, panels, or CLI scripts.

2. **config/ is the single source of truth.**
   Import from `config/settings.py` via `settings_manager.get(...)`.
   Never hardcode paths, model names, thresholds, or counts anywhere else.

3. **shared/ has no internal imports.**
   `script_detector.py`, `metrics.py`, `path_resolver.py`, `settings_manager.py`
   import only stdlib + third-party. Never import from backend modules.

4. **pymodels.py per module for Pydantic models.**
   `generation/pymodels.py` holds generation-boundary shapes.
   `training/pymodels.py` holds training-boundary shapes.
   Co-location means less cross-module coupling.

5. **Frozen dataclass for trusted internal configs.**
   `LanguageConfig`, `IndustryConfig`, `ScenarioConfig` — frozen dataclasses.
   Pydantic only where users can edit values or data crosses a trust boundary.

6. **StrEnum for all domain keys.**
   `LanguageKey`, `IndustryKey`, `ScenarioKey` — all StrEnum.
   Members serialize as strings, enable IDE autocomplete, give instant squiggles on typos.

7. **Tests are mandatory.**
   Every public service method needs a corresponding test.
   Services must be testable without starting a UI or API server.

---

## CODE STYLE

```python
# Type hints on every function signature
def build_prompt(language: LanguageKey, industry: IndustryKey, n: int) -> str: ...

# Docstrings: one-line summary for all public methods
def generate_cell(self, cell: GenerationCell) -> list[dict]:
    """Generate target_count prompts for one distribution cell."""

# Constants: UPPER_SNAKE_CASE in config/
# Classes: PascalCase
# Functions / variables: snake_case
# No underscore prefix for non-private methods (avoid _helper everywhere)

# Early return over nested if
if is_pure_script(text):
    return RouterPrediction(label=1, confidence=1.0, rule="script_detector")

# Explicit names, not magic numbers
label = SAFE_DEFAULT_LABEL   # not: label = 1

# Pathlib everywhere
model_path = get_experiment_path("v1", "classical") / "models" / f"{key}.pkl"
model_path.parent.mkdir(parents=True, exist_ok=True)

# Pydantic v2 API
model.model_dump()   # not .dict()
Model.model_validate(data)   # not Model.parse_obj(data)
```

---

## TRAINING DATA RULES

- ALL 60,000 training rows use English alphabet letters (romanized).
- `singlish_light` = mostly English with 1–3 ROMANIZED Sinhala words.
- `singlish_heavy` = predominantly ROMANIZED Sinhala vocabulary.
- `tanglish_light` = mostly English with 1–3 ROMANIZED Tamil words.
- `tanglish_heavy` = predominantly ROMANIZED Tamil vocabulary.
- No Sinhala/Tamil script in training data. `is_pure_script()` handles those.
- Location names CAN and SHOULD appear in any prompt type.
  Routing signal is INTENT, not presence/absence of a location name.
- Reference points in location_relative are NOT limited to business locations:
  schools, landmarks, junctions, roads, buildings, neighborhoods.

---

## PLATFORM CONTEXT

Sprout is deployed on multiple channels, NOT just WhatsApp:
  WhatsApp Business, Instagram DMs, Facebook Messenger, Viber, SMS,
  website chatbot widgets, mobile app embedded chat, Shopify/WooCommerce.

Platform style rotates across 8 styles per API call in PLATFORM_STYLES tuple.
It is NOT a 4th hierarchy level — the routing signal is language/intent, not platform.

---

## GENERATION MODEL

`gpt-5-nano` in `config/settings.py`.
Read via `settings_manager.get("GENERATION_LLM")`.
50 prompts per API call (GENERATION_BATCH_SIZE).
Always use `instructor` library for structured outputs — never raw JSON parsing.

---

## DISTRIBUTION CONFIG RULES

Three-level fraction hierarchy: LanguageBucket → IndustryBucket → ScenarioBucket.
All fractions at each level must sum to 1.0 — Pydantic validates at import time.
DISTRIBUTION = GenerationDistribution(...).resolve() runs at module load.
Misconfigured fractions = ValidationError = visible squiggly in VS Code immediately.

Adding a new scenario: 5 steps (see docs/02_DATA_PLAN.md — PROMPT FACTORY section).
Adding a new language: add LanguageKey + LanguageConfig + LanguageBucket.
Adding a new industry: add IndustryKey + IndustryConfig + IndustryBucket entries.

---

## ROLLING CONTEXT

GeneratorService passes 50 RANDOM samples from ALL previously generated rows
in the current cell to each API call.
Not the last 25 — full random coverage prevents repetition across the whole cell.
No semantic dedup — rolling context handles this. Exact-string dedup only.

---

## COMMON PATTERNS

```python
# Reading settings
from backend.shared.settings_manager import settings_manager
model = settings_manager.get("GENERATION_LLM")

# Distribution access
from backend.config.distribution import DISTRIBUTION
cell = DISTRIBUTION.find_cell(LanguageKey.SINGLISH_LIGHT, IndustryKey.BANKING, ScenarioKey.CONTINUATION)

# Prompt building
from backend.generation.prompt_factory import PROMPT_FACTORY
prompt = PROMPT_FACTORY.build(language=..., industry=..., scenario=..., ...)

# Path resolution
from backend.shared.path_resolver import get_dataset_path
path = get_dataset_path("v1") / "train.csv"

# Resolve label
from backend.config.keys import resolve_label
label = resolve_label(LanguageKey.SINGLISH_LIGHT, ScenarioKey.SIMPLE_TRANSACTIONAL)  # → 1
```

---

## STUB IMPLEMENTATION GUIDE

When implementing a file with `raise NotImplementedError`:
1. Read the corresponding `docs/` plan section.
2. Implement all methods with type hints and docstrings.
3. Run `ruff check . && ty check . && pytest tests/ -v` — all must pass.
4. No `pass` or `raise NotImplementedError` in final committed code.