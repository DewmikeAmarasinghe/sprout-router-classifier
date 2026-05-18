# sprout-router-classifier — Master Plan (v6)

## PROJECT OVERVIEW

hSenid Mobile's Sprout chatbot routes ALL messages to gpt-4o, which is expensive.
This project builds a fast binary classifier to decide:

  label=0  →  gpt-4o-mini   (pure English, simple, no complexity signals)
  label=1  →  gpt-4o        (Singlish, Tanglish, location reasoning, complex tasks,
                              sensitive context, escalation, response language, continuation)

False negative (predicting 0 when it should be 1) = bad UX. Always default to 1 when uncertain.

---

## HOW TO READ THESE FILES

  00_MASTER_README.md       ← You are here
  01_ARCHITECTURE.md        ← System layers, routing signals, code architecture
  02_DATA_PLAN.md           ← 3-level hierarchy, generation strategy, pipeline
  03_CLASSICAL_ML_PLAN.md   ← All classical ML approaches explained
  04_TRANSFORMERS_PLAN.md   ← Fine-tuning pipeline + model guide
  05_EVALUATION_PLAN.md     ← Metrics, ablation, cost simulation
  06_EXPERIMENT_TRACKING.md ← MLflow setup, dataset logging
  07_FILE_TREE.md           ← Complete file structure
  08_WORKFLOW.md            ← VS Code + Kaggle GPU workflow
  TASKS.md                  ← Step-by-step implementation checklist
  HOW_SYSTEM_PROMPTS_WORK.md ← System prompt architecture explained

---

## LABEL=1 CONDITIONS (9 routing signals)

  1. Singlish — romanized Sinhala words in English letters (light or heavy)
  2. Tanglish — romanized Tamil words in English letters (light or heavy)
  3. Pure Sinhala/Tamil script → caught by unicode rule, not the ML model
  4. Reply-in-language request → "answer in sinhala", "tamil la kiyanna"
  5. Location proximity intent → "nearest branch to me" (no named place needed)
  6. Location relative → spatial relationship between named places OR landmarks
     "from Kandy which branch", "near Dharmapala Vidyalaya", "100m from the Cargills"
  7. Complex multi-step task → policy comparison, multi-condition eligibility
  8. Sensitive context → fraud, medical concern, financial distress
  9. Escalation → complaint, frustration, manager request
  10. Response language request → "answer in sinhala"
  11. Continuation → previous action failed OR intent is genuinely unclear

  label=0 ONLY: pure English + simple_transactional OR named_location (simple lookup)

---

## THREE SYSTEM LAYERS

  LAYER 1 — UNICODE DETECTOR     < 0.1ms    is_pure_script(text) → bool
  LAYER 2 — ML CLASSIFIER        1–30ms     trained binary model
  LAYER 3 — SAFE DEFAULT          0ms       confidence < threshold → label=1

---

## DATASET STRATEGY

  Generation target: ~60,000 raw rows
  After quality filter (~12%) + exact dedup (~2%): ~50,000 clean rows

  3-LEVEL HIERARCHY:
    Level 1 — LanguageKey     (pure_english, singlish_light, singlish_heavy,
                               tanglish_light, tanglish_heavy)
    Level 2 — IndustryKey     (ecommerce, healthcare, banking, insurance,
                               telecom, logistics, hospitality, education)
    Level 3 — ScenarioKey     (9 scenarios — see 02_DATA_PLAN.md)

  DIVERSITY STRATEGY:
    50 prompts per API call (max quality vs diversity sweet spot)
    Rolling context: 50 RANDOM samples from ALL previously generated rows
    in the current cell passed to each API call — LLM naturally avoids repetition
    No semantic embedding dedup — rolling context handles this
    Exact-string dedup (hash set, free) catches literal duplicates only

  INDIA EXPANSION:
    tanglish_light + tanglish_heavy together = 38% of total rows
    boosted from what Sri Lanka alone would justify

---

## TOOLS

  instructor          Pydantic-enforced structured output from OpenAI API
  openai              API calls for generation and evaluation
  FastAPI + Gradio    Backend API + frontend UI (mounted at /ui)
  MLflow              Experiment tracking + dataset lineage
  sentence-transformers  NOT used for dedup (removed). Used only for EDA similarity plots.
  Optuna              HPO
  transformers        HuggingFace fine-tuning
  ruff + ty           Linting + type checking (run before every commit)

---

## SOURCE LAYOUT

  src/backend/        ← all server-side logic (config, generation, training, evaluation, router, api)
  src/frontend/       ← Gradio UI panels (calls backend services only, no logic)
  phases/             ← CLI entry points for Kaggle (thin wrappers calling backend)
  tests/              ← pytest test suite

---

## KEY ARCHITECTURE PRINCIPLES

  1. StrEnum for all domain keys — IDE autocomplete, type safety, instant squiggles on typos
  2. Frozen dataclass for trusted internal configs (LanguageConfig, IndustryConfig, ScenarioConfig)
  3. Pydantic for data crossing trust boundaries (fraction validation, API responses)
  4. pymodels.py per module — Pydantic models co-located with the code that uses them
  5. PromptFactory (Abstract Factory) — one SectionBuilder per scenario, shared context template
  6. ExampleStore — auto-generated, language-appropriate, cached in examples.json
  7. SettingsManager singleton — all services use get/set, Gradio panel calls save()
  8. services/ owns all logic — api routes and Gradio panels are thin wrappers only