# 02 — DATA PLAN (v6)

---

## OVERVIEW

  Generation target:  ~60,000 raw rows
  After quality filter (~12% drop): ~52,800
  After exact dedup (~2% drop):     ~51,700
  Training target: ~50,000 clean rows

  SEPARATE GROUNDING DATASET (not counted in 60k, not used for training):
    data/grounding/unicode_verification.csv
    500 pure Sinhala + 500 pure Tamil = 1,000 rows
    Proves is_pure_script() catches 100% of script inputs

---

## 3-LEVEL HIERARCHY

  Level 1 — LanguageKey (5 values)
    pure_english, singlish_light, singlish_heavy, tanglish_light, tanglish_heavy

  Level 2 — IndustryKey (8 values)
    ecommerce, healthcare, banking, insurance, telecom, logistics, hospitality, education

  Level 3 — ScenarioKey (9 values)
    simple_transactional    label=0 if English, label=1 if Singlish/Tanglish
    named_location          label=0 if English, label=1 if Singlish/Tanglish
    location_proximity      always label=1
    location_relative       always label=1
    complex_task            always label=1
    sensitive_context       always label=1
    escalation              always label=1
    response_language       always label=1
    continuation            always label=1

  Total unique cells: 5 × 8 × 9 = 360 cells
  Average rows per cell: 60,000 / 360 ≈ 167 rows

---

## LABEL LOGIC

  from backend.config.keys import resolve_label
  label = resolve_label(language, scenario)

  Returns 0 only when:
    language == LanguageKey.PURE_ENGLISH
    AND scenario in {SIMPLE_TRANSACTIONAL, NAMED_LOCATION}

  Returns 1 in all other cases.

---

## SCENARIO DEFINITIONS (9 scenarios)

  simple_transactional
    Routine query: pricing, availability, hours, order status.
    Pure English → label=0. Singlish/Tanglish → label=1.

  named_location
    Mentions a named SL location for a simple lookup (hours, address, contact).
    Does NOT require distance reasoning.
    Pure English → label=0. Singlish/Tanglish → label=1.
    CONTRAST: "From Kandy which branch?" → location_relative (needs spatial reasoning)

  location_proximity
    User wants nearest/closest location. Named places MAY appear.
    Signal is proximity INTENT, not presence of a location name.
    Always label=1.

  location_relative
    Spatial relationship between named places OR landmarks.
    Reference points are NOT limited to branch names:
    schools, junctions, roads, shopping centers, any local landmark.
    "near Dharmapala Vidyalaya", "100m from the Cargills", "past the Galle road junction"
    Always label=1.

  complex_task
    Multi-step, multi-condition: policy comparison, upgrade + pending claim,
    eligibility with several variables.
    Always label=1.

  sensitive_context
    Fraud, unauthorized transactions, medical symptoms, financial distress,
    account compromise. Even simple English needs gpt-4o for tone.
    Always label=1.

  escalation
    Frustration, complaints, demands for manager, threats to leave.
    Always label=1.

  response_language
    User asks chatbot to reply in Sinhala, Tamil, or another non-English language.
    Always label=1.

  continuation
    Merged from agentic_retry + ambiguous_short:
    (A) ~55%: previous chatbot action FAILED — "it still shows error", "tried again same problem"
    (B) ~45%: intent unclear OR user asks for re-explanation — "puriyala", "I didn't get that"
    Both signals need gpt-4o to reason carefully before responding.
    Always label=1.

---

## DISTRIBUTION FRACTIONS

  All fractions defined in backend/config/distribution.py.
  Three-level validation: language fractions, industry fractions, scenario fractions
  all must each sum to 1.0 — Pydantic validates at import time.

  Language fractions (approximate):
    pure_english:   28%  (~16,800 rows)
    singlish_light: 21%  (~12,600 rows)
    singlish_heavy: 13%  ( ~7,800 rows)
    tanglish_light: 19%  (~11,400 rows)
    tanglish_heavy: 19%  (~11,400 rows)

  Label split estimate:
    label=0: ~24% (~14,400 rows — pure English + simple_transactional/named_location)
    label=1: ~76% (~45,600 rows)

---

## GENERATION STRATEGY

  BATCH SIZE: 50 prompts per API call
    Sweet spot: quality and diversity both degrade beyond 80 per call.
    50 fits comfortably in context with rolling context included.

  ROLLING CONTEXT: 50 RANDOM SAMPLES from ALL previously generated rows in the current cell
    Not the last 25 — random sampling from the full cell history gives full coverage.
    Passed to each API call as "DO NOT repeat or closely paraphrase these".
    Prevents repetition across the entire cell's generation history.
    Cost: ~2,500 tokens per call for 50 context sentences — manageable.

  EXACT DEDUP: hash set of generated texts
    Free. Catches literal duplicate strings.
    Applied during generation (seen_texts: set[str]) — duplicates are skipped immediately.

  NO SEMANTIC DEDUP:
    Rolling context handles within-cell repetition (~98% of near-duplicates).
    Cross-cell contamination is not an issue (different scenarios → structurally different).
    Embedding 60k rows: 10+ minutes compute, extra dependency, marginal gain.
    Removed from pipeline.

  QUALITY FILTER: LLM-as-judge (EvaluatorService)
    25 rows per evaluation call.
    Judge scores: realism (1–5), label_correct (bool), has_signal (bool).
    Drop if realism < EVAL_MIN_REALISM OR label_correct == False.
    Expected drop rate: ~12%.

---

## EXAMPLES SYSTEM (ExampleStore)

  backend/generation/example_store.py + examples.json

  WHAT IT DOES:
    Generates language-appropriate examples per (language, industry, scenario) cell.
    Singlish examples contain romanized Sinhala words.
    Tanglish examples contain romanized Tamil words.
    Pure English examples contain no code-mixing.

  HOW IT WORKS:
    First call to example_store.get(language, industry, scenario):
      → generates 8 examples via API
      → saves to examples.json
      → returns examples

    Subsequent calls:
      → loads from examples.json
      → returns cached examples (no API call)

    force_regenerate=True:
      → generates fresh examples
      → overwrites cache

  GRADIO INTEGRATION:
    Generation tab shows examples for any selected cell.
    "Regenerate" button → force_regenerate=True.
    Editable text field → "Save" → example_store.update(...).

  ANTI-EXAMPLES:
    Each ScenarioConfig.anti_scenario_keys defines which scenarios
    are used as contrast examples in the system prompt.
    example_store.get_anti_examples(scenario, language, industry) pulls
    2 examples from each anti-scenario's cached examples.

---

## PROMPT FACTORY

  backend/generation/prompt_factory.py

  Pattern: Abstract Factory + Template Method

  SharedContextBuilder:
    SPROUT_CONTEXT — same across all calls
    Industry + language context
    Platform style (rotated across 8 styles per call)

  SectionBuilder per scenario (9 subclasses):
    StandardSectionBuilder — used for most scenarios
    LocationRelativeSectionBuilder — adds landmark instruction
    ContinuationSectionBuilder — specifies 55/45 sub-type mix

  Adding a new scenario:
    1. Add key to ScenarioKey in config/keys.py
    2. Add ScenarioConfig to config/scenario_configs.py
    3. Add SectionBuilder subclass in generation/prompt_factory.py (or reuse Standard)
    4. Register in PromptFactory.SECTION_BUILDERS
    5. Add ScenarioBucket entries in config/distribution.py

---

## PIPELINE SCRIPTS

  backend/generation/pipeline.py — DataPipeline.run_all() / run_step()
  phases/phase_1_grounding.py    — generates grounding dataset (run once)
  phases/phase_2_generate.py     — GeneratorService.run()
  phases/phase_3_evaluate.py     — EvaluatorService.run()
  phases/phase_4_split.py        — DataSplitter.run()

  CHECKPOINT:
    Generator checkpoints every CHECKPOINT_EVERY rows to data/datasets/v1/raw/checkpoint.csv.
    Run with --resume to continue from checkpoint.

  TEST SET POLICY:
    data/datasets/v1/test.csv is accessed ONCE — in final evaluation only.
    All development uses train.csv and val.csv.

---

## DATASET VERSIONING

  DATASET_VERSION in config/settings.py controls active folder.
  v1 is the default. To create v2:
    - Change DATASET_VERSION to "v2" in settings (or via Gradio Config tab)
    - Click "Initialize v2" → creates data/datasets/v2/ and experiments/v2/
    - All scripts auto-use the new version
    - compare_all.py discovers both versions automatically
    - No refactoring needed