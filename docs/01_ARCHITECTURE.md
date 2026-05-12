# 01 — ARCHITECTURE (v4)

---

## FULL LABEL TAXONOMY

  label=1 fires when ANY of these signals is present:

  Signal A — Sinhala/Tamil unicode (rule engine)
    if any('\u0D80' <= c <= '\u0DFF' for c in text): return 1  # Sinhala
    if any('\u0B80' <= c <= '\u0BFF' for c in text): return 1  # Tamil
    Covers all modern script inputs. Legacy non-Unicode fonts: safely ignored in 2025.

  Signal B — Sri Lankan location named (rule engine — gazetteer)
    rapidfuzz match against ~17,000 place names + alias shortforms → 1
    Catches: English names, Sinhala romanized, Tamil romanized, shortforms, typos

  Signal C — Singlish vocabulary in context (model)
    WHY model not rule engine: a vocab list fires on individual words regardless
    of sentence context. "solution" could match "sollu" if rules are too loose.
    The model sees the full sentence and decides correctly.

  Signal D — Tanglish vocabulary in context (model)
    Same reasoning as Signal C.

  Signal E — Implied location intent (model)
    "nearest branch to me", "closest outlet in my area"
    No location name present — rule engine has nothing to match.
    Model explicitly trained on these patterns. Critical signal.

  Signal F — Response language request (model)
    "answer me in sinhala", "tamil la kiyanna", "reply in tamil please"

  Signal G — Complex / multi-step task (model)
    Fraud triage, chargeback disputes, policy comparisons,
    medical eligibility, multi-condition queries.

  Signal H — Extreme ambiguity (model)
    1–3 word messages where intent is unclear → default to gpt-4o.

---

## RULE ENGINE — shared/rule_engine.py

  def rule_label(text: str, gazetteer: SLGazetteer) -> int | None:
      if any('\u0D80' <= c <= '\u0DFF' for c in text): return 1  # Sinhala unicode
      if any('\u0B80' <= c <= '\u0BFF' for c in text): return 1  # Tamil unicode
      if gazetteer.is_sl_location(text, threshold=85):  return 1  # SL location
      return None  # ML model handles the rest

  Singlish/Tanglish vocab is NOT in the rule engine.
  Vocab lists misfire on individual words without sentence context.
  The model handles these signals correctly because it sees the full sentence.

---

## GAZETTEER — shared/gazetteer.py

  Sources merged into data/processed/sl_gazetteer.csv (~17,000 rows):
    madurapa GitHub    Provinces → Districts → Cities, EN/SI/TA, postal codes
    HDX OSM LK        All villages, hamlets, name:en, name:si, name:ta
    GeoNames LK.zip   ~15,000 place names with lat/lng

  Hand-curated aliases: data/processed/sl_location_aliases.csv
    No external dataset has WhatsApp shortforms. This file is maintained manually.
    Git-tracked. Extended over time as new shortforms appear in production.

    Key entries (partial):
      col 1–15       → Colombo postal districts
      nugey          → Nugegoda
      mah'gama       → Maharagama
      fort           → Colombo Fort
      mt lav         → Mount Lavinia
      hikka          → Hikkaduwa
      batti          → Batticaloa
      trinco         → Trincomalee
      wella          → Wellawatte
      pettah         → Pettah
      bamb           → Bambalapitiya
      raja           → Rajagiriya
      battaramulla   → Battaramulla
      (80+ entries — full list in the CSV)

  class SLGazetteer:
      def __init__(self, gazetteer_csv, aliases_csv):
          # Load both, lowercase, merge into flat name set

      def is_sl_location(self, text: str, threshold: int = 85) -> bool:
          # Slide 1/2/3-word windows over text
          # rapidfuzz token_sort_ratio against all known names
          # Return True if any window scores >= threshold

  WHY python code not trained model for location detection:
    17,000+ names × multiple language variants = impossible to memorize reliably
    Fuzzy matching catches spelling errors deterministically
    Adding a new place = add one CSV row, no retraining
    < 0.1ms per query

---

## SINGLE MODEL DECISION

  One binary classifier (label 0/1).
  The model does NOT separately classify language then route.
  It learns all signals together from full sentence context.

  WHY single model:
    Simpler training, versioning, and deployment
    One inference call not two
    Research shows multitask fine-tuned models match dual-model accuracy at lower latency
    Our task is fundamentally binary — the WHY doesn't matter for routing

  ABLATION TESTED IN PHASE 5:
    Config 1: rule_only (no ML at all)
    Config 2: llm_classifier (gpt-4o-mini classifies — cost reference)
    Config 3: rule + single_binary_model (recommended)

---

## SYSTEM FLOW

  User message
       │
       ▼
  ┌──────────────────────────────────────┐
  │  RULE ENGINE  < 0.1ms               │
  │  Sinhala/Tamil unicode? → 1         │
  │  SL location name? → 1             │
  └────────────────────┬────────────────┘
                       │ (only if nothing fired)
                       ▼
  ┌──────────────────────────────────────┐
  │  ML CLASSIFIER  1–30ms              │
  │  Singlish/Tanglish context          │
  │  Implied location intent            │
  │  Response language request          │
  │  Complex task                       │
  │  Ambiguous short message            │
  └────────────────────┬────────────────┘
                       │
            confidence >= threshold?
            YES → return label
            NO  → return 1 (safe default)

---

## GRADIO LAUNCHER — launcher/app.py

  The launcher triggers scripts from the browser. It is fully dynamic.
  No list is hardcoded anywhere — everything is read from disk or config files.

  WHAT IS DYNAMIC (read from filesystem / config at runtime):
    Dataset dropdown        → scans data/datasets/ for folder names
    Classifier dropdown     → imports CLASSIFIER_REGISTRY keys from phase 3 config
    Vectorizer dropdown     → imports VECTORIZER_REGISTRY keys from phase 3 config
    Transformer dropdown    → imports TRANSFORMER_REGISTRY keys from phase 4 config
    Trained model dropdown  → scans experiments/{dataset}/classical_ml/models/
    Results dropdown        → scans results/ for existing CSV/JSON

  WHAT YOU CAN TRIGGER FROM UI:
    Phase 1 — Data:
      Individual sub-steps (select from dropdown: download, gazetteer, generate,
      evaluate, dedup, label, quality-check, split)
      OR "Run all" to run the full pipeline in sequence
      Dataset name text field (creates data/datasets/{name}/ when run)

    Phase 3 — Classical ML:
      Select dataset (from dropdown)
      Select vectorizer + classifier (from dropdowns)
      Custom params: JSON text field merged with config defaults
      Actions: Train single / Train all / HPO

    Phase 4 — Transformers:
      Select dataset (from dropdown)
      Select model (from dropdown)
      Custom params: JSON text field
      Actions: Train single / Train all / HPO / Inference + ONNX

    Phase 5 — Evaluation:
      Actions: Compare all / Ablation / Cost sim / Error analysis
      Results rendered inline after completion

    Refresh button:
      Re-scans filesystem, updates all dropdowns
      Auto-triggered after any run that creates new folders

  WHAT IS NOT IN THE UI:
    System prompts (live in 03_distilabel_generator.py, versioned in git)
    Multiple dataset generation (one dataset is the plan)

  OUTPUT DISPLAY:
    Live stdout streaming from subprocess (Popen, not run)
    After completion: link to MLflow experiment for that run
    After Phase 5: comparison table rendered inline

  HOW CUSTOM PARAMS FLOW:
    JSON text field → validated → passed as --params '{"C": 10}' to script
    Script merges with config defaults using: {**config_defaults, **json_overrides}
    MLflow logs the FINAL merged params, not just the overrides
    Every run is fully reproducible from its MLflow params tab
