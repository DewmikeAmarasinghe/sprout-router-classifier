# 02 — DATA PLAN (v4)

---

## WHAT THE TRAINING DATA TEACHES THE MODEL

The rule engine handles: Sinhala/Tamil unicode, named SL locations.
Training data teaches the model everything the rule engine CANNOT catch:

  Target A — Singlish (romanized Sinhala + English, 4 varieties)
    Variety 1: Mostly English, 1–3 Sinhala words inserted
      "Can I get a refund? Mata dunnoth awith eka wage"
    Variety 2: Half and half
      "Kohomada mata denna? Cash on delivery thiyanawada?"
    Variety 3: Mostly Sinhala words in English letters, English grammar
      "Api kohomada order karanne? Delivery eka koheda enawa?"
    Variety 4: Sinhala sentence structure, English content words
      "Mata oone eka size medium da, blue color eka"

  Target B — Tanglish (romanized Tamil + English)
    "enna price inge? discount irukka?"
    "enakku free delivery irukka?"
    "return policy enna sollu"

  Target C — Implied location intent (no named location)
    "what's the nearest branch to me"
    "closest outlet in my area please"
    "recommend a branch near my municipal area"
    "which store should I visit near me"
    These contain NO location name → rule engine fires nothing → model must catch.

  Target D — Response language request
    "answer me in sinhala please"
    "sinhala wala explain karanna"
    "tamil la reply karanna"

  Target E — Complex task requiring gpt-4o
    "If I have X plan and want to upgrade to Y during a promo, and I also have
    a pending claim, what happens to my billing?"
    "someone made a transfer I didn't authorize, what do I do?"
    Multi-branch or multi-condition eligibility queries.

  Target F — Pure English label=0
    "What are the delivery charges?"
    "Do you have this in blue?"
    "How long does shipping take?"
    "What is your return policy?"

  The model is NOT trained to detect location names — that is the gazetteer's job.

---

## WHY NO SINHALA/TAMIL SCRIPT IN TRAINING DATA

  The unicode check catches 100% of modern Sinhala/Tamil script:
    if any('\u0D80' <= c <= '\u0DFF' for c in text): return 1  # Sinhala
    if any('\u0B80' <= c <= '\u0BFF' for c in text): return 1  # Tamil

  Unicode is a universal standard. Legacy non-Unicode fonts (pre-2005) are
  essentially nonexistent in 2025 WhatsApp/chatbot inputs.

  Every training row is in English letters. Training budget goes entirely
  to the hard cases: romanized mixed-language and contextual signals.

  GROUNDING DATASET (separate, not for training):
    data/grounding/unicode_verification.csv — 200 Sinhala/Tamil script rows.
    Rule engine applied to all 200. Expected result: 100% return 1.
    This is a proof document, built once in 02_gazetteer_builder.py.

---

## DATA GENERATION FRAMEWORK: distilabel

  distilabel (Argilla) is a purpose-built LLM data pipeline.
  Handles: batching, checkpointing, retry on failure, parallel calls.

  Generator LLM: gpt-4o-mini (cost-efficient, sufficient for this task)
  Evaluator LLM: gpt-4o-mini (second pass — judges realism + label correctness)

  WHY distilabel over template loops:
    Template loops produce near-identical sentences → model memorizes structure.
    distilabel produces semantically diverse outputs → model learns real signal.
    Built-in checkpointing: safe to resume if Kaggle session dies mid-generation.
    Reproducible and shareable pipeline.

---

## GROUNDING CONTEXT FOR GENERATION

  The generator LLM is given Sprout's real client sectors as context so it
  generates realistic customer messages, not generic chatbot conversations.

  Sectors used in system prompts:
    eCommerce / fashion (Thambili Island style): product queries, sizes, cart
    Healthcare (Vision Care, dental): appointments, reports, symptoms
    Banking: transfers, fraud, lost card, account queries
    Insurance: quotes, claims, renewals, document upload
    Telecom: plan upgrades, billing, outages
    Logistics: shipment tracking, delivery issues, pickup scheduling
    Hospitality (Mount Havana style): availability, reservations, amenities
    Education (AOD style): course queries, admissions, fees

  Generation instructions:
    WhatsApp message style (casual, short, emoji optional)
    3–40 words per message
    Realistic code-mixing (not artificially forced)
    Do NOT include specific Sri Lankan location names in generated text
    DO generate implied-location patterns for Target C rows
    Vary sentence length and complexity

---

## TARGET DISTRIBUTION: ~62,000 ROWS

  Category                          Label  Target   Notes
  ──────────────────────────────────────────────────────────────────────────
  Pure English                        0    18,000   All 8 sectors, varied complexity
  Singlish light (few Sinhala words)  1    12,000   Most common real-world pattern
  Singlish heavy (mostly Sinhala)     1     7,000   Dense romanized Sinhala
  Tanglish (Tamil + English roman)    1     8,000   Northern/Eastern SL users
  Implied location intent             1     5,000   Target C — most critical signal
  Response language request           1     3,000   Target D
  Complex task                        1     5,000   Target E — multi-step reasoning
  Ambiguous / short messages          1     2,000   Target F — edge cases
  Mixed signals (multiple categories) 1     2,000   Real-world combo cases
  ──────────────────────────────────────────────────────────────────────────
  Total raw                                62,000
  After evaluator filter (~12% drop)      ~54,500
  After deduplication (~10% drop)         ~49,000
  ──────────────────────────────────────────────────────────────────────────
  Label split: ~29% label=0, ~71% label=1

  TIERED GENERATION:
    Generate in category batches. Run full pipeline after each batch.
    Track recall(1) on a held-out sample as data accumulates.
    If it plateaus before 62k: stop. distilabel checkpoints every 500 rows.
    62k is the ceiling based on task complexity, not a mandatory target.

---

## DATA GENERATION PIPELINE

  phases/phase_1_data/

  01_download_sources.py
      Downloads all external data to data/raw/. Never edits raw files.
      Sources:
        HuggingFace: Programmer-RD-AI/sinhala-english-singlish-translation
        HuggingFace: mayurasandakalum/singlish-to-sinhala-dataset
        HuggingFace: sinhala-nlp/SOLD (Sinhala twitter, closest to chatbot)
        HuggingFace: Deepakvictor/tanglish-tamil
        HuggingFace: rajpurkar/squad (English questions — label=0 vocabulary source)
        GeoNames: download.geonames.org/export/dump/LK.zip
        madurapa: github.com/madurapa/sri-lanka-provinces-districts-cities

  02_gazetteer_builder.py
      Merges all location sources → data/processed/sl_gazetteer.csv (~17k rows)
      Loads data/processed/sl_location_aliases.csv (hand-curated shortforms)
      Builds shared/gazetteer.py (SLGazetteer class)
      Generates data/grounding/unicode_verification.csv (200 script rows)
      Verifies rule engine catches all 200 — prints verification result

  03_distilabel_generator.py
      Args: --dataset-name  (creates data/datasets/{name}/)
            --category      (one category or "all", default: all)
            --n-rows        (target rows for that category, default: per config)
            --batch-size    (rows per LLM call, default: 50)
      Checkpoints every 500 rows → safe to resume
      Saves: data/datasets/{name}/raw/generated_raw.csv
      Logs generation run to MLflow: experiment "phase1_data_generation"
        params: dataset_name, category, n_rows, model_used

  04_distilabel_evaluator.py
      Args: --dataset-name
      Second LLM pass: scores realism (1–5), label_correct (bool), has_signal (bool)
      Drops: realism < 3 OR label_correct == False
      Saves: data/datasets/{name}/raw/generated_evaluated.csv
      Prints per-category quality report: Total | Dropped | Drop% | Avg_realism
      Logs drop rates to MLflow

  05_deduplication.py
      Args: --dataset-name  --threshold (default: 0.88)  --review-mode
      Embeds all rows: paraphrase-multilingual-MiniLM-L12-v2
      Stores in Milvus Lite: data/datasets/{name}/dedup.db
      Marks near-duplicates (cosine similarity > threshold)
      Writes: data/datasets/{name}/raw/dedup_review.csv
        columns: text_a, text_b, similarity_score, decision
      If --review-mode: prints top-20 closest pairs and exits (no dedup performed)
      Saves: data/datasets/{name}/raw/generated_deduped.csv

      WHY Milvus Lite over FAISS:
        Single .db file — portable between Kaggle, Colab, local.
        Persistent — requery without re-embedding.
        No manual index serialization needed.

      ON DEDUP FALSE POSITIVES:
        At threshold=0.88, false positives are rare for varied-length multilingual text.
        dedup_review.csv is there for transparency. Inspect with --review-mode
        if you want to verify a batch before committing to the dedup.

  06_rule_labeler.py
      Args: --dataset-name
      Applies rule_engine() to every row
      Compares auto-label vs generated label → prints agreement % per category
      Target agreement: >= 97%
      Flags disagreement rows for review
      Saves: data/datasets/{name}/raw/generated_labeled.csv

  07_data_quality_eval.py
      Args: --dataset-name
      Steps:
        1. Rule consistency check (agreement %)
        2. Distribution check: actual vs expected label %
        3. Human spot-check: prints 20 random samples per category
        4. Sanity model: TF-IDF char + SVM on 80% split
           (val error > 12% per category = generation problem — regenerate that category)
      Saves: data/datasets/{name}/quality_report.json

  08_split_and_register.py
      Args: --dataset-name
      Stratified 80/10/10 split by label AND category
      Saves train.csv, val.csv, test.csv to data/datasets/{name}/
      Registers in MLflow via mlflow.log_input():
        train_ds = mlflow.data.from_pandas(train_df, source=path, name=f"{name}-train")
        mlflow.log_input(train_ds, context="training")
      Prints final stats: rows per split, label distribution
      NEVER touch test.csv until Phase 5 final evaluation

---

## GAZETTEER: LOCATION ALIAS FILE

  data/processed/sl_location_aliases.csv is hand-curated.
  No external dataset has WhatsApp-style Sri Lankan location shortforms.
  This file is git-tracked and extended over time as new shortforms appear.

  Format: alias, canonical_name
  Examples (partial — full list in 01_ARCHITECTURE.md):
    col 7, Colombo 7
    nugey, Nugegoda
    mah'gama, Maharagama
    fort, Colombo Fort
    mt lav, Mount Lavinia
    hikka, Hikkaduwa
    batti, Batticaloa
    trinco, Trincomalee
    ... (80+ entries covering all major urban areas and tourist destinations)

---

## DATASET VERSIONING VIA MLflow

  mlflow.log_input() records dataset name, digest (hash), source, schema.
  Every training run in MLflow shows which dataset it used.
  This gives full lineage: model → training run → dataset name → CSV files.
  No DVC required.

  If a v2 is ever needed (production feedback, new failure patterns):
    Run Phase 1 again with --dataset-name v2_production_fix
    A new folder is created. All experiments, models, and results
    for v2 are isolated in experiments/v2_production_fix/.
    compare_all.py discovers both and compares automatically.
    Zero refactoring required.
