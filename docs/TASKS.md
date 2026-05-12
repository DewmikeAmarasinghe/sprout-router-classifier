# TASKS.md (v4) — Implementation Checklist

This is the working reference for building sprout-router-classifier.
Read 00_MASTER_README.md first. Then follow this file phase by phase.
Each block explains WHAT to build, WHY it is designed this way,
and HOW it connects to the rest of the system.

---

## BEFORE ANYTHING ELSE

### T-0: Project setup
  [ ] Run setup_project.py → creates all folders and __init__.py stubs
  [ ] pip install -r requirements.txt
  [ ] Confirm shared/config.py auto-detects environment correctly
      (print BASE_DIR, DATA_DIR — verify paths)
  [ ] Local only: run `mlflow ui` to confirm MLflow is working

---

## PHASE 1 — DATA

### T-1.1: Download sources (01_download_sources.py)
  Goal: pull all external data to data/raw/. Never edit raw files.

  Download:
    HuggingFace (datasets library):
      Programmer-RD-AI/sinhala-english-singlish-translation → data/raw/hf_singlish/
      mayurasandakalum/singlish-to-sinhala-dataset           → data/raw/hf_singlish/
      sinhala-nlp/SOLD  (Sinhala Twitter — closest to chatbot) → data/raw/hf_sold/
      Deepakvictor/tanglish-tamil                            → data/raw/hf_tanglish/
      rajpurkar/squad   (English questions — vocab reference) → data/raw/hf_squad/
    GeoNames: download.geonames.org/export/dump/LK.zip       → data/raw/geonames_lk/
    madurapa: raw GitHub CSV URL                             → data/raw/madurapa/

  These are RAW MATERIAL for:
    a) Building the location gazetteer
    b) Providing vocabulary context for generation LLM prompts
  They are NOT directly used as training rows.

### T-1.2: Build gazetteer (02_gazetteer_builder.py)
  Goal: build shared/gazetteer.py and data/processed/sl_gazetteer.csv

  Steps:
    Load GeoNames LK: extract name, name:en, name_si, name_ta
    Load madurapa CSV: EN/SI/TA city names
    Merge, lowercase, strip, deduplicate → ~17k unique strings
    Load data/processed/sl_location_aliases.csv (hand-curated shortforms)
    Save: data/processed/sl_gazetteer.csv
    Build shared/gazetteer.py SLGazetteer class

  Also generate grounding dataset:
    Pull 200 actual Sinhala/Tamil script sentences from hf_sold + hf_tanglish
    Save: data/grounding/unicode_verification.csv
    Apply rule_engine() to all 200
    Assert: 100% return 1
    Print: "Unicode verification passed: 200/200"
    This is a one-time proof document — not training data

  WHY: Gazetteer is used by rule_engine() at inference time. Built once.
  sl_location_aliases.csv is maintained manually in git — add new shortforms
  whenever they appear in production logs.

### T-1.3: Generate data (03_distilabel_generator.py)
  Goal: generate ~62,000 diverse WhatsApp-style messages

  CLI args:
    --dataset-name   required. creates data/datasets/{name}/
    --category       optional. one of the 9 categories, or "all" (default)
    --n-rows         target rows for that category (default: per CATEGORY_CONFIG)
    --batch-size     rows per LLM call (default: 50)

  No --system-prompt argument. System prompts live in this file, versioned in git.
  To change prompts: edit the file, commit, re-run. Clean git diff shows what changed.

  CATEGORY_CONFIG (defined in this file — not imported from elsewhere):
    english:            {"target": 18000, "label": 0}
    singlish_light:     {"target": 12000, "label": 1}
    singlish_heavy:     {"target":  7000, "label": 1}
    tanglish:           {"target":  8000, "label": 1}
    implied_location:   {"target":  5000, "label": 1}
    response_language:  {"target":  3000, "label": 1}
    complex_task:       {"target":  5000, "label": 1}
    ambiguous_short:    {"target":  2000, "label": 1}
    mixed_signals:      {"target":  2000, "label": 1}

  TIERED GENERATION APPROACH:
    Generate categories in order of difficulty: english first, then singlish_light,
    then others. After each batch, check recall(1) on a 500-row held-out sample.
    If recall(1) stops improving across 2 consecutive categories: stop generating.
    distilabel checkpoints every 500 rows so partial work is never lost.

  What to implement:
    distilabel Pipeline with TextGeneration step
    Generator LLM: OpenAILLM (gpt-4o-mini)
    System prompt per category: WhatsApp style, Sprout sector context,
      category-specific instructions, output JSON schema
    Batch rows in parallel, checkpoint every 500 rows
    On resume: load checkpoint, skip already-generated rows
    Output: data/datasets/{name}/raw/generated_raw.csv
    Log to MLflow experiment "phase1_data_generation":
      params: dataset_name, category, n_rows, batch_size, model_used, timestamp

  SECTOR CONTEXT FOR PROMPTS (use all 8):
    eCommerce/fashion, Healthcare, Banking, Insurance,
    Telecom, Logistics, Hospitality, Education
    Include specific realistic examples from each sector per category.

  CRITICAL — what prompts must NOT do:
    Must NOT include specific Sri Lankan location names in generated text
    Must NOT include Sinhala/Tamil script
  CRITICAL — what prompts MUST do for implied_location category:
    Must generate sentences with implied proximity/location need but NO place name
    Examples: "nearest branch to me", "store near my area", "closest outlet"

### T-1.4: Evaluate generated data (04_distilabel_evaluator.py)
  Goal: filter unrealistic or mislabeled rows

  CLI args: --dataset-name

  Second distilabel Pipeline: evaluator LLM scores each row
  Evaluation criteria: realism (1–5), label_correct (bool), has_signal (bool)
  Drop if: realism < 3 OR label_correct == False
  Save: data/datasets/{name}/raw/generated_evaluated.csv
  Print per-category report:
    Category | Total | Dropped | Drop% | Avg_realism
  Log drop rates to MLflow
  If any category has drop rate > 25%: warn. Prompt for that category needs fixing.

### T-1.5: Deduplicate (05_deduplication.py)
  Goal: remove near-duplicate rows

  CLI args: --dataset-name  --threshold (default: 0.88)  --review-mode

  Embed all rows: sentence-transformers paraphrase-multilingual-MiniLM-L12-v2
  Init Milvus Lite: data/datasets/{name}/dedup.db
  Insert all embeddings with row index as ID
  For each row: query top-2; if similarity > threshold → mark as duplicate
  Write: data/datasets/{name}/raw/dedup_review.csv
    columns: text_a, text_b, similarity_score, decision
  If --review-mode: print top-20 closest pairs, exit without deleting anything
  Remove duplicates
  Save: data/datasets/{name}/raw/generated_deduped.csv
  Print: rows removed, % per category

  WHY Milvus Lite not FAISS:
    Single .db file — portable between Kaggle, Colab, local
    Persistent — requery without re-embedding
    No manual index serialization

  ON FALSE POSITIVES:
    At 0.88 threshold with varied-length multilingual text: false positives are rare.
    Use --review-mode to inspect top-20 pairs if concerned. Not a required step.

### T-1.6: Label (06_rule_labeler.py)
  Goal: auto-label with rule_engine(), check agreement

  CLI args: --dataset-name

  Apply rule_label(text, gazetteer) to every row
  Compare auto-label vs distilabel-generated label
  Target agreement per category: >= 97%
  Flag disagreement rows for review (print examples)
  Save: data/datasets/{name}/raw/generated_labeled.csv

### T-1.7: Quality check (07_data_quality_eval.py)
  Goal: validate dataset is ready for training

  CLI args: --dataset-name

  Steps:
    1. Rule agreement check (from generated_labeled.csv)
    2. Distribution check: actual label % vs expected (29/71)
    3. Human spot-check: print 20 random rows per category
    4. Sanity model: TF-IDF char + SVM on 80%/20% split of labeled data
       If val error > 12% per category → warn, consider regenerating that category
  Save: data/datasets/{name}/quality_report.json
  Log quality metrics to MLflow

### T-1.8: Split and register (08_split_and_register.py)
  Goal: create train/val/test splits, register in MLflow

  CLI args: --dataset-name

  Stratified 80/10/10 split by BOTH label AND category
  Save: data/datasets/{name}/train.csv, val.csv, test.csv
  Register in MLflow:
    train_ds = mlflow.data.from_pandas(train_df, source=path, name=f"{name}-train")
    mlflow.log_input(train_ds, context="training")
  Print: rows per split, label distribution per split
  *** DO NOT TOUCH test.csv UNTIL PHASE 5 HPO FINAL EVAL ***

---

## SHARED MODULES — BUILD THESE FIRST (before Phase 3)

  shared/config.py
    auto-detect IS_KAGGLE, IS_COLAB, IS_LOCAL
    BASE_DIR, DATA_DIR, EXPERIMENTS, RESULTS_DIR, MLFLOW_URI
    get_dataset_path(name) → data/datasets/{name}/
    get_experiment_path(name, approach) → experiments/{name}/{approach}/
    discover_datasets() → list folder names in data/datasets/

  shared/metrics.py
    compute_all_metrics(y_true, y_pred, y_proba=None) → dict with all metrics
    compute_latency_stats(model_fn, texts, n=1000) → {p50, p95, p99} in ms
    get_peak_ram_mb() → peak RAM during last tracemalloc context
    get_model_size_mb(obj) → pickle to bytes, return MB
    estimate_cost(n_queries_day, label1_ratio, cost_4o, cost_mini) → float
    print_metrics_table(metrics_dict, label)

  shared/rule_engine.py
    rule_label(text, gazetteer) → 1 or None
    Unicode checks only. No vocab lists.

  shared/gazetteer.py
    SLGazetteer(gazetteer_csv, aliases_csv)
    is_sl_location(text, threshold=85) → bool
    Uses rapidfuzz token_sort_ratio over 1/2/3-gram windows

---

## PHASE 2 — EDA

### T-2.1: EDA (09_eda.py)
  CLI args: --dataset-name

  Generate and log to MLflow as artifacts:
    Label distribution bar chart
    Category distribution bar chart
    Token length histogram per category
    Top-50 unigrams per class (label=0 vs label=1)
    Sample messages per category (10 per category, logged as text artifact)

---

## PHASE 3 — CLASSICAL ML

### T-3.1: Config (config.py)
  VECTORIZER_REGISTRY, CLASSIFIER_REGISTRY, HPO_SEARCH_SPACES
  See 03_CLASSICAL_ML_PLAN.md for full registry contents.
  This is the ONLY file to edit when adding a new model or vectorizer.

### T-3.2: vectorizers.py and classifiers.py
  Implement all classes. Expose build_vectorizer(name, params) and
  build_classifier(name, params) factory functions.
  All vectorizers: fit(X), transform(X) interface.
  All classifiers: fit(X,y), predict(X), predict_proba(X) interface.

### T-3.3: train_single.py
  CLI args: --dataset-name --vec X --clf Y [--params '{"C": 10}']

  Flow:
    Load train.csv and val.csv from get_dataset_path(dataset_name)
    Build vectorizer from registry, merge --params overrides
    Build classifier from registry, merge --params overrides
    Track: time, tracemalloc
    Fit vectorizer on train texts, transform train + val
    Fit classifier
    Evaluate: compute_all_metrics on val
    Measure: latency p50/p95/p99, peak RAM, model size
    Log to MLflow experiment "classical_ml_{dataset_name}":
      run_name: "{vec}__{clf}"
      params: all vectorizer + classifier params + dataset_name
      metrics: all from compute_all_metrics + latency + ram + size
      input: mlflow.log_input(train_dataset)
      artifact: confusion_matrix.png
    Save: experiments/{dataset_name}/classical_ml/models/{vec}__{clf}.pkl
    Save: experiments/{dataset_name}/classical_ml/models/{vec}__{clf}_vectorizer.pkl
    Print metric table

### T-3.4: train_all.py
  CLI args: --dataset-name
  Loops VECTORIZER_REGISTRY × CLASSIFIER_REGISTRY
  Calls train_single logic for each combination
  FastText runs as standalone at end
  Exports MLflow runs to experiments/{dataset_name}/classical_ml/results/runs.csv

### T-3.5: hpo.py
  CLI args: --dataset-name --vec X --clf Y --n-trials N
  Optuna study, each trial = nested MLflow run
  Search spaces from HPO_SEARCH_SPACES in config.py
  After HPO:
    Save best params → experiments/{dataset_name}/classical_ml/results/best_params.json
    Retrain best config on train+val
    Evaluate ONCE on test.csv (first test access for this model)
    Log final test metrics with tag "hpo_final_test"

### T-3.6: Resource tracking — apply to every training run
  Training time:     time.perf_counter() around .fit()
  Inference latency: compute_latency_stats() on 1000 val samples
  Peak RAM training: tracemalloc context around .fit()
  Peak RAM inference: tracemalloc context around batch predict
  Model size:        get_model_size_mb()
  All logged to MLflow as metrics, included in runs.csv

---

## PHASE 4 — TRANSFORMERS

### T-4.1: Config (config.py)
  TRANSFORMER_REGISTRY, TRAIN_CONFIG, HPO_SEARCH_SPACE
  See 04_TRANSFORMERS_PLAN.md for full contents.
  Adding a model: one line in TRANSFORMER_REGISTRY.

### T-4.2: dataset.py
  load_and_tokenize(train_df, val_df, test_df, model_name, max_length)
  Returns HuggingFace Dataset objects ready for Trainer

### T-4.3: train_single.py
  CLI args: --dataset-name --model X [--params '{"learning_rate": 3e-5}']
  Same pattern as classical ML but uses HuggingFace Trainer
  MLflow experiment: "transformers_{dataset_name}"
  Checkpoint saved to: experiments/{dataset_name}/transformers/models/{model_key}/
  Resource tracking: same as classical ML (latency, RAM, size, train_time)

### T-4.4: train_all.py, hpo.py
  Same structure as Phase 3 equivalents
  hpo.py uses MedianPruner — kills underperforming trials after epoch 1

### T-4.5: inference.py
  CLI args: --dataset-name --model X [--export-onnx]
  Latency benchmark: 1000 predictions, p50/p95/p99
  ONNX export if flag set:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    2–4× faster on CPU, < 1% accuracy drop
    Saves to: experiments/{dataset_name}/transformers/models/{model_key}_onnx/

---

## PHASE 5 — EVALUATION

### T-5.1: compare_all.py
  Scans experiments/ automatically — no hardcoded paths
  Reads all MLflow runs
  Builds results/master_comparison.csv with columns:
    dataset_name, approach, model_key, vec, clf,
    f1_macro, recall_1, precision_0, mcc, roc_auc,
    latency_p99_ms, train_time_s, model_size_mb,
    peak_ram_inference_mb
  Generates and logs charts to MLflow:
    f1_macro bar chart (all models)
    latency vs f1_macro scatter
  Renders table in Gradio evaluation panel

### T-5.2: ablation.py
  Tests on the SAME test set:
    rule_only: apply rule_label(), unmatched rows → predict 0
    llm_classifier: send each test row to gpt-4o-mini API for classification
      Record: f1_macro, recall_1, latency_p99, cost_per_1k
    rule+best_model: hybrid router with best trained model from compare_all
  Save: results/ablation_results.csv

### T-5.3: cost_simulation.py
  Configurable: n_queries_per_day (default: 100,000), label1_ratio
  Strategies compared: all-gpt-4o / rule-only / rule+classical / rule+transformer / llm-classifier
  Uses OpenAI current API pricing (configurable constant at top of file)
  Save: results/cost_simulation.json

### T-5.4: error_analysis.py
  For best model: find all false negatives on test set
  Group by category, print 5 examples per category
  Identify failure patterns

---

## PHASE 6 — HYBRID ROUTER

### T-6.1: threshold_tuning.py
  Sweep threshold on val set (0.3 to 0.9 in steps of 0.05)
  Target: recall(1) >= 0.97 with highest possible precision(0)
  Save: experiments/{dataset_name}/hybrid_threshold.json

### T-6.2: hybrid_router.py
  Loads gazetteer, rule_engine, best trained model
  .predict(text: str) → int  (0 or 1)
  .predict_batch(texts: list) → list
  Logs latency per call
  This is the final production artifact

---

## GRADIO LAUNCHER — IMPLEMENTATION NOTES

  launcher/app.py:
    On startup: ui_utils.refresh_all() → populate all dropdowns
    subprocess.Popen (not .run) → enables live output streaming
    After run completes: trigger dropdown refresh automatically
    Dataset name field: text input, validated (alphanumeric + underscore only)

  launcher/ui_utils.py:
    refresh_datasets()          → discover_datasets() from shared/config.py
    refresh_classifiers()       → importlib load phase 3 config, return REGISTRY keys
    refresh_vectorizers()       → same
    refresh_transformers()      → importlib load phase 4 config, return REGISTRY keys
    refresh_trained_models(ds, approach) → scan experiments/{ds}/{approach}/models/
    validate_json_params(s)     → try json.loads, return dict or raise with message

  launcher/panels/panel_data.py:
    Sub-step dropdown: all 8 phase 1 scripts + "Run all"
    Dataset name: text input (default: "v1_baseline")
    Category dropdown: all 9 categories + "all"
    N-rows: number input
    Batch size: number input
    Run button + output textbox

  launcher/panels/panel_classical.py:
    Dataset dropdown (dynamic)
    Vectorizer dropdown (dynamic from REGISTRY)
    Classifier dropdown (dynamic from REGISTRY)
    Params: JSON text input (optional, validated)
    Action: radio (Train single / Train all / HPO)
    N-trials: number input (shown only when HPO selected)
    Run button + output textbox

  launcher/panels/panel_transformer.py:
    Dataset dropdown (dynamic)
    Model dropdown (dynamic from REGISTRY)
    Params: JSON text input (optional, validated)
    Action: radio (Train single / Train all / HPO / Inference+ONNX)
    N-trials: number input (shown only when HPO selected)
    Run button + output textbox

  launcher/panels/panel_evaluation.py:
    Action: radio (Compare all / Ablation / Cost sim / Error analysis)
    Run button + output textbox + inline results table

---

## MLFLOW CONVENTIONS — FOLLOW EVERYWHERE

  Experiment naming: "{phase}_{dataset_name}"
    classical_ml_v1_baseline
    transformers_v1_baseline
    phase1_data_generation
    phase5_evaluation
    phase6_hybrid

  Always log in every training run:
    mlflow.log_param("dataset_name", dataset_name)
    mlflow.log_input(train_dataset, context="training")
    mlflow.log_metrics(compute_all_metrics(...))
    mlflow.log_metric("latency_p99_ms", ...)
    mlflow.log_metric("train_time_s", ...)
    mlflow.log_metric("model_size_mb", ...)
    mlflow.log_metric("peak_ram_inference_mb", ...)
    mlflow.log_figure(confusion_fig, "confusion_matrix.png")

  Model registry (after Phase 5 winner is decided):
    mlflow.register_model(model_uri, name="sprout-router-v1")
    Transition: None → Staging → Production in MLflow UI
