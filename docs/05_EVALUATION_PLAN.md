# 05 — EVALUATION PLAN (v3)

---

## METRICS SUITE — shared/metrics.py

  Primary sort key: f1_macro
  Trust most on imbalanced data: mcc (Matthews Correlation Coefficient)
  Must not fail on: recall_1 (catching all non-English) — NEVER below 0.96

  def compute_all_metrics(y_true, y_pred, y_proba=None):
      return {
          "accuracy":     accuracy_score(...),
          "precision_0":  precision_score(..., pos_label=0),
          "precision_1":  precision_score(..., pos_label=1),
          "recall_0":     recall_score(..., pos_label=0),
          "recall_1":     recall_score(..., pos_label=1),   # MAXIMIZE
          "f1_0":         f1_score(..., pos_label=0),
          "f1_1":         f1_score(..., pos_label=1),
          "f1_macro":     f1_score(..., average="macro"),
          "f1_weighted":  f1_score(..., average="weighted"),
          "mcc":          matthews_corrcoef(...),
          # if y_proba given:
          "roc_auc":      roc_auc_score(...),
          "pr_auc":       average_precision_score(...),
          "log_loss":     log_loss(...),
          "ece":          compute_ece(...),  # calibration quality
      }

  Latency (in shared/metrics.py):
    compute_latency_stats(model, texts, n=1000)
      → {p50_ms, p95_ms, p99_ms}

  Resource cost (in shared/metrics.py):
    estimate_cost(n_queries_per_day, label1_ratio, gpt4o_cost, gpt4omini_cost)
      → daily cost for a given routing strategy

---

## FALSE NEGATIVE VS FALSE POSITIVE

  False positive: English query sent to gpt-4o → costs a bit extra. Tolerable.
  False negative: Singlish query sent to gpt-4o-mini → garbled response. NEVER acceptable.

  Design target: recall(1) >= 0.97, no hard ceiling on false positives.
  Threshold tuning in Phase 6 optimizes this tradeoff.

---

## BASELINES TO BEAT

  Before comparing ML models against each other, establish these baselines:

  Baseline A — Rule engine only (no ML at all)
    Apply rule_engine() to test set.
    Queries that return None are predicted 0 (default to mini).
    Measure: f1_macro, recall_1, precision_0.
    This tells you what pure Python gets you — the floor.

  Baseline B — LLM as classifier (gpt-4o-mini asked to classify)
    Prompt: "Is the following message in non-English or does it ask about locations
    or imply location awareness? Answer only 0 or 1."
    Send every test row to gpt-4o-mini API.
    Measure: f1_macro, recall_1, latency_p99, cost_per_1k_queries.
    This tells you: how much does it cost to use an LLM for classification?
    Our trained models must match this quality at a fraction of the cost.

  Both baselines are implemented in phases/phase_5_evaluation/ablation.py.

---

## RESOURCE TRACKING PER MODEL

  For every trained model (classical ML and transformer), log:

  Metric               What it measures                  Why it matters
  ──────────────────────────────────────────────────────────────────────────
  latency_p99_ms        Worst-case inference time          Router runs on EVERY query
  train_time_s          How long training took             Planning future retrains
  model_size_mb         File size of saved model           Memory budget in deployment
  peak_ram_training_mb  Peak RAM during training           Kaggle/Colab limit planning
  peak_ram_inference_mb Peak RAM during inference batch    Production server sizing
  cost_per_1k_queries   $$ if deployed as router           Business case for this project

  All in master_comparison.csv and MLflow metrics.

---

## PHASE 5 SCRIPTS

  compare_all.py
    - Scans experiments/ for all trained models across all datasets
    - Loads MLflow runs for each experiment
    - Builds master_comparison.csv
    - Generates charts logged to MLflow:
        f1_macro bar chart (all models × all datasets)
        latency vs f1_macro scatter
        cost vs f1_macro scatter
    - Prints top-5 models by f1_macro

  ablation.py
    - Baseline A: rule_only
    - Baseline B: llm_classifier
    - Config C: rule + best classical ML model
    - Config D: rule + best transformer model
    - Comparison table: f1_macro, recall_1, latency, cost

  cost_simulation.py
    - Configurable: n_queries_per_day, label1_ratio
    - For each strategy: daily cost at OpenAI current pricing
    - Prints table + saves cost_simulation.json

  error_analysis.py
    - For best model: find all false negatives on test set
    - Group by category
    - Print 5 examples per category
    - Identify systematic failure modes

  ensemble.py
    - Soft voting: SVM + LightGBM + FastText (average probabilities)
    - Only worth pursuing if best single model has recall(1) < 0.97
    - Likely not needed — run as confirmation

---

## MASTER COMPARISON TABLE COLUMNS

  dataset_name        which dataset was used
  approach            "classical_ml" or "transformer"
  model_key           e.g. "tfidf_char__svm" or "xlmr-base"
  f1_macro
  recall_1            most important metric
  precision_0
  mcc
  roc_auc
  latency_p99_ms
  train_time_s
  model_size_mb
  peak_ram_inference_mb
  cost_per_1k_queries  (vs current all-gpt4o cost)
