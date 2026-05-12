# 06 — EXPERIMENT TRACKING (v3)
## MLflow Only — No DVC

---

## WHY MLflow REPLACES DVC FOR DATASET TRACKING

  DVC was in the previous plan for dataset versioning.
  MLflow 3 now supports mlflow.log_input() which logs:
    - Dataset name
    - Dataset digest (hash of the data — auto-computed)
    - Dataset source (file path or URL)
    - Schema (column names and types)
    - Context (training / validation / test)

  This gives you full dataset lineage without an additional tool.
  Every training run in MLflow automatically shows which dataset it used.
  You can answer "which data trained this model?" by clicking the run in MLflow UI.

  DVC adds: remote storage setup, .dvc pointer files, git hooks, dvc pull/push.
  For this project, MLflow's built-in dataset tracking is sufficient.
  We keep dependencies minimal.

---

## EXPERIMENT NAMING CONVENTION

  One MLflow experiment per phase per dataset:
    "phase1_data_generation"        ← generation pipeline runs
    "classical_ml_{dataset_name}"   ← e.g. "classical_ml_v1_baseline"
    "transformers_{dataset_name}"   ← e.g. "transformers_v2_spelling_noise"
    "phase5_evaluation"             ← ablation + comparison
    "phase6_hybrid"                 ← threshold tuning

  This keeps runs organized by dataset. The MLflow UI shows all experiments
  in a list. Filtering by experiment shows only runs for that dataset.

---

## WHAT TO LOG IN EVERY TRAINING RUN

  with mlflow.start_run(run_name=f"{vec}__{clf}"):

      # Dataset lineage
      mlflow.log_input(train_dataset, context="training")
      mlflow.log_input(val_dataset, context="validation")
      mlflow.log_param("dataset_name", dataset_name)

      # All hyperparameters
      mlflow.log_params({
          "vectorizer": vec,
          "classifier": clf,
          **vectorizer_params,
          **classifier_params,
      })

      # All metrics
      mlflow.log_metrics(compute_all_metrics(y_val, y_pred, y_proba))
      mlflow.log_metric("latency_p99_ms", latency_stats["p99"])
      mlflow.log_metric("train_time_s", train_time)
      mlflow.log_metric("model_size_mb", model_size_mb)
      mlflow.log_metric("peak_ram_mb", peak_ram_mb)

      # Artifacts
      mlflow.log_figure(confusion_matrix_fig, "confusion_matrix.png")
      mlflow.sklearn.log_model(clf, "model")
      mlflow.sklearn.log_model(vec, "vectorizer")

---

## DATASET REGISTRATION PATTERN (08_split_and_register.py)

  import mlflow.data

  train_dataset = mlflow.data.from_pandas(
      train_df,
      source=str(train_csv_path),
      name=f"{dataset_name}-train",
      targets="label"
  )
  val_dataset = mlflow.data.from_pandas(...)
  test_dataset = mlflow.data.from_pandas(...)

  # These are registered and can be referenced in any later run
  # MLflow computes a digest (hash) automatically — detects if data changed

---

## MLflow UI — WHAT YOU CAN SEE AND DO

  VIEWING:
    - All experiments listed by name
    - All runs per experiment in a sortable table
    - Compare runs side-by-side (parallel coordinates, bar charts)
    - Each run: params tab, metrics tab, artifacts tab, datasets tab
    - Datasets tab: shows train/val/test datasets used, with digest
    - Metric history: training loss per epoch (for transformers)
    - Artifacts: click to view confusion_matrix.png inline

  COMPARING:
    - Select multiple runs → Compare → parallel coordinates plot
    - Filter runs by param value (e.g. "show only SVM runs")
    - Sort by f1_macro, recall_1, latency — find best model instantly

  MODEL REGISTRY:
    - Register winning model: right-click run → Register Model
    - Name: "sprout-router-{dataset_name}"
    - Transition: None → Staging → Production (one click in UI)
    - Production model has a stable URI: models:/sprout-router-v1/Production

  WHAT MLflow UI CANNOT DO:
    - Trigger training runs (use Gradio launcher for this)
    - Edit configs
    - Schedule runs

---

## EXPORTING RESULTS ON KAGGLE (no browser UI)

  # At end of any training script — export to CSV for offline analysis
  client = mlflow.tracking.MlflowClient()
  runs   = client.search_runs(
      experiment_names=[f"classical_ml_{dataset_name}"],
      order_by=["metrics.f1_macro DESC"]
  )
  rows = [{"run": r.info.run_name, **r.data.params, **r.data.metrics} for r in runs]
  pd.DataFrame(rows).to_csv(results_path, index=False)
  print(df[["run","f1_macro","recall_1","mcc","latency_p99_ms"]].to_string())

---

## SYNCING MLFLOW RUNS BACK TO GITHUB

  mlruns/ is git-tracked. After each Kaggle session:
    git add mlruns/ results/ experiments/*/results/
    git commit -m "sync: $(date '+%Y-%m-%d')"
    git push origin main

  Model checkpoint folders (large) go to Google Drive:
    cp -r experiments/{dataset}/transformers/models/ /content/drive/MyDrive/sprout/
