# 07 — FILE TREE (v4)

```
sprout-router-classifier/
│
├── 00_MASTER_README.md
├── 01_ARCHITECTURE.md
├── 02_DATA_PLAN.md
├── 03_CLASSICAL_ML_PLAN.md
├── 04_TRANSFORMERS_PLAN.md
├── 05_EVALUATION_PLAN.md
├── 06_EXPERIMENT_TRACKING.md
├── 07_FILE_TREE.md
├── 08_WORKFLOW.md
├── TASKS.md
│
├── requirements.txt              ← all pip deps
├── setup_project.py              ← run once: creates all folders + __init__.py stubs
├── sync.sh                       ← git push at end of Kaggle session
│
│
├── shared/                       ← imported by every phase, never duplicated
│   ├── __init__.py
│   ├── config.py                 ← path resolver: auto-detects Kaggle/Colab/local
│   │                                get_dataset_path(name), get_experiment_path(name, approach)
│   │                                discover_datasets() → scans data/datasets/
│   ├── metrics.py                ← compute_all_metrics(), compute_latency_stats(),
│   │                                estimate_cost(), print_metrics_table()
│   ├── rule_engine.py            ← rule_label(text, gazetteer) → 1 or None
│   │                                unicode detection only — no vocab lists
│   └── gazetteer.py              ← SLGazetteer(gazetteer_csv, aliases_csv)
│                                    is_sl_location(text, threshold=85) → bool
│
│
├── launcher/                     ← Gradio UI: triggers all phases from browser
│   ├── app.py                    ← entry point: python launcher/app.py
│   ├── ui_utils.py               ← filesystem scanner for all dynamic dropdowns
│   │                                refresh_datasets(), refresh_classifiers(),
│   │                                refresh_vectorizers(), refresh_transformers()
│   └── panels/
│       ├── panel_data.py         ← Phase 1 controls (sub-step selector, dataset name)
│       ├── panel_classical.py    ← Phase 3 controls (vec/clf dropdowns, param override)
│       ├── panel_transformer.py  ← Phase 4 controls (model dropdown, param override)
│       └── panel_evaluation.py   ← Phase 5 controls (compare, ablation, cost sim)
│
│
├── phases/
│   │
│   ├── phase_1_data/
│   │   ├── __init__.py
│   │   ├── 01_download_sources.py       ← HuggingFace + GeoNames + madurapa
│   │   ├── 02_gazetteer_builder.py      ← sl_gazetteer.csv + grounding verification
│   │   ├── 03_distilabel_generator.py   ← args: --dataset-name --category --n-rows
│   │   ├── 04_distilabel_evaluator.py   ← args: --dataset-name
│   │   ├── 05_deduplication.py          ← args: --dataset-name --threshold --review-mode
│   │   ├── 06_rule_labeler.py           ← args: --dataset-name
│   │   ├── 07_data_quality_eval.py      ← args: --dataset-name
│   │   └── 08_split_and_register.py     ← args: --dataset-name (MLflow log_input)
│   │
│   ├── phase_2_eda/
│   │   ├── __init__.py
│   │   └── 09_eda.py                    ← args: --dataset-name
│   │                                       logs all plots to MLflow as artifacts
│   │
│   ├── phase_3_classical_ml/
│   │   ├── __init__.py
│   │   ├── config.py                    ← VECTORIZER_REGISTRY, CLASSIFIER_REGISTRY,
│   │   │                                   HPO_SEARCH_SPACES — only file to edit for new models
│   │   ├── vectorizers.py               ← TF-IDF char/word, Word2Vec, spaCy, Combined
│   │   ├── classifiers.py               ← LogReg, SVM, LGBM, XGBoost, CatBoost, RF, FastText
│   │   ├── train_all.py                 ← args: --dataset-name
│   │   │                                   full grid: all vecs × all clfs
│   │   ├── train_single.py              ← args: --dataset-name --vec X --clf Y [--params '{}']
│   │   └── hpo.py                       ← args: --dataset-name --vec X --clf Y --n-trials N
│   │
│   ├── phase_4_transformers/
│   │   ├── __init__.py
│   │   ├── config.py                    ← TRANSFORMER_REGISTRY, TRAIN_CONFIG, HPO_SEARCH_SPACE
│   │   ├── dataset.py                   ← CSV → HuggingFace Dataset + tokenizer
│   │   ├── train_all.py                 ← args: --dataset-name
│   │   ├── train_single.py              ← args: --dataset-name --model X [--params '{}']
│   │   ├── hpo.py                       ← args: --dataset-name --model X --n-trials N
│   │   └── inference.py                 ← args: --dataset-name --model X [--export-onnx]
│   │
│   ├── phase_5_evaluation/
│   │   ├── __init__.py
│   │   ├── compare_all.py               ← scans experiments/, builds master_comparison.csv
│   │   ├── ablation.py                  ← rule-only vs llm-classifier vs trained model
│   │   ├── latency_benchmark.py         ← p50/p95/p99 per saved model
│   │   ├── cost_simulation.py           ← $/day per routing strategy
│   │   ├── error_analysis.py            ← false negatives by category
│   │   └── ensemble.py                  ← soft voting experiments (if needed)
│   │
│   └── phase_6_hybrid/
│       ├── __init__.py
│       ├── hybrid_router.py             ← rule engine + best model, .predict(text) → 0/1
│       └── threshold_tuning.py          ← sweep threshold, optimize recall(1) >= 0.97
│
│
├── data/
│   ├── raw/                             ← downloaded sources — never edited
│   │   ├── hf_singlish/
│   │   ├── hf_sold/
│   │   ├── hf_tanglish/
│   │   ├── hf_squad/
│   │   ├── geonames_lk/
│   │   └── madurapa/
│   │
│   ├── processed/
│   │   ├── sl_gazetteer.csv             ← merged location dataset (~17k rows)
│   │   └── sl_location_aliases.csv      ← hand-curated WhatsApp shortforms
│   │
│   ├── grounding/
│   │   └── unicode_verification.csv     ← 200 Sinhala/Tamil script rows (rule engine proof)
│   │
│   └── datasets/                        ← dataset-agnostic: one folder per dataset
│       └── v1_baseline/                 ← the one dataset for this project
│           ├── raw/
│           │   ├── generated_raw.csv
│           │   ├── generated_evaluated.csv
│           │   ├── generated_deduped.csv
│           │   ├── generated_labeled.csv
│           │   └── dedup_review.csv     ← similarity pairs for manual inspection
│           ├── train.csv
│           ├── val.csv
│           ├── test.csv                 ← DO NOT TOUCH until Phase 5
│           ├── quality_report.json
│           └── dedup.db                 ← Milvus Lite index (portable .db file)
│
│
├── experiments/                         ← all training artifacts, isolated per dataset
│   └── v1_baseline/                     ← mirrors the dataset folder name
│       ├── classical_ml/
│       │   ├── models/
│       │   │   ├── tfidf_char__svm.pkl
│       │   │   ├── tfidf_char__svm_vectorizer.pkl
│       │   │   └── ...                  ← one .pkl pair per combination
│       │   └── results/
│       │       ├── runs.csv             ← exported MLflow run table
│       │       ├── best_params.json     ← HPO winner params
│       │       └── plots/
│       │           └── confusion_matrix__tfidf_char__svm.png
│       │
│       └── transformers/
│           ├── models/
│           │   ├── xlmr-base/           ← HuggingFace checkpoint folder
│           │   │   ├── model.safetensors
│           │   │   ├── config.json
│           │   │   └── tokenizer files
│           │   ├── xlmr-base_onnx/      ← ONNX export (if inference.py run with flag)
│           │   └── ...
│           └── results/
│               ├── runs.csv
│               ├── best_params.json
│               └── plots/
│                   └── training_curves__xlmr-base.png
│
│
├── results/                             ← GLOBAL results across all models
│   ├── master_comparison.csv            ← all models × all metrics in one table
│   ├── ablation_results.csv             ← rule-only vs llm-classifier vs trained
│   ├── cost_simulation.json
│   ├── latency_benchmark.json
│   └── final_recommendation.md         ← winning config with rationale
│
│
├── mlruns/                              ← MLflow tracking folder (git-tracked)
│
└── assets/
    └── graphs/                          ← global comparison plots (.png)
```

---

## KEY DESIGN DECISIONS

### Single dataset, dataset-agnostic structure
One dataset (v1_baseline) for this project. The folder structure under
data/datasets/ and experiments/ is named by dataset, not hardcoded.
If a v2 is ever needed, run Phase 1 with --dataset-name v2_production_fix.
A new folder appears. compare_all.py discovers it automatically.
No refactoring. No code changes.

### Why experiments/ mirrors data/datasets/ naming
Every model in experiments/v1_baseline/ was trained on data/datasets/v1_baseline/.
The naming makes this obvious without checking MLflow.
MLflow provides the authoritative lineage record via mlflow.log_input().
The folder naming is human-readable redundancy.

### Why test.csv is untouched until Phase 5
Training on or evaluating against the test set during development is
data leakage. All development uses train.csv and val.csv only.
Test set is accessed ONCE — in Phase 5 final evaluation for the HPO winner.

### Gradio UI is fully dynamic — no hardcoded lists
ui_utils.py scans data/datasets/ and experiments/ to populate dropdowns.
REGISTRY keys are imported from config.py at runtime.
After any phase run that creates new folders: click Refresh → dropdowns update.
The UI reflects the actual state of the filesystem at all times.

### No system prompt editing in Gradio UI
System prompts live in phases/phase_1_data/03_distilabel_generator.py.
They are versioned in git alongside the code.
Editing prompts = editing code = git diff shows exactly what changed.
This is cleaner and safer than a freeform UI text area.

### MLflow replaces DVC
mlflow.log_input() provides dataset lineage: name, hash, source, schema.
Every training run links to the exact dataset it used.
This covers the dataset tracking use case without adding DVC as a dependency.
