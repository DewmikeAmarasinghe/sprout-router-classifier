# 07 — File Tree (v7, current)

Project root: `model-router-classifier/`

```
model-router-classifier/
│
├── cli.py                          ← dev tool: preview, dryrun, examples-all, distribution
│
├── src/
│   ├── backend/
│   │   ├── config/
│   │   │   ├── keys.py             ← StrEnum: LanguageKey, IndustryKey, ScenarioKey
│   │   │   ├── settings.py         ← global constants
│   │   │   ├── language_configs.py ← LANGUAGE_CONFIGS: 5 formats
│   │   │   ├── industry_configs.py ← INDUSTRY_CONFIGS: 8 industries
│   │   │   ├── scenario_configs.py ← SCENARIO_CONFIGS: 9 scenarios
│   │   │   └── distribution.py     ← DISTRIBUTION: 360 cells, fraction table
│   │   │
│   │   ├── shared/
│   │   │   ├── script_detector.py  ← is_pure_script(text) → bool  [THE ONLY RULE]
│   │   │   ├── metrics.py          ← compute_all_metrics(), time_inference()
│   │   │   ├── path_resolver.py    ← path helpers, IS_KAGGLE / IS_COLAB detection
│   │   │   └── settings_manager.py ← SettingsManager singleton (in-memory only)
│   │   │
│   │   ├── generation/
│   │   │   ├── pymodels.py         ← LengthRange, GenerationCell, GenerationBatch
│   │   │   ├── prompt_factory.py   ← PromptFactory + 4 SectionBuilders
│   │   │   ├── example_store.py    ← ExampleStore: thread-safe, LengthRange fallback
│   │   │   ├── examples.json       ← generated once with cli.py examples-all (git-tracked)
│   │   │   ├── generator.py        ← GeneratorService: parallel cells, multi-turn, backoff
│   │   │   ├── splitter.py         ← DataSplitter: stratified 80/10/10
│   │   │   └── pipeline.py         ← DataPipeline: generate → split
│   │   │
│   │   ├── training/
│   │   │   ├── pymodels.py         ← MetricsResult, ExperimentResult ✅
│   │   │   ├── classical/
│   │   │   │   ├── config.py       ← VectorizerSpec, ClassifierSpec, ACTIVE_COMBOS ✅
│   │   │   │   ├── vectorizers.py  ← build_vectorizer() + W2V + spaCy transformers ✅
│   │   │   │   ├── classifiers.py  ← build_classifier() factory ✅
│   │   │   │   ├── trainer.py      ← ClassicalMLTrainer ✅
│   │   │   │   └── hpo.py          ← ClassicalHPORunner [TODO — Phase 5 HPO]
│   │   │   └── transformers/
│   │   │       ├── config.py       ← TransformerSpec registry [TODO — Phase 6]
│   │   │       ├── dataset.py      ← CSV → HuggingFace Dataset + tokenizer [TODO]
│   │   │       ├── trainer.py      ← TransformerTrainer [TODO]
│   │   │       ├── hpo.py          ← TransformerHPORunner [TODO]
│   │   │       └── onnx_exporter.py ← export_to_onnx() [TODO]
│   │   │
│   │   ├── evaluation/
│   │   │   ├── pymodels.py         ← ComparisonRow, AblationResult [TODO — Phase 7]
│   │   │   ├── comparator.py       ← ModelComparator [TODO]
│   │   │   ├── ablation.py         ← AblationRunner [TODO]
│   │   │   ├── cost_simulator.py   ← CostSimulator [TODO]
│   │   │   └── error_analyzer.py   ← ErrorAnalyzer [TODO]
│   │   │
│   │   ├── router/
│   │   │   ├── pymodels.py         ← RouterPrediction, ThresholdConfig [TODO — Phase 8]
│   │   │   ├── predictor.py        ← RouterPredictor [TODO]
│   │   │   └── threshold_tuner.py  ← ThresholdTuner [TODO]
│   │   │
│   │   └── api/
│   │       ├── main.py             ← FastAPI + Gradio at /ui  (no --reload during generation)
│   │       ├── routes_config.py    ← GET /api/config (read-only)
│   │       └── routes_generation.py ← prompt preview endpoint
│   │
│   └── frontend/
│       ├── app.py                  ← gr.Blocks: 5 tabs (Generation, EDA, Train, Evaluate, Router)
│       └── panels/
│           ├── panel_generation.py ← Generation tab ✅
│           ├── panel_eda.py        ← [TODO — Phase 4 results viewer]
│           ├── panel_training.py   ← [TODO — Phase 5+6 results viewer]
│           ├── panel_evaluation.py ← [TODO — Phase 7 results viewer]
│           └── panel_router.py     ← [TODO — Phase 8 router]
│
├── phases/
│   ├── phase_1_grounding.py        ← 500 Sinhala + 500 Tamil → verify is_pure_script() ✅
│   ├── phase_2_generate.py         ← MAIN GENERATION: 60k rows parallel ✅
│   ├── phase_3_split.py            ← 80/10/10 stratified split ✅
│   ├── phase_4_eda.py              ← EDA: plots + summary stats ✅
│   ├── phase_5_train_classical.py  ← 13 (vectorizer × classifier) experiments ✅
│   ├── phase_6_train_transformers.py ← XLM-RoBERTa, MuRIL, mBERT [TODO]
│   ├── phase_7_evaluate.py         ← Full evaluation on test.csv [TODO]
│   └── phase_8_router.py           ← Threshold tuning + router [TODO]
│
├── data/
│   ├── grounding/
│   │   └── unicode_verification.csv  ← written by phase_1_grounding.py
│   └── datasets/
│       └── v1/
│           ├── raw/
│           │   ├── generated_raw.csv   ← after phase_2_generate.py
│           │   └── checkpoint.csv      ← live during generation
│           ├── train.csv               ← 80%  — all development here
│           ├── val.csv                 ← 10%  — development evaluation
│           ├── test.csv                ← 10%  — DO NOT TOUCH until Phase 7
│           └── split_stats.json
│
├── experiments/
│   └── v1/
│       ├── eda_summary.json
│       ├── eda_plots/              ← 5 EDA plots
│       ├── classical/
│       │   ├── models/
│       │   │   ├── tfidf_combined__lightgbm/model.pkl
│       │   │   ├── tfidf_combined__logistic_regression/model.pkl
│       │   │   └── ...
│       │   └── results/
│       └── transformers/
│           ├── models/
│           └── results/
│
├── results/
│   ├── master_comparison.csv       ← all experiments, all approaches
│   └── final_recommendation.md
│
├── mlruns/                         ← MLflow tracking (run: mlflow ui)
├── docs/
├── AGENTS.md
├── pyproject.toml
├── .env                            ← OPENAI_API_KEY (gitignored)
├── .env.example
└── sync.sh
```

---

## FILES TO DELETE

```bash
rm phases/phase_3_evaluate.py       # LLM-as-judge removed
rm src/frontend/panels/panel_config.py  # Config tab removed
rm -rf src/backend/config/snapshots/    # snapshot functionality removed
```

---

## PHASE STATUS

| Phase | File | Status | Command |
|-------|------|--------|---------|
| 0 | Setup examples | ✅ Done | `python cli.py examples-all --workers 20` |
| 1 | Grounding | ✅ Implemented | `python phases/phase_1_grounding.py` |
| 2 | Generate 60k | 🔄 Running | `python phases/phase_2_generate.py --workers 10` |
| 3 | Split | ⏳ After phase 2 | `python phases/phase_3_split.py` |
| 4 | EDA | ✅ Implemented | `python phases/phase_4_eda.py` |
| 5 | Classical ML | ✅ Implemented | `python phases/phase_5_train_classical.py --all` |
| 6 | Transformers | ⏳ TODO | `python phases/phase_6_train_transformers.py` |
| 7 | Evaluate | ⏳ TODO | `python phases/phase_7_evaluate.py` |
| 8 | Router | ⏳ TODO | `python phases/phase_8_router.py` |

---

## KEY DESIGN DECISIONS

- **No LLM-as-judge** — removed. Trust structured output + distribution design.
- **No underscore prefix** on module-level helpers (only on private class methods).
- **20 workers safe** with exponential backoff on RateLimitError.
- **No --reload** when running uvicorn during generation.
- **examples.json** — generate once with `cli.py examples-all`. Used as-is during generation.
- **test.csv** — never accessed before Phase 7. All development on train/val only.