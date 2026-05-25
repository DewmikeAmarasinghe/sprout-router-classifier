# Sprout Router Classifier

Binary ML classifier that routes incoming WhatsApp/chat messages to the right LLM for [hellosprout.ai](https://hellosprout.ai) (hSenid Mobile).

| Label | Model | When |
|-------|-------|------|
| **0** | `gpt-4o-mini` | Pure English + simple intent (cheap) |
| **1** | `gpt-4o` | Code-mixed language, complex/sensitive scenarios, Unicode scripts (Sinhala/Tamil) |

The router sits in front of the chatbot LLM. A fast classifier decides which model handles each message, cutting API cost while keeping complex messages on the capable model.

---

## How routing works

```
Incoming message
       │
       ▼
┌──────────────────────┐
│ Layer 1 — Script     │  Sinhala/Tamil Unicode detected?
│ detection            │  → label=1, confidence=1.0 (skip ML)
└──────────┬───────────┘
           │ pure romanized text
           ▼
┌──────────────────────┐
│ Layer 2 — ML model   │  P(label=1) from classical or transformer
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Layer 3 — Threshold  │  confidence ≥ threshold → gpt-4o (label=1)
│                      │  confidence < threshold → gpt-4o-mini (label=0)
└──────────────────────┘
```

Script detection lives in `RouterPredictor.predict()` — all callers get it automatically.

---

## Project structure

```
sprout-router-classifier/
├── cli.py                          # Dev CLI (preview, dry-run, clean, examples)
├── kaggle_train_transformers.py    # Full training pipeline for Kaggle GPU
├── phases/                         # Numbered pipeline scripts (phase_1 … phase_8)
├── data/
│   ├── datasets/v1/                # train.csv, val.csv, test.csv (80/10/10 split)
│   └── grounding/                  # Unicode script verification samples
├── experiments/v1/                 # Results, plots, router config (models gitignored)
├── src/
│   ├── backend/
│   │   ├── config/                 # Distribution, language/industry/scenario configs
│   │   ├── generation/             # LLM dataset generation
│   │   ├── training/
│   │   │   ├── classical/          # TF-IDF + sklearn/boosting models
│   │   │   └── transformers/       # XLM-R, MuRIL, mBERT fine-tuning
│   │   ├── evaluation/             # Comparison, cost sim, error analysis
│   │   ├── router/                 # Predictor + threshold tuning
│   │   └── api/                    # FastAPI routes
│   └── frontend/                   # Gradio UI panels
└── docs/                           # Additional documentation
```

---

## Requirements

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- OpenAI API key (dataset generation only)
- GPU optional locally; **Kaggle T4** recommended for transformer training

---

## Setup

```bash
git clone https://github.com/DewmikeAmarasinghe/sprout-router-classifier.git
cd sprout-router-classifier

uv sync                    # install dependencies
cp .env.example .env       # add OPENAI_API_KEY
```

Start the Gradio UI + FastAPI server:

```bash
uv run python src/backend/api/main.py
# or
uv run uvicorn backend.api.main:app --host 0.0.0.0 --port 7860
```

| URL | Purpose |
|-----|---------|
| http://localhost:7860/ui | Gradio UI (Generation, EDA, Train, Evaluate, Router) |
| http://localhost:7860/docs | FastAPI auto-docs |

---

## Pipeline overview

| Phase | Script | Purpose |
|-------|--------|---------|
| 1 | `phase_1_grounding.py` | Verify Unicode script detector |
| 2 | `phase_2_generate.py` | Generate training dataset via LLM |
| 3 | `phase_3_split.py` | Stratified 80/10/10 split |
| 4 | `phase_4_eda.py` | Exploratory data analysis |
| 5 | `phase_5_train_classical.py` | Classical ML (CPU, ~minutes) |
| 6 | `phase_6_train_transformers.py` | Transformer fine-tuning (GPU, ~hours) |
| 7 | `phase_7_evaluate.py` | Model comparison + cost simulation |
| 8 | `phase_8_router.py` | Threshold sweep on val set |

### Dataset already generated?

If `data/datasets/v1/train.csv` exists, skip phases 1–4 and go straight to training.

### Full local workflow

```bash
# 1. Generate cell examples (once, before dataset generation)
uv run python cli.py examples-all --workers 7

# 2. Generate dataset (~58k rows, requires OPENAI_API_KEY)
uv run python phases/phase_2_generate.py --workers 7

# 3. Split
uv run python phases/phase_3_split.py

# 4. EDA (optional)
uv run python phases/phase_4_eda.py

# 5. Train classical ML locally
uv run python phases/phase_5_train_classical.py --all-active   # 14 curated combos (~30 min)
uv run python phases/phase_5_train_classical.py --all          # all 25 combos (~60–90 min)
uv run python phases/phase_5_train_classical.py --all --no-hpo # skip Optuna HPO

# 6. Train transformers (GPU recommended — use Kaggle for this)
uv run python phases/phase_6_train_transformers.py --model xlmr-base

# 7. Evaluate
uv run python phases/phase_7_evaluate.py --dataset v1

# 8. Tune router threshold
uv run python phases/phase_8_router.py --dataset v1
uv run python phases/phase_8_router.py --dataset v1 --test "nearest branch to me"
```

### Clean training outputs

```bash
uv run python cli.py clean --dataset v1
```

Removes `experiments/v1/classical/`, `experiments/v1/transformers/`, `experiments/v1/router/`, comparison CSVs, MLflow DB, and `mlruns/`. Does **not** touch train/val/test CSVs.

---

## Kaggle training (recommended for transformers)

Upload `kaggle_train_transformers.py` as a Kaggle notebook. Enable **GPU T4 x2**, add a `GITHUB_PAT` secret, and run all cells.

The notebook runs:

1. Clone repo → install deps
2. **Phase 5** — classical ML (`--all`, 25 combos, CPU)
3. **Phase 6** — transformer models one cell each (xlmr-base, papluca, muril, mbert, xlmr-large)
4. **Phase 7** — evaluation
5. **Phase 8** — router threshold tuning
6. Zip `experiments/`, `mlflow.db`, `mlruns/` for download

After download:

```bash
cp -r ~/Downloads/experiments/v1/ ./experiments/v1/
uv run python phases/phase_7_evaluate.py --dataset v1
uv run python phases/phase_8_router.py --dataset v1
```

Model artifacts (`.pkl`, transformer checkpoints) are **not** committed to git — download from Kaggle outputs.

---

## Classical ML

Five vectorizers × five classifiers = 25 combinations. The backend supports any pair; `--all-active` runs a curated subset of 14 known-good combos.

| Vectorizer | Description |
|------------|-------------|
| `tfidf_combined` | Char + word TF-IDF (usually best) |
| `tfidf_char` | Character ngrams — good for Singlish/Tanglish |
| `tfidf_word` | Word unigrams + bigrams |
| `word2vec` | Corpus-trained W2V mean pooling |
| `spacy` | Pre-trained `en_core_web_md` vectors |

| Classifier | Notes |
|------------|-------|
| `logistic_regression` | Fast baseline |
| `svm` | LinearSVC + calibration — **best recall/MCC on TF-IDF** |
| `lightgbm` | Gradient boosting |
| `xgboost` | Best with dense vectorizers (word2vec, spacy) |
| `catboost` | Handles class imbalance |

All classical models run on **CPU**. No epochs — each experiment is a single `fit()` call.

### Auto-HPO (Optuna)

After `--all` or `--all-active`, HPO runs automatically unless `--no-hpo` is passed:

1. Filter models that pass **recall_1 ≥ 0.95** (production safety gate)
2. Pick the one with **highest MCC** (balanced precision/recall)
3. Run 10 Optuna trials optimizing MCC on **val.csv**
4. Retrain with best hyperparameters

HPO never touches `test.csv`.

**Production pick (v1 dataset):** `tfidf_combined__svm` — recall_1 ≈ 0.97, MCC ≈ 0.81, p95 ≈ 3.5 ms.

---

## Transformers

| Model key | HuggingFace name | VRAM | Notes |
|-----------|------------------|------|-------|
| `xlmr-base` | `xlm-roberta-base` | 8 GB+ | Start here |
| `papluca` | `papluca/xlm-roberta-base-language-detection` | 8 GB+ | Language detection tuned |
| `muril` | `google/muril-base-cased` | 8 GB+ | South Asian scripts |
| `mbert` | `bert-base-multilingual-cased` | 6 GB+ | Smallest |
| `xlmr-large` | `xlm-roberta-large` | 12 GB+ | Best quality, slowest |

Full fine-tuning of all weights, **3 epochs**, AdamW lr=2e-5. Auto-HPO runs if recall_1 < threshold after training.

```bash
uv run python phases/phase_6_train_transformers.py --model xlmr-base
uv run python phases/phase_6_train_transformers.py --all --dataset v1
uv run python phases/phase_6_train_transformers.py --model xlmr-base --hpo --n-trials 5
```

---

## Train / val / test split

| Set | Used for |
|-----|----------|
| `train.csv` | Fitting vectorizers and classifiers |
| `val.csv` | Development metrics, HPO, threshold tuning |
| `test.csv` | Final benchmark only (phase 7 `--ablate`) |

**Target distribution (v1):** ~30% label=0 (gpt-4o-mini), ~70% label=1 (gpt-4o).

---

## Key metrics

| Metric | Meaning | Production use |
|--------|---------|----------------|
| **recall_1** | Fraction of complex messages correctly routed to gpt-4o | **Primary safety gate** (≥ 0.95) |
| **precision_1** | Of messages sent to gpt-4o, how many truly needed it | Cost control |
| **MCC** | Matthews Correlation — balanced score across both classes | **Tiebreaker** among passing models |
| **ROC-AUC** | Ranking quality independent of threshold | Model comparison |
| **p95 latency** | 95th percentile inference time (ms) | Production SLA |

Selection rule: pass recall_1 threshold → highest MCC → lowest latency.

---

## Configuration

All defaults live in `src/backend/config/settings.py` and are read via `settings_manager.get("KEY")`.

| Key | Default | Purpose |
|-----|---------|---------|
| `PRODUCTION_RECALL_THRESHOLD` | `0.95` | Minimum recall_1 to deploy |
| `CONFIDENCE_THRESHOLD` | `0.70` | Router routing threshold (tuned by phase 8) |
| `SAFE_DEFAULT_LABEL` | `1` | Route to gpt-4o when confidence is low |
| `DAILY_MESSAGES_ESTIMATE` | `25000` | Cost simulation baseline (~$50/day all-gpt-4o) |
| `DATASET_VERSION` | `v1` | Active dataset |

Prompt content (languages, industries, scenarios) comes from config dataclasses — never hardcoded in `prompt_factory.py`. See `docs/HOW_SYSTEM_PROMPTS_WORK.md` for details.

---

## Dev CLI

```bash
uv run python cli.py distribution          # print dataset distribution breakdown
uv run python cli.py preview  --language pure_english --industry banking --scenario simple_transactional
uv run python cli.py dryrun   --language singlish_light --industry banking --scenario continuation --n 5
uv run python cli.py examples --language singlish_light --industry banking --scenario location_proximity
uv run python cli.py examples-all --workers 7
uv run python cli.py clean --dataset v1
```

---

## Linting

```bash
uv run ruff format .
uv run ruff check --fix .
uv run ty check
```

---

## Outputs

```
experiments/v1/
├── classical/models/{experiment_id}/   # model.pkl + result.json (gitignored)
├── transformers/models/{model_key}/    # HF checkpoint + result.json (gitignored)
├── router/threshold_curve.json         # optimal confidence threshold
├── master_comparison.csv               # all models ranked
├── cost_simulation.json                # daily cost vs routing strategy
├── error_analysis.json                 # false negative breakdown
└── eda_plots/                          # PNG charts from phase 4
```

Track experiments locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## License

Private — hSenid Mobile / hellosprout.ai. All rights reserved.
