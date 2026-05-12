# 04 — TRANSFORMERS PLAN (v3)

---

## OVERVIEW

Fine-tune pretrained multilingual models on the labeled dataset.
Everything lives in phases/phase_4_transformers/.
Run these scripts on Kaggle GPU (2× T4, 30h/week free).

WHY transformers (vs just classical ML):
- Transformers understand sentence context — "sollu tion" is not Tamil
- Captures implied location patterns ("nearest branch to me")
  because it attends to the whole sentence meaning
- Captures response-language requests ("answer in sinhala")
- Higher ceiling on all metrics, especially for ambiguous edge cases
- XLM-RoBERTa has seen 2.5TB of multilingual text — it already understands
  Singlish and Tanglish patterns without explicit vocabulary rules

WHY transformers may NOT win:
- ~20–50ms inference vs ~1ms for classical ML
- Requires GPU for training (CPU inference is slow)
- If classical ML reaches recall(1) >= 0.97: classical ML wins on latency + cost

---

## CONFIG — phases/phase_4_transformers/config.py

TRANSFORMER_REGISTRY = {
    "xlmr-base":  "xlm-roberta-base",
    "xlmr-large": "xlm-roberta-large",
    "papluca":    "papluca/xlm-roberta-base-language-detection",
    "muril":      "google/muril-base-cased",
    "mbert":      "bert-base-multilingual-cased",
    # ADD NEW MODELS HERE — one line:
    # "indic-bert": "ai4bharat/indic-bert",
}

TRAIN_CONFIG = {
    "max_length":    64,     # most Sprout queries < 50 tokens
    "num_epochs":    3,
    "batch_size":    32,
    "learning_rate": 2e-5,
    "warmup_ratio":  0.06,
    "weight_decay":  0.01,
    "fp16":          True,   # requires GPU — disable for CPU runs
}

HPO_SEARCH_SPACE = {
    "learning_rate": ["float_log", 1e-5, 5e-5],
    "batch_size":    ["categorical", [16, 32, 64]],
    "num_epochs":    ["categorical", [2, 3, 5]],
    "warmup_ratio":  ["float", 0.0, 0.1],
    "weight_decay":  ["categorical", [0.0, 0.01, 0.1]],
    "max_length":    ["categorical", [64, 128]],
}

---

## HOW FINE-TUNING WORKS (plain English)

XLM-RoBERTa has already read 2.5TB of multilingual text including
Sinhala, Tamil, English, and their romanized code-mixed variants.
Fine-tuning means:

  1. Take the pretrained model (125M weights for base, 560M for large)
  2. Attach a new classification head: Linear(768 → 2) for base models
  3. Train 2–3 epochs on your 49k labeled rows
  4. All weights shift slightly toward your routing task
  5. The multilingual knowledge is preserved — you're steering, not retraining

The [CLS] token's final hidden state (768-dim) represents the whole sentence.
The classification head maps it to 2 logits → softmax → P(label=0), P(label=1).

Training time on Kaggle T4:
  xlmr-base:  ~50 minutes per run
  xlmr-large: ~100 minutes per run
  muril:      ~60 minutes per run

---

## MODEL GUIDE

Model          HuggingFace ID                              Sinhala Tamil Tanglish Notes
─────────────────────────────────────────────────────────────────────────────────────────
xlmr-base      xlm-roberta-base                            ✅ Best  ✅    ✅    START HERE
xlmr-large     xlm-roberta-large                           ✅ Best  ✅    ✅    4× slower, higher ceiling
papluca        papluca/xlm-roberta-base-language-detection ✅       ✅    ✅    Pre-finetuned on 20 langs
muril          google/muril-base-cased                     ❌       ✅Best ✅Best For Tamil/Tanglish focus
mbert          bert-base-multilingual-cased                ❌       ✅    ⚠️   Comparison baseline only

Notes:
  xlmr-base:  Consistently best for Sinhala in academic literature.
              Start here. If it achieves recall(1) >= 0.98, stop.
  xlmr-large: Only worth the extra GPU time if xlmr-base plateaus.
  papluca:    Already fine-tuned on 20-language detection.
              May need fewer epochs. Test with 1–2 epochs first.
  muril:      Trained on transliterated Indian languages including Tamil.
              Best complement to xlmr-base if you build an ensemble.
  mbert:      Older, weaker. Run once for comparison, not production.

---

## dataset.py — CSV TO HUGGINGFACE DATASET

  from datasets import Dataset
  from transformers import AutoTokenizer

  def load_and_tokenize(train_df, val_df, test_df, model_name, max_length):
      tokenizer = AutoTokenizer.from_pretrained(model_name)

      def tokenize(batch):
          return tokenizer(
              batch["text"],
              truncation=True,
              padding="max_length",
              max_length=max_length
          )

      train_ds = Dataset.from_pandas(train_df[["text","label"]]).map(tokenize, batched=True)
      val_ds   = Dataset.from_pandas(val_df[["text","label"]]).map(tokenize, batched=True)
      test_ds  = Dataset.from_pandas(test_df[["text","label"]]).map(tokenize, batched=True)

      return train_ds, val_ds, test_ds, tokenizer

  WHY max_length=64:
  95th percentile of Sprout queries is < 50 tokens.
  64 covers even edge cases. 128 wastes memory and slows training by 30%.

---

## train_single.py — IMPLEMENTATION SPEC

  CLI args: --dataset-name, --model (key from TRANSFORMER_REGISTRY),
            --params '{"learning_rate": 3e-5}'

  1. Parse args, load dataset CSVs
  2. Load model_name from TRANSFORMER_REGISTRY[model]
  3. Merge --params with TRAIN_CONFIG defaults
  4. tokenize train and val via dataset.py
  5. Load AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
  6. Build TrainingArguments from merged config
  7. Build Trainer with compute_metrics callback
  8. trainer.train()
  9. val_results = trainer.evaluate()
  10. Measure latency: compute_latency_stats on 1000 val samples
  11. Measure peak RAM: tracemalloc during inference batch
  12. Log everything to MLflow:
        experiment: "transformers_{dataset_name}"
        run_name: model key (e.g. "xlmr-base")
        params: all TRAIN_CONFIG values + model_name + dataset_name
        metrics: val f1_macro, recall_1, mcc, latency_p99_ms, train_time_s, etc.
        artifacts: training_curves.png, confusion_matrix.png
        dataset: mlflow.log_input(train_dataset)
  13. Save checkpoint: experiments/{dataset_name}/transformers/models/{model_key}/
  14. Print metrics table

  def compute_metrics_hf(eval_pred):
      logits, labels = eval_pred
      preds = np.argmax(logits, axis=-1)
      proba = softmax(logits, axis=-1)[:, 1]
      m = compute_all_metrics(labels, preds, proba)
      return {"f1_macro": m["f1_macro"], "recall_1": m["recall_1"], "mcc": m["mcc"]}

---

## train_all.py — LOOP ALL MODELS

  Calls train_single logic for each model in TRANSFORMER_REGISTRY.
  All models run on the same dataset version.
  At end: export MLflow runs to experiments/{dataset_name}/transformers/results/runs.csv

---

## hpo.py — OPTUNA HPO FOR TRANSFORMERS

  CLI args: --dataset-name --model X --n-trials N

  Uses MedianPruner: kills underperforming trials after epoch 1.
  Saves ~50% GPU time vs running all trials to completion.

  def objective(trial):
      lr    = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
      bs    = trial.suggest_categorical("batch_size", [16, 32, 64])
      epochs = trial.suggest_categorical("num_epochs", [2, 3, 5])
      # train, evaluate, prune if epoch 1 is below median
      return val_f1_macro

  Expected time for 10 trials with pruning: ~2–3 hours on Kaggle T4.
  Best params saved to: experiments/{dataset_name}/transformers/results/best_params.json

---

## inference.py — LATENCY BENCHMARK + ONNX EXPORT

  CLI args: --dataset-name --model X [--export-onnx]

  What it does:
    1. Load saved checkpoint
    2. Run 1000 predictions, measure p50/p95/p99 latency
    3. Test on 50 tricky edge cases (Singlish typos, ambiguous messages)
    4. Optional: export to ONNX for CPU deployment speed

  ONNX export (2–4× faster on CPU, < 1% accuracy drop):
    from optimum.onnxruntime import ORTModelForSequenceClassification
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        checkpoint_path, export=True, provider="CPUExecutionProvider"
    )
    ort_model.save_pretrained(f"{checkpoint_path}_onnx")

  When to use ONNX:
    If the transformer wins and must be deployed on CPU (no GPU in production).
    ONNX reduces latency from ~50ms to ~12–15ms on CPU.

---

## RUNNING SCRIPTS

  # Fine-tune one model
  python phases/phase_4_transformers/train_single.py \
      --dataset-name v1_baseline \
      --model xlmr-base

  # With custom hyperparams
  python phases/phase_4_transformers/train_single.py \
      --dataset-name v1_baseline \
      --model xlmr-base \
      --params '{"learning_rate": 3e-5, "num_epochs": 5}'

  # Fine-tune all models
  python phases/phase_4_transformers/train_all.py \
      --dataset-name v1_baseline

  # HPO
  python phases/phase_4_transformers/hpo.py \
      --dataset-name v1_baseline \
      --model xlmr-base \
      --n-trials 10

  # Latency benchmark + ONNX
  python phases/phase_4_transformers/inference.py \
      --dataset-name v1_baseline \
      --model xlmr-base \
      --export-onnx

---

## ADDING A NEW TRANSFORMER MODEL

  Edit phases/phase_4_transformers/config.py:
    TRANSFORMER_REGISTRY["indic-bert"] = "ai4bharat/indic-bert"

  Run:
    python train_single.py --dataset-name v1_baseline --model indic-bert

  Constraint: model must be compatible with AutoModelForSequenceClassification.
  This covers ~99% of BERT-family models on HuggingFace.
  Nothing else changes. train_all.py picks it up automatically.

---

## OUTPUT PER TRAINING RUN

  MLflow: all params + metrics + training curves + confusion matrix
  Checkpoint: experiments/{dataset_name}/transformers/models/{model_key}/
              (full HuggingFace checkpoint — model.safetensors, config.json, tokenizer)
  ONNX (if exported): experiments/{dataset_name}/transformers/models/{model_key}_onnx/
  Results CSV: experiments/{dataset_name}/transformers/results/runs.csv
