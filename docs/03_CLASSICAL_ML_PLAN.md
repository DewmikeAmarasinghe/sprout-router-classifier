# 03 — CLASSICAL ML PLAN (v3)

---

## OVERVIEW

Classical ML = text → vectorizer → classifier → 0 or 1.
Everything lives in phases/phase_3_classical_ml/.
Every vectorizer × classifier combination = one MLflow run.
All combinations run automatically in train_all.py.
Individual combos tested via train_single.py during development.
HPO runs on the best combo after train_all completes.

WHY test classical ML at all (vs just transformers):
- Classical ML runs on CPU — no GPU budget needed
- Inference is ~1ms (vs ~20–50ms for transformers)
- TF-IDF char n-grams naturally capture Sinhala/Tamil unicode sequences
  and romanized Singlish patterns without needing language understanding
- If classical ML reaches recall(1) >= 0.97, it is the production winner
  due to its massive latency and cost advantage over transformers

---

## CONFIG — phases/phase_3_classical_ml/config.py

The ONLY file to edit when adding a new vectorizer or classifier.
Keys are strings used as CLI args and MLflow run name components.

VECTORIZER_REGISTRY = {
    "tfidf_char":  {type: "tfidf", analyzer: "char_wb", ngram_range: (2,4), max_features: 100000, sublinear_tf: True},
    "tfidf_word":  {type: "tfidf", analyzer: "word",    ngram_range: (1,3), max_features: 100000},
    "word2vec":    {type: "word2vec", vector_size: 200, window: 5, min_count: 2, pooling: "mean"},
    "spacy":       {type: "spacy", model: "xx_ent_wiki_sm"},
    "combined":    {type: "combined", base: "tfidf_char", extra: "word2vec"},
}

CLASSIFIER_REGISTRY = {
    "logreg":    LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"),
    "svm":       CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3),
    "lgbm":      LGBMClassifier(n_estimators=500, num_leaves=63, class_weight="balanced"),
    "xgboost":   XGBClassifier(n_estimators=500, scale_pos_weight=3, tree_method="hist"),
    "catboost":  CatBoostClassifier(iterations=500, auto_class_weights="Balanced", verbose=0),
    "rf":        RandomForestClassifier(n_estimators=300, class_weight="balanced"),
}

FASTTEXT_CONFIG = {
    "lr": 1.0, "epoch": 25, "wordNgrams": 2, "dim": 100, "loss": "softmax"
}

HPO_SEARCH_SPACES = {
    "tfidf_char": {
        "ngram_range":  ["categorical", [(2,3),(2,4),(2,5),(3,5)]],
        "max_features": ["int_log", 50000, 200000],
        "min_df":       ["int", 1, 5],
    },
    "tfidf_word": {
        "ngram_range":  ["categorical", [(1,1),(1,2),(1,3)]],
        "max_features": ["int_log", 50000, 200000],
    },
    "word2vec": {
        "vector_size":  ["categorical", [100, 200, 300]],
        "window":       ["int", 3, 7],
        "pooling":      ["categorical", ["mean", "max"]],
    },
    "logreg":    {"C":              ["float_log", 0.01, 100]},
    "svm":       {"C":              ["float_log", 0.01, 100]},
    "lgbm":      {"num_leaves":     ["categorical", [31, 63, 127]],
                  "learning_rate":  ["float_log", 0.01, 0.1],
                  "n_estimators":   ["int", 200, 1000]},
    "xgboost":   {"max_depth":      ["int", 4, 8],
                  "learning_rate":  ["float_log", 0.01, 0.1],
                  "subsample":      ["float", 0.7, 1.0]},
    "catboost":  {"depth":          ["int", 4, 8],
                  "learning_rate":  ["float_log", 0.01, 0.1],
                  "iterations":     ["int", 200, 1000]},
}

---

## VECTORIZERS — phases/phase_3_classical_ml/vectorizers.py

Each vectorizer must implement the sklearn interface: fit(X), transform(X).
All are importable from vectorizers.py via build_vectorizer(name, params).

### A — TF-IDF Char N-gram (tfidf_char)

WHY this is the strongest baseline:
Splits text into overlapping character windows of length 2–4.
Sinhala/Tamil unicode characters form unique char sequences.
Singlish words like "kohomada" produce trigrams (koh, oho, hom, oma...)
that never appear in English text. The vectorizer learns to fire on these
without understanding any language. No language model needed.

  TfidfVectorizer(
      analyzer="char_wb",
      ngram_range=(2, 4),
      max_features=100_000,
      sublinear_tf=True,
      min_df=2,
  )

### B — TF-IDF Word N-gram (tfidf_word)

Captures word-level frequency patterns.
Useful for: multi-word Singlish phrases, "nearest branch", "near me" patterns.

  TfidfVectorizer(
      analyzer="word",
      ngram_range=(1, 3),
      max_features=100_000,
      min_df=2,
  )

### C — Word2Vec Average Pooling (word2vec)

Trains unsupervised on all training text.
Sentence vector = mean of word vectors.
Captures semantic similarity — "nearest" and "closest" map nearby.
Complementary to TF-IDF (dense vs sparse features).

  class Word2VecVectorizer:
      def fit(self, texts):
          sentences = [t.lower().split() for t in texts]
          self.model = Word2Vec(sentences, vector_size=200, window=5,
                                min_count=2, workers=4, seed=42)
      def transform(self, texts):
          # for each text: mean of word vectors for known words
          # unknown words → zero vector
          ...

### D — spaCy Multilingual Vectors (spacy)

Pretrained 300-dim vectors from xx_ent_wiki_sm.
No fitting required — pretrained static vectors.
Captures general multilingual semantic meaning.
Useful as a complementary signal to TF-IDF.

  Requires: python -m spacy download xx_ent_wiki_sm

  class SpacyVectorizer:
      def fit(self, texts): return self  # pretrained, no fitting
      def transform(self, texts):
          return np.array([self.nlp(t).vector for t in texts])

### E — Combined Stack (combined)

Horizontally concatenates TF-IDF char sparse matrix + Word2Vec dense matrix.
Often outperforms either alone because they capture complementary signals.

  class CombinedVectorizer:
      def fit(self, texts):
          self.tfidf.fit(texts)
          self.w2v.fit(texts)
      def transform(self, texts):
          X_tfidf = self.tfidf.transform(texts)        # sparse (N, 100k)
          X_w2v   = csr_matrix(self.w2v.transform(texts))  # dense→sparse (N, 200)
          return hstack([X_tfidf, X_w2v])

### F — FastText (runs standalone, not paired with external vectorizers)

FastText uses its own internal char + word n-gram embeddings.
Trains end-to-end: vectorization + classification in one model.
Write training data to temp file in fasttext __label__ format, train, predict.
FastText does NOT appear in the vectorizer × classifier grid.
It runs as its own standalone experiment in train_all.py.

---

## CLASSIFIERS — phases/phase_3_classical_ml/classifiers.py

All classifiers implement sklearn's fit/predict/predict_proba interface.
All are importable from classifiers.py via build_classifier(name, params).

### LogisticRegression (logreg)
Best for: sparse TF-IDF features. Interpretable coefficients.
Has native predict_proba — calibration not needed.
  LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced")

### LinearSVC (svm)
Best for: high-dimensional sparse features (TF-IDF char). Fastest at inference.
Does NOT have predict_proba natively. Wrap in CalibratedClassifierCV(cv=3).
  CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, class_weight="balanced"), cv=3)

### LightGBM (lgbm)
Best for: dense features (Word2Vec, combined). Leaf-wise growth handles imbalance well.
Fast training. Good out-of-the-box on text classification.
  LGBMClassifier(n_estimators=500, num_leaves=63, learning_rate=0.05,
                 class_weight="balanced", n_jobs=-1, random_state=42)

### XGBoost (xgboost)
Best for: dense features with careful tuning. scale_pos_weight handles class imbalance.
  XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                tree_method="hist", scale_pos_weight=3, n_jobs=-1, seed=42)
  Note: scale_pos_weight = count(label=0) / count(label=1) from training data.

### CatBoost (catboost)
Best for: minimal tuning. auto_class_weights handles imbalance automatically.
Often the strongest boosting model with default settings.
  CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                     auto_class_weights="Balanced", verbose=0, random_seed=42)

### Random Forest (rf)
Stable baseline. Less competitive than boosting models but never catastrophically bad.
  RandomForestClassifier(n_estimators=300, class_weight="balanced",
                         n_jobs=-1, random_state=42)

---

## VECTORIZER × CLASSIFIER GRID

Total combinations: 5 vectorizers × 6 classifiers = 30 runs.
FastText: 1 standalone run.
Total classical ML runs per dataset: 31.

Not all combinations are equally useful:
  tfidf_char  + svm/logreg     → STRONGEST EXPECTED (sparse + linear)
  combined    + lgbm/catboost  → SECOND BEST EXPECTED (dense + boosting)
  word2vec    + lgbm/xgboost   → WORTH TESTING
  spacy       + any            → COMPARISON BASELINE
  tfidf_word  + svm/logreg     → COMPLEMENTARY TO tfidf_char

---

## RUNNING SCRIPTS

  # Full grid for a dataset
  python phases/phase_3_classical_ml/train_all.py --dataset-name v1_baseline

  # Single combination (for development and testing)
  python phases/phase_3_classical_ml/train_single.py \
      --dataset-name v1_baseline \
      --vec tfidf_char \
      --clf svm

  # With custom param override
  python phases/phase_3_classical_ml/train_single.py \
      --dataset-name v1_baseline \
      --vec tfidf_char \
      --clf svm \
      --params '{"C": 10.0, "ngram_range": [2, 5]}'

  # HPO on best combo
  python phases/phase_3_classical_ml/hpo.py \
      --dataset-name v1_baseline \
      --vec tfidf_char \
      --clf svm \
      --n-trials 30

---

## train_single.py — IMPLEMENTATION SPEC

  1. Parse args: --dataset-name, --vec, --clf, --params (JSON string)
  2. Load shared/config.py → get dataset path
  3. Load train.csv and val.csv from data/datasets/{dataset_name}/
  4. Build vectorizer from VECTORIZER_REGISTRY[vec], merge --params overrides
  5. Build classifier from CLASSIFIER_REGISTRY[clf], merge --params overrides
  6. Track: train_start = time.perf_counter()
  7. Fit vectorizer on train texts
  8. Transform train and val
  9. Fit classifier on X_train, y_train
  10. train_time = time.perf_counter() - train_start
  11. Predict on val: y_pred, y_proba
  12. metrics = compute_all_metrics(y_val, y_pred, y_proba)
  13. latency = compute_latency_stats(clf, X_val[:1000])  # p50/p95/p99
  14. peak_ram = get_peak_ram_mb()  # tracemalloc
  15. model_size = get_model_size_mb(clf, vec)  # pickle and measure
  16. Log everything to MLflow (see 06_EXPERIMENT_TRACKING.md for full list)
  17. Save model: experiments/{dataset_name}/classical_ml/models/{vec}__{clf}.pkl
  18. Save vectorizer alongside model
  19. Print metrics table to console

---

## hpo.py — IMPLEMENTATION SPEC

  Uses Optuna with MLflow callback (each trial = nested MLflow run).

  def objective(trial):
      # Sample vectorizer params from HPO_SEARCH_SPACES[vec]
      # Sample classifier params from HPO_SEARCH_SPACES[clf]
      # Train and evaluate
      # Return val f1_macro (Optuna maximizes this)

  study = optuna.create_study(direction="maximize",
                               pruner=optuna.pruners.MedianPruner())
  study.optimize(objective, n_trials=n_trials)

  After HPO:
    - Save best params to experiments/{dataset_name}/classical_ml/results/best_params.json
    - Retrain best config on train+val combined
    - Evaluate ONCE on test.csv (first and only test set access for this model)
    - Log final test metrics to MLflow with tag "hpo_final_eval"

---

## ADDING A NEW CLASSIFIER

  Edit phases/phase_3_classical_ml/config.py:

    from sklearn.naive_bayes import ComplementNB
    CLASSIFIER_REGISTRY["cnb"] = ComplementNB()

  Add its HPO search space:
    HPO_SEARCH_SPACES["cnb"] = {"alpha": ["float_log", 0.01, 10.0]}

  Test with:
    python train_single.py --dataset-name v1_baseline --vec tfidf_char --clf cnb

  Nothing else changes. train_all.py picks it up automatically.

---

## ADDING A NEW VECTORIZER

  Edit phases/phase_3_classical_ml/config.py:
    VECTORIZER_REGISTRY["tfidf_char_large"] = {
        type: "tfidf", analyzer: "char_wb",
        ngram_range: (2, 5), max_features: 200000
    }

  Implement in vectorizers.py if it needs a custom class.
  build_vectorizer() must handle the new type key.
  train_all.py picks it up automatically.

---

## OUTPUT PER TRAINING RUN

  MLflow: all params + all metrics logged (see 06_EXPERIMENT_TRACKING.md)
  File: experiments/{dataset_name}/classical_ml/models/{vec}__{clf}.pkl
  File: experiments/{dataset_name}/classical_ml/models/{vec}__{clf}_vectorizer.pkl
  File: experiments/{dataset_name}/classical_ml/results/plots/confusion_matrix__{vec}__{clf}.png
  Console: formatted metrics table
