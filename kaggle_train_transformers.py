"""
Sprout Router — Full Training Pipeline on Kaggle GPU.

Run all cells sequentially. Everything saves to /kaggle/working/experiments/
which appears in the Kaggle Outputs tab and can be downloaded as a zip.

BEFORE RUNNING:
  1. GitHub PAT: Settings → Developer settings → Fine-grained tokens
     Repository: model-router-classifier, Contents: Read
  2. Kaggle Secret: Account → Secrets → Add → Name: GITHUB_PAT
  3. Notebook settings: GPU T4 x2, Internet: ON

TRAINING ORDER:
  Classical ML: ~20 min (CPU) — all vectorizer × classifier combos, sequential
  Transformer:  ~50 min (T4)  — one model at a time, sequential

After training, download /kaggle/working/experiments/ from the Outputs tab.
"""

# %% [markdown]
# ## Cell 1 — Environment check

# %%
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

print(f"Python:  {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.version.cuda}")
print(f"GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY'}")
if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM:    {vram_gb:.1f} GB")

# %% [markdown]
# ## Cell 2 — Clone private repo

# %%
try:  # noqa: E402
    from kaggle_secrets import UserSecretsClient

    pat = UserSecretsClient().get_secret("GITHUB_PAT")
except ImportError:
    pat = os.environ.get("GITHUB_PAT", "")

REPO = "/kaggle/working/repo"
url = f"https://oauth2:{pat}@github.com/DewmikeAmarasinghe/model-router-classifier.git"
pat = ""  # clear from memory immediately after use

if os.path.exists(f"{REPO}/.git"):
    print("Repo exists — pulling latest...")
    subprocess.run(["git", "-C", REPO, "pull", "--rebase"], check=True)
else:
    print("Cloning repo...")
    subprocess.run(["git", "clone", url, REPO], check=True)

url = ""  # clear url containing PAT
print(f"✅ Repo ready at {REPO}")

# %% [markdown]
# ## Cell 3 — Install dependencies

# %%
subprocess.run(
    [  # noqa: E402
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers>=4.40",
        "datasets",
        "accelerate>=0.27",
        "optimum[onnxruntime]",
        "mlflow",
        "scikit-learn",
        "pydantic>=2",
        "python-dotenv",
        "instructor",
        "scipy",
        "lightgbm",
        "catboost",
        "xgboost",
        "gensim",
        "optuna",
        "seaborn",
        "matplotlib",
    ],
    check=True,
)
print("✅ Dependencies installed")

# %% [markdown]
# ## Cell 4 — Configure paths
#
# All training outputs go to /kaggle/working/experiments/ — persisted in Outputs tab.

# %%
DATASET_VERSION = "v1"  # noqa: E402
EXPERIMENT_DIR = "/kaggle/working/experiments"
DATA_DIR = f"{REPO}/data/datasets"

os.makedirs(EXPERIMENT_DIR, exist_ok=True)

dotenv_path = f"{REPO}/.env"
with open(dotenv_path, "w") as f:
    f.write(f"EXPERIMENT_BASE={EXPERIMENT_DIR}\n")
    f.write(f"DATASET_BASE={DATA_DIR}\n")

print(f"✅ .env written → {dotenv_path}")
print(f"   EXPERIMENT_BASE = {EXPERIMENT_DIR}")
print(f"   DATASET_BASE    = {DATA_DIR}")

# %% [markdown]
# ## Cell 5 — Verify training data
#
# train.csv / val.csv / test.csv must be committed to the repo.
# If missing: run phase_3_split.py locally, commit, push, and re-clone here.

# %%
import pandas as pd  # noqa: E402

train_df = pd.read_csv(f"{DATA_DIR}/{DATASET_VERSION}/train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/{DATASET_VERSION}/val.csv")
test_df = pd.read_csv(f"{DATA_DIR}/{DATASET_VERSION}/test.csv")

print(
    f"Train: {len(train_df):,}  label=0: {(train_df.label == 0).sum():,}  label=1: {(train_df.label == 1).sum():,}"
)
print(
    f"Val:   {len(val_df):,}    label=0: {(val_df.label == 0).sum():,}    label=1: {(val_df.label == 1).sum():,}"
)
print(
    f"Test:  {len(test_df):,}   label=0: {(test_df.label == 0).sum():,}   label=1: {(test_df.label == 1).sum():,}"
)

# %% [markdown]
# ## Cell 6 — Classical ML training
#
# Trains all (vectorizer × classifier) combos one after another (sequential, not parallel).
# Models: tfidf_combined__svm, xgboost, lightgbm, catboost, logistic_regression, and more.
# Results → /kaggle/working/experiments/v1/classical/results/
# Models  → /kaggle/working/experiments/v1/classical/models/

# %%
print("Training all classical ML experiments (sequential, ~20 min)...\n")

result = subprocess.run(  # noqa: E402
    [
        sys.executable,
        f"{REPO}/phases/phase_5_train_classical.py",
        "--all",
        "--dataset",
        DATASET_VERSION,
    ],
    cwd=REPO,
    text=True,
)
print(
    f"{'✅' if result.returncode == 0 else '❌'} Classical training {'complete' if result.returncode == 0 else 'FAILED'}"
)

# %% [markdown]
# ## Cell 7 — Transformer training
#
# Each model trains sequentially (one at a time). GPU is used automatically.
#
# Timing on T4 GPU (~60k training rows, 3 epochs):
#   xlmr-base : ~50 min  ← START HERE (best for Sinhala/Tamil romanized text)
#   papluca   : ~45 min  (language-detection pretrained — strong baseline)
#   muril     : ~55 min  (Google multilingual — strong for Tamil)
#   mbert     : ~50 min  (baseline multilingual BERT)
#   xlmr-large: ~100 min (only if xlmr-base plateaus below 0.97 recall)
#
# All 5 sequentially: ~5 hours — within the 12-hour Kaggle session limit.
# Models → /kaggle/working/experiments/v1/transformers/models/

# %%
MODEL_TO_TRAIN = "xlmr-base"  # noqa: E402

print(f"Training {MODEL_TO_TRAIN}...")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}\n")

result = subprocess.run(
    [
        sys.executable,
        f"{REPO}/phases/phase_6_train_transformers.py",
        "--model",
        MODEL_TO_TRAIN,
        "--dataset",
        DATASET_VERSION,
    ],
    cwd=REPO,
    text=True,
)
print(
    f"{'✅' if result.returncode == 0 else '❌'} {MODEL_TO_TRAIN} {'complete' if result.returncode == 0 else 'FAILED'}"
)

# %% [markdown]
# ## Cell 8 — (Optional) Train additional transformer models
#
# Uncomment the list to train multiple models sequentially.
# Total for all 5: ~5 hours.

# %%
additional_models: list[str] = []  # noqa: E402
# additional_models = ["papluca", "muril", "mbert"]

for model_name in additional_models:
    print(f"\nTraining {model_name}...")
    model_result = subprocess.run(
        [
            sys.executable,
            f"{REPO}/phases/phase_6_train_transformers.py",
            "--model",
            model_name,
            "--dataset",
            DATASET_VERSION,
        ],
        cwd=REPO,
        text=True,
    )
    print(
        f"{'✅' if model_result.returncode == 0 else '❌'} {model_name} {'done' if model_result.returncode == 0 else 'FAILED'}"
    )

# %% [markdown]
# ## Cell 9 — Evaluation
#
# Compares all trained models, runs cost simulation, and error analysis.

# %%
eval_result = subprocess.run(  # noqa: E402
    [sys.executable, f"{REPO}/phases/phase_7_evaluate.py", "--dataset", DATASET_VERSION],
    cwd=REPO,
    text=True,
)
print(
    f"{'✅' if eval_result.returncode == 0 else '❌'} Evaluation {'complete' if eval_result.returncode == 0 else 'FAILED'}"
)

# %% [markdown]
# ## Cell 10 — Router threshold tuning

# %%
router_result = subprocess.run(  # noqa: E402
    [sys.executable, f"{REPO}/phases/phase_8_router.py", "--dataset", DATASET_VERSION],
    cwd=REPO,
    text=True,
)
print(
    f"{'✅' if router_result.returncode == 0 else '❌'} Router tuning {'complete' if router_result.returncode == 0 else 'FAILED'}"
)

test_result = subprocess.run(
    [
        sys.executable,
        f"{REPO}/phases/phase_8_router.py",
        "--dataset",
        DATASET_VERSION,
        "--test",
        "nearest branch to me",
    ],
    cwd=REPO,
    text=True,
)

# %% [markdown]
# ## Cell 11 — Show outputs + download instructions
#
# DOWNLOAD: Go to Kaggle session → Outputs tab (right sidebar) →
# find /kaggle/working/experiments/ → click ⋮ → Download zip.
#
# LOCALLY after unzip:
#   cp -r /tmp/kaggle_out/experiments/* ./experiments/
#
# COMMIT RESULTS (not models):
#   git add experiments/v1/master_comparison.csv
#   git add experiments/v1/cost_simulation.json
#   git add experiments/v1/eda_plots/
#   git add experiments/v1/classical/results/
#   git add experiments/v1/transformers/results/
#   git commit -m "Add v1 results" && git push
#
# PLACE MODELS LOCALLY (too large for git):
#   Classical:   experiments/v1/classical/models/{experiment_id}/model.pkl
#   Transformer: experiments/v1/transformers/models/xlmr-base/

# %%
listing = subprocess.run(  # noqa: E402
    [
        "find",
        EXPERIMENT_DIR,
        "-type",
        "f",
        "(",
        "-name",
        "*.json",
        "-o",
        "-name",
        "*.csv",
        "-o",
        "-name",
        "*.pkl",
        ")",
    ],
    capture_output=True,
    text=True,
)
print("Key output files in /kaggle/working/experiments/:\n")
print(listing.stdout[:4000] or "(no files yet)")

disk_usage = subprocess.run(["du", "-sh", EXPERIMENT_DIR], capture_output=True, text=True)
print(f"\nTotal size: {disk_usage.stdout.split()[0] if disk_usage.stdout else 'unknown'}")
