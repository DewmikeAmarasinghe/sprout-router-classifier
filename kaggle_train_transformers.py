"""
Sprout Router — Full Training Pipeline on Kaggle GPU.

REPO path is /kaggle/working/sprout-router-classifier to match path_resolver.py.
Each transformer model has its own cell so you can see per-model output clearly.

Session planning (T4 x2, 12-hour limit):
  Cell 5  — Classical (~20 min CPU)
  Cell 6  — xlmr-base (~50 min T4)
  Cell 7  — papluca   (~45 min T4)
  Cell 8  — muril     (~55 min T4)
  Cell 9  — mbert     (~50 min T4)
  Cell 10 — xlmr-large (~100 min T4, only if needed)
  Total classical + 4 models: ~220 min / ~3.7 hours (well within 12-hour limit)
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
# ## Cell 2 — Clone repo
#
# CRITICAL: Clone to /kaggle/working/sprout-router-classifier to match path_resolver.py.
# path_resolver.py sets BASE_DIR = "/kaggle/working/sprout-router-classifier" for Kaggle.

# %%
try:  # noqa: E402
    from kaggle_secrets import UserSecretsClient

    pat = UserSecretsClient().get_secret("GITHUB_PAT")
except ImportError:
    pat = os.environ.get("GITHUB_PAT", "")

REPO = "/kaggle/working/sprout-router-classifier"
DATASET_VERSION = "v1"

url = f"https://oauth2:{pat}@github.com/DewmikeAmarasinghe/sprout-router-classifier.git"
pat = ""

if os.path.exists(f"{REPO}/.git"):
    print("Repo exists — pulling latest...")
    subprocess.run(["git", "-C", REPO, "pull", "--rebase"], check=True)
else:
    print("Cloning repo...")
    subprocess.run(["git", "clone", url, REPO], check=True)

url = ""
print(f"✅ Repo ready at {REPO}")

# %% [markdown]
# ## Cell 3 — Install dependencies via uv sync

# %%
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)  # noqa: E402
subprocess.run(["uv", "sync"], cwd=REPO, check=True)
print("✅ Dependencies installed via uv sync")

# %% [markdown]
# ## Cell 4 — Verify training data
#
# path_resolver.py auto-detects Kaggle and uses BASE_DIR = /kaggle/working/sprout-router-classifier.
# Train/val/test CSVs must be committed to the repo. No manual CSV reading needed.

# %%
sys.path.insert(0, f"{REPO}/src")  # noqa: E402

from backend.shared.path_resolver import get_dataset_path  # noqa: E402

dataset_dir = get_dataset_path(DATASET_VERSION)
for split in ("train.csv", "val.csv", "test.csv"):
    path = dataset_dir / split
    if not path.exists():
        raise FileNotFoundError(
            f"{split} not found at {path}.\n"
            "Commit train/val/test CSVs: git add data/datasets/v1/*.csv && git push"
        )
    print(f"✅ {split}  ({path.stat().st_size // 1024:,} KB)")

# %% [markdown]
# ## Cell 5 — Classical ML training
#
# Trains all 13 (vectorizer × classifier) combos sequentially — ~20 min CPU.
# Then auto-HPO on the best model (Optuna, 10 trials, ~5 min extra).

# %%
print("Training all classical ML combos + auto-HPO on best model...\n")  # noqa: E402

classical_result = subprocess.run(
    [
        sys.executable,
        f"{REPO}/phases/phase_5_train_classical.py",
        "--all",
        "--dataset",
        DATASET_VERSION,
        "--n-trials",
        "10",
    ],
    cwd=REPO,
    text=True,
)
print(
    f"{'✅' if classical_result.returncode == 0 else '❌'} Classical {'complete' if classical_result.returncode == 0 else 'FAILED'}"
)

# %% [markdown]
# ## Cell 6 — xlmr-base (~50 min)
#
# START HERE. Best for Sinhala/Tanglish romanized text.
# XLM-RoBERTa base (125M params), 3 epochs, lr=2e-5.
# Full fine-tuning: all layers trained, not just the head.

# %%
print("Training xlmr-base (3 epochs, full fine-tuning)...")  # noqa: E402
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}\n")

xlmr_base_result = subprocess.run(
    [
        sys.executable,
        f"{REPO}/phases/phase_6_train_transformers.py",
        "--model",
        "xlmr-base",
        "--dataset",
        DATASET_VERSION,
    ],
    cwd=REPO,
    text=True,
)
print(
    f"{'✅' if xlmr_base_result.returncode == 0 else '❌'} xlmr-base {'complete' if xlmr_base_result.returncode == 0 else 'FAILED'}"
)

# %% [markdown]
# ## Cell 7 — papluca (~45 min)
#
# XLM-RoBERTa pre-finetuned on 20-language detection.
# Warm-started on language signal — may need fewer epochs for the same performance.

# %%
print("Training papluca...")  # noqa: E402

papluca_result = subprocess.run(
    [
        sys.executable,
        f"{REPO}/phases/phase_6_train_transformers.py",
        "--model",
        "papluca",
        "--dataset",
        DATASET_VERSION,
    ],
    cwd=REPO,
    text=True,
)
print(
    f"{'✅' if papluca_result.returncode == 0 else '❌'} papluca {'complete' if papluca_result.returncode == 0 else 'FAILED'}"
)

# %% [markdown]
# ## Cell 8 — muril (~55 min)
#
# Google MuRIL — trained on 17 South Asian languages including transliterated Tamil.
# Best for Tanglish; complements xlmr-base (ensemble potential).

# %%
print("Training muril...")  # noqa: E402

muril_result = subprocess.run(
    [
        sys.executable,
        f"{REPO}/phases/phase_6_train_transformers.py",
        "--model",
        "muril",
        "--dataset",
        DATASET_VERSION,
    ],
    cwd=REPO,
    text=True,
)
print(
    f"{'✅' if muril_result.returncode == 0 else '❌'} muril {'complete' if muril_result.returncode == 0 else 'FAILED'}"
)

# %% [markdown]
# ## Cell 9 — mbert (~50 min)
#
# mBERT (bert-base-multilingual-cased) — baseline multilingual BERT.
# Weaker than XLM-R on Sinhala, weaker than MuRIL on Tamil.
# Keep for documentation: measures improvement over baseline.

# %%
print("Training mbert...")  # noqa: E402

mbert_result = subprocess.run(
    [
        sys.executable,
        f"{REPO}/phases/phase_6_train_transformers.py",
        "--model",
        "mbert",
        "--dataset",
        DATASET_VERSION,
    ],
    cwd=REPO,
    text=True,
)
print(
    f"{'✅' if mbert_result.returncode == 0 else '❌'} mbert {'complete' if mbert_result.returncode == 0 else 'FAILED'}"
)

# %% [markdown]
# ## Cell 10 — xlmr-large (~100 min) — OPTIONAL
#
# Only run if xlmr-base recall_1 < 0.97 after Cell 8 evaluation.
# 560M params, 4× slower. Uncomment to run.

# %%
# xlmr_large_result = subprocess.run(  # noqa: E402
#     [
#         sys.executable, f"{REPO}/phases/phase_6_train_transformers.py",
#         "--model", "xlmr-large", "--dataset", DATASET_VERSION,
#     ],
#     cwd=REPO, text=True,
# )
# print(f"{'✅' if xlmr_large_result.returncode == 0 else '❌'} xlmr-large {'complete' if xlmr_large_result.returncode == 0 else 'FAILED'}")
print("xlmr-large skipped (run only if xlmr-base recall_1 < 0.97)")

# %% [markdown]
# ## Cell 11 — Evaluation

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
# ## Cell 12 — Router threshold tuning

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
# ## Cell 13 — Show outputs + download instructions
#
# DOWNLOAD: Outputs tab → /kaggle/working/sprout-router-classifier/experiments/ → ⋮ → Download zip
#
# LOCALLY after unzip:
#   cp -r /tmp/kaggle_out/experiments/* ./experiments/
#
# COMMIT RESULTS (not models):
#   git add experiments/v1/master_comparison.csv experiments/v1/cost_simulation.json
#   git add experiments/v1/eda_plots/ experiments/v1/classical/results/
#   git add experiments/v1/transformers/results/
#   git commit -m "Add v1 results" && git push
#
# PLACE MODELS (download from zip → copy manually):
#   Classical:   experiments/v1/classical/models/{experiment_id}/model.pkl
#   Transformer: experiments/v1/transformers/models/xlmr-base/

# %%
from backend.shared.path_resolver import EXPERIMENTS  # noqa: E402

listing = subprocess.run(
    ["find", str(EXPERIMENTS / DATASET_VERSION), "-name", "result.json"],
    capture_output=True,
    text=True,
)
print("result.json files found:\n")
print(listing.stdout or "(none yet)")

disk_usage = subprocess.run(["du", "-sh", str(EXPERIMENTS)], capture_output=True, text=True)
print(
    f"\nTotal experiments size: {disk_usage.stdout.split()[0] if disk_usage.stdout else 'unknown'}"
)
