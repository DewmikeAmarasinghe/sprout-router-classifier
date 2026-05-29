"""
Full Sprout Router Training Pipeline — Kaggle GPU.

Pipeline:
    Cell  1  Environment check
    Cell  2  Clone repo
    Cell  3  Install dependencies
    Cell  4  [OFF] Data generation (phases 1-4 — uncomment only when regenerating dataset)
    Cell  5  Verify training data  +  define run_model()
    Cell  6  Phase 5 — Classical ML training          (CPU, ~30 min, ACTIVE_COMBOS only)
    Cell  7  Phase 6 — xlmr-base transformer          (~80 min on T4 @ 5 epochs)
    Cell  8  Phase 6 — papluca transformer
    Cell  9  Phase 6 — muril transformer
    Cell 10  Phase 6 — mbert transformer
    Cell 11  Phase 6 — xlmr-large transformer         [OFF — commented out]
    Cell 12  Phase 7 — Evaluate all models
    Cell 13  Phase 8 — Router threshold tuning
    Cell 14  Output summary
    Cell 15  Download results

DATASET: v4 (50/50 label=0/label=1, 2 scenarios: simple_transactional + named_location)
EPOCHS:  5 (was 3). With 50/50 balance and only 2 scenarios, the distinction is more
         nuanced (named_location in pure_english looks superficially like simple_transactional).
         load_best_model_at_end=True ensures early stopping with no accuracy penalty.

REPO path: /kaggle/working/sprout-router-classifier (matches path_resolver.py).

WHY subprocess AND NOT ! magic:
    This file uses # %% Jupyter cell markers and runs as a .py notebook.
    subprocess.run() without capture_output streams output in real time.
    subprocess.Popen() + readline keeps Python actively executing, which prevents
    Kaggle's 40-min idle timeout during long training runs.

Checkpoints:
    HuggingFace Trainer writes checkpoint-N/ dirs after each epoch (~500 MB each).
    trainer.py's cleanup_checkpoints() removes them after trainer.save_model().
    save_total_limit=2 and save_only_model=True in TRAIN_CONFIG limit accumulation.
    Classical .pkl files are small (5-100 MB each) and never cause disk pressure.
"""

# %% [markdown]
# ## Cell 1 — Environment check

# %%
import os
import shutil
import subprocess
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_PLATFORMS"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

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
# Clones to /kaggle/working/sprout-router-classifier — matches path_resolver.py exactly.

# %%
REPO = "/kaggle/working/sprout-router-classifier"
DATASET_VERSION = "v1"
REPO_URL = "https://github.com/DewmikeAmarasinghe/sprout-router-classifier.git"

if os.path.exists(f"{REPO}/.git"):
    print("Repo exists — pulling latest...")
    subprocess.run(["git", "-C", REPO, "pull", "--rebase"], check=True)
else:
    print("Cloning repo...")
    subprocess.run(["git", "clone", REPO_URL, REPO], check=True)

print(f"✅ Repo ready at {REPO}")

# %% [markdown]
# ## Cell 3 — Install dependencies

# %%
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
subprocess.run(
    ["uv", "pip", "install", "-r", "pyproject.toml", "--system"],
    cwd=REPO,
    check=True,
)
print("✅ Dependencies installed via uv")

# %% [markdown]
# ## Cell 4 — Data generation  [OFF — uncomment only when regenerating the dataset]
#
# These phases generate the training dataset from scratch using the OpenAI API.
# The current dataset is already committed to git and does NOT need to be regenerated.
#
# Prerequisites to uncomment:
#   1. Add OPENAI_API_KEY to Kaggle Secrets (Settings → Secrets → Add new secret)
#   2. Uncomment the entire block below and run
#
# Phase 1 — Grounding (~5 min, 1k API calls):
#   Verifies is_pure_script() catches Sinhala/Tamil unicode correctly.
#
# Phase 2 — Generation (hours, many API calls):
#   Generates ~60k training messages via gpt-5-nano.
#   Run locally or in a separate long-running Kaggle session.
#
# Phase 3 — Split (~1 min, no API):
#   Stratified 80/10/10 split → train.csv / val.csv / test.csv
#
# Phase 4 — EDA (~2 min, no API):
#   Generates distribution plots and statistics to data/eda_plots/.

# %%
# from kaggle_secrets import UserSecretsClient
# openai_key = UserSecretsClient().get_secret("OPENAI_API_KEY")
# _gen_env = {**os.environ, "PYTHONPATH": f"{REPO}/src", "OPENAI_API_KEY": openai_key}
# openai_key = ""
#
# subprocess.run([sys.executable, f"{REPO}/phases/phase_1_grounding.py", "--force"],
#                cwd=REPO, env=_gen_env, text=True, check=True)
#
# subprocess.run([sys.executable, f"{REPO}/phases/phase_2_generate.py", "--workers", "7"],
#                cwd=REPO, env=_gen_env, text=True)
#
# subprocess.run([sys.executable, f"{REPO}/phases/phase_3_split.py"],
#                cwd=REPO, env={**os.environ, "PYTHONPATH": f"{REPO}/src"}, text=True, check=True)
#
# subprocess.run([sys.executable, f"{REPO}/phases/phase_4_eda.py"],
#                cwd=REPO, env={**os.environ, "PYTHONPATH": f"{REPO}/src"}, text=True, check=True)
print("Cell 4 is OFF — data generation phases are commented out.")

# %% [markdown]
# ## Cell 5 — Verify training data  +  helpers

# %%
sys.path.insert(0, f"{REPO}/src")

import pandas as pd

from backend.shared.path_resolver import get_dataset_path, get_experiment_path

dataset_dir = get_dataset_path(DATASET_VERSION)
for split in ("train.csv", "val.csv", "test.csv"):
    path = dataset_dir / split
    if not path.exists():
        raise FileNotFoundError(
            f"{split} not found at {path}.\n"
            "Run phase_3_split.py or git push the CSV files before running this notebook."
        )
    df = pd.read_csv(path)
    label_0 = int((df["label"] == 0).sum())
    label_1 = int((df["label"] == 1).sum())
    print(f"✅ {split:10s}  {len(df):,} rows  label=0: {label_0:,}  label=1: {label_1:,}")


def disk_usage() -> str:
    total, used, free = shutil.disk_usage("/kaggle/working")
    return f"{used / 1e9:.1f} GB used / {total / 1e9:.1f} GB total ({free / 1e9:.1f} GB free)"


# PYTHONPATH: child processes can import backend.* without inheriting parent sys.path.
SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": f"{REPO}/src"}


def run_model(model_key: str) -> int:
    """Train one transformer model in a subprocess, streaming output line by line.

    subprocess.run() leaves the Python kernel IDLE (waiting for subprocess),
    which triggers Kaggle's 40-min idle timeout for long runs.
    Popen + readline keeps Python actively reading/printing, preventing that.

    GPU memory is released when the subprocess exits.
    Checkpoints are cleaned up inside trainer.py (cleanup_checkpoints).
    Final weights + result.json stay in experiments/{DATASET}/transformers/models/{key}/.
    """
    cmd = [
        sys.executable,
        f"{REPO}/phases/phase_6_train_transformers.py",
        "--model",
        model_key,
        "--dataset",
        DATASET_VERSION,
        "--no-hpo",
    ]

    print(f"\n[before {model_key}] Disk: {disk_usage()}", flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=REPO,
        env=SUBPROCESS_ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    print(f"[after  {model_key}] Disk: {disk_usage()}", flush=True)
    print(
        f"{'✅' if proc.returncode == 0 else '❌'} {model_key} {'complete' if proc.returncode == 0 else 'FAILED'}",
        flush=True,
    )
    return proc.returncode


# %% [markdown]
# ## Cell 6 — Phase 5: Classical ML training  (CPU, ~30 min)
#
# Trains ACTIVE_COMBOS only (14 curated vectorizer × classifier pairs).
# Use --all for all 25 combinations (~60-90 min).
#
# Auto-HPO runs on the best passing model (highest MCC among recall_1 ≥ threshold).
# Results saved to experiments/{DATASET}/classical/models/{experiment_id}/result.json
# CPU-only — no GPU needed.

# %%
print(f"\n[before classical] Disk: {disk_usage()}", flush=True)

proc_classical = subprocess.Popen(
    [sys.executable, f"{REPO}/phases/phase_5_train_classical.py", "--all-active"],
    cwd=REPO,
    env=SUBPROCESS_ENV,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
assert proc_classical.stdout is not None
for line in proc_classical.stdout:
    print(line, end="", flush=True)
proc_classical.wait()

print(f"[after  classical] Disk: {disk_usage()}", flush=True)
print(
    f"{'✅' if proc_classical.returncode == 0 else '❌'} Classical ML {'complete' if proc_classical.returncode == 0 else 'FAILED'}",
    flush=True,
)

# %% [markdown]
# ## Cell 7 — Phase 6: xlmr-base
#
# XLM-RoBERTa base (278M params, 1,000 MB, min 8 GB VRAM).
# Best for Sinhala — empirically verified (U. of Moratuwa 2022).
# Full fine-tuning: all weights, 5 epochs, lr=2e-5. ~80 min on Kaggle T4.

# %%
run_model("xlmr-base")

# %% [markdown]
# ## Cell 8 — Phase 6: papluca
#
# XLM-RoBERTa base pre-finetuned on 20-language detection (278M params, 1,000 MB, min 8 GB VRAM).
# Warm-started on language signal — may need fewer epochs.

# %%
run_model("papluca")

# %% [markdown]
# ## Cell 9 — Phase 6: muril
#
# Google MuRIL (278M params, 950 MB, min 8 GB VRAM).
# Trained on 17 South Asian languages including transliterated Tamil.
# Best for Tanglish; complement to xlmr-base.

# %%
run_model("muril")

# %% [markdown]
# ## Cell 10 — Phase 6: mbert
#
# bert-base-multilingual-cased (178M params, 700 MB, min 6 GB VRAM).
# Older multilingual BERT baseline. Weaker than XLM-R on Sinhala.

# %%
run_model("mbert")

# %% [markdown]
# ## Cell 11 — Phase 6: xlmr-large  [OFF]
#
# XLM-RoBERTa large (560M params, 2,200 MB, min 12 GB VRAM).
# 24-layer; 4× compute of base. Uncomment when you want to compare against base models.
# 5 epochs × 4× compute = ~160 min on T4.

# %%
run_model("xlmr-large")

# %% [markdown]
# ## Cell 12 — Phase 7: Evaluate all models
#
# Loads all result.json files from both classical/ and transformers/, builds the unified
# comparison table, runs cost simulation, and runs error analysis on the best model.

# %%
result_eval = subprocess.run(
    [sys.executable, f"{REPO}/phases/phase_7_evaluate.py", "--dataset", DATASET_VERSION],
    cwd=REPO,
    env=SUBPROCESS_ENV,
    text=True,
)
print(
    f"{'✅' if result_eval.returncode == 0 else '❌'} Evaluation {'complete' if result_eval.returncode == 0 else 'FAILED'}"
)

# %% [markdown]
# ## Cell 13 — Phase 8: Router threshold tuning
#
# Sweeps confidence thresholds on val.csv using the best model.
# Finds the highest threshold where recall_1 >= 0.95.
# Saves optimal threshold to experiments/{DATASET}/router/threshold_curve.json.
# Copy the printed CONFIDENCE_THRESHOLD value to src/backend/config/settings.py.

# %%
result_router = subprocess.run(
    [sys.executable, f"{REPO}/phases/phase_8_router.py", "--dataset", DATASET_VERSION],
    cwd=REPO,
    env=SUBPROCESS_ENV,
    text=True,
)
print(
    f"{'✅' if result_router.returncode == 0 else '❌'} Router tuning {'complete' if result_router.returncode == 0 else 'FAILED'}"
)

for test_msg in ("nearest branch to me", "meka kohomada"):
    subprocess.run(
        [
            sys.executable,
            f"{REPO}/phases/phase_8_router.py",
            "--dataset",
            DATASET_VERSION,
            "--test",
            test_msg,
        ],
        cwd=REPO,
        env=SUBPROCESS_ENV,
        text=True,
    )

# %% [markdown]
# ## Cell 14 — Output summary

# %%
from backend.shared.path_resolver import EXPERIMENTS

result_listing = subprocess.run(
    ["find", str(EXPERIMENTS / DATASET_VERSION), "-name", "result.json"],
    capture_output=True,
    text=True,
)
print("result.json files:\n")
print(result_listing.stdout or "(none — training may not have completed)")

result_du = subprocess.run(
    ["du", "-sh", str(EXPERIMENTS)],
    capture_output=True,
    text=True,
)
print(f"\nTotal experiments size: {result_du.stdout.split()[0] if result_du.stdout else 'unknown'}")
print(f"Disk: {disk_usage()}")

# %% [markdown]
# ## Cell 15 — Download results
#
# Zips experiments/ + mlflow.db into a single file and shows a download link.
# mlruns/ is excluded — its artifacts duplicate experiments/; mlflow.db alone is
# enough to run `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
#
# NOTE: FileLink must use a relative path (from /kaggle/working) — absolute paths
# return 404 from Kaggle's proxy.
#
# After downloading, extract locally:
#   unzip sprout_results.zip -d /tmp/kaggle_out
#   cp -r /tmp/kaggle_out/experiments/* ./experiments/
#   cp /tmp/kaggle_out/mlflow.db ./mlflow.db
#
# Commit small result files (model weights are gitignored):
#   git add experiments/v1/master_comparison.csv
#   git add experiments/v1/cost_simulation.json experiments/v1/error_analysis.json
#   git add experiments/v1/router/threshold_curve.json
#   git commit -m "Add v1 training results" && git push

# %%
import glob as _glob

ZIP_NAME = "sprout_results.zip"
ZIP_PATH = f"/kaggle/working/{ZIP_NAME}"

# Remove stale files from any previous run to reclaim disk space.
for stale in _glob.glob("/kaggle/working/sprout_results*"):
    os.remove(stale)
    print(f"Removed stale: {os.path.basename(stale)}", flush=True)

print(f"Disk before zip: {disk_usage()}", flush=True)
print(f"Zipping experiments/ + mlflow.db → {ZIP_PATH} ...", flush=True)

zip_proc = subprocess.Popen(
    ["zip", "-r", ZIP_PATH, "experiments/", "mlflow.db"],
    cwd=REPO,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
assert zip_proc.stdout is not None
for line in zip_proc.stdout:
    print(line, end="", flush=True)
zip_proc.wait()

if zip_proc.returncode == 0 and os.path.exists(ZIP_PATH):
    size_mb = os.path.getsize(ZIP_PATH) / 1e6
    print(f"\n✅ Zip created: {size_mb:.0f} MB. Disk: {disk_usage()}", flush=True)
else:
    print(f"\n❌ Zip failed (exit {zip_proc.returncode}). Disk: {disk_usage()}", flush=True)

from IPython.display import FileLink

# FileLink must use a path relative to /kaggle/working — absolute paths return 404.
os.chdir("/kaggle/working")
display(FileLink(ZIP_NAME))  # noqa: F821
