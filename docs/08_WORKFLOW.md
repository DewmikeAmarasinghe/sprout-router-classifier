# 08 — WORKFLOW (v4)
## VS Code + Kaggle/Colab GPU

---

## THE CORE IDEA

You write .py files in local VS Code like a normal Python project.
GPU-heavy phases (transformers) run on Kaggle or Colab.
CPU phases (classical ML, data generation) can run locally or on Kaggle.
Everything stays in sync through GitHub.

  Local VS Code                  GitHub Repo              Kaggle / Colab GPU
  ─────────────                  ────────────             ──────────────────
  Edit .py files    ──git push──► main branch ──clone──►  python train.py
  View MLflow UI    ◄──git pull── results/    ◄──push──   MLflow logs
  Use Gradio UI                                            Model checkpoints

---

## WHICH PHASE RUNS WHERE

  Phase                    Platform           Why
  ──────────────────────────────────────────────────────────────────────
  Phase 1: Data gen        Kaggle CPU         API calls to gpt-4o-mini, HF downloads
  Phase 1: Gazetteer       Local              One-time build, small files, no GPU
  Phase 2: EDA             Local              Fast, no GPU needed
  Phase 3: Classical ML    Local or Kaggle    TF-IDF + sklearn runs fine on CPU < 1hr
  Phase 4: Transformers    Kaggle GPU         T4 GPU required. 2× T4, 30h/week free
  Phase 4: HPO             Kaggle GPU         Multiple runs need GPU
  Phase 5: Evaluation      Local              Load saved results, generate plots
  Phase 6: Hybrid          Local              Pure Python logic, no training

---

## OPTION A — VS CODE REMOTE TUNNEL (recommended)

Edit files in LOCAL VS Code while they run on Kaggle/Colab GPU.
No window switching. One editor. Full terminal access.

Step 1 — Install VS Code extension (once ever):
  In VS Code: install "Remote - Tunnels" (Microsoft official)

Step 2 — Bootstrap cell in Kaggle/Colab (once per session):
  !pip install -q vscode-colab
  import vscode_colab
  vscode_colab.login()
  vscode_colab.connect(name="sprout-gpu")

Step 3 — Connect from VS Code:
  Ctrl+Shift+P → "Remote Tunnels: Connect to Tunnel..." → select "sprout-gpu"
  Open folder: /kaggle/working

Step 4 — Clone repo inside the session:
  cd /kaggle/working
  git clone https://github.com/YOUR_USERNAME/sprout-router-classifier.git
  cd sprout-router-classifier
  pip install -r requirements.txt

Step 5 — Launch Gradio from remote terminal:
  python launcher/app.py
  # Prints a public URL (share=True). Open in your local browser.
  # Trigger any phase from browser. Scripts execute on the GPU machine.

Step 6 — Sync back:
  bash sync.sh

---

## OPTION B — GIT SYNC ONLY (simpler)

1. Edit .py files locally
2. git push
3. In Kaggle terminal: git pull → python train.py
4. bash sync.sh to push results back

---

## GRADIO LAUNCHER — HOW TO USE

  Start:   python launcher/app.py
  Access:  Open the printed public URL in your local browser

  Data panel:
    Select Phase 1 sub-step from dropdown (or "Run all")
    Enter dataset name (default: v1_baseline)
    Set category, row count, batch size as needed
    Hit Run → live output streams in the output box

  Classical ML panel:
    Select dataset from dropdown (auto-scanned from data/datasets/)
    Select vectorizer and classifier
    Optional: paste JSON param overrides
    Choose action: Train single / Train all / HPO
    Hit Run → live output → MLflow link printed on completion

  Transformer panel:
    Select dataset from dropdown
    Select model from dropdown
    Optional: paste JSON param overrides
    Choose action: Train single / Train all / HPO / Inference+ONNX
    Hit Run → live output → MLflow link

  Evaluation panel:
    Choose: Compare all / Ablation / Cost sim / Error analysis
    Results render inline after completion

  Refresh button:
    Re-scans data/datasets/ and experiments/
    Updates all dropdowns to reflect new folders

---

## FILE PERSISTENCE STRATEGY

  What to persist         Where              How
  ─────────────────────────────────────────────────────────────────
  Code (.py files)        GitHub             git push after every edit
  MLflow runs (mlruns/)   GitHub             git add mlruns/ && git push
  Result CSVs             GitHub             git add results/ && git push
  Classical ML .pkl       GitHub             Small enough (< 50MB each)
  Transformer checkpoints Google Drive       500MB+, too large for GitHub
  data/datasets/          GitHub             CSVs are small enough for git

---

## sync.sh — RUN AT END OF EVERY KAGGLE SESSION

  #!/bin/bash
  set -e

  echo "Syncing code + results to GitHub..."
  git add mlruns/ results/ \
      experiments/v1_baseline/classical_ml/ \
      data/datasets/v1_baseline/train.csv \
      data/datasets/v1_baseline/val.csv \
      data/datasets/v1_baseline/test.csv \
      data/datasets/v1_baseline/quality_report.json
  git commit -m "sync: $(date '+%Y-%m-%d %H:%M')" || echo "Nothing to commit"
  git push origin main

  echo "Syncing transformer models to Google Drive..."
  cp -r experiments/v1_baseline/transformers/models/ \
      /content/drive/MyDrive/sprout-router/models/ 2>/dev/null || true

  echo "Done."

---

## ENVIRONMENT AUTO-DETECTION — shared/config.py

  import os

  IS_KAGGLE = os.path.exists("/kaggle/input")
  IS_COLAB  = os.path.exists("/content/drive")
  IS_LOCAL  = not IS_KAGGLE and not IS_COLAB

  if IS_KAGGLE:
      BASE_DIR = "/kaggle/working/sprout-router-classifier"
  elif IS_COLAB:
      BASE_DIR = "/content/sprout-router-classifier"
  else:
      BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

  DATA_DIR     = os.path.join(BASE_DIR, "data")
  EXPERIMENTS  = os.path.join(BASE_DIR, "experiments")
  RESULTS_DIR  = os.path.join(BASE_DIR, "results")
  MLFLOW_URI   = os.path.join(BASE_DIR, "mlruns")

  def get_dataset_path(name):
      return os.path.join(DATA_DIR, "datasets", name)

  def get_experiment_path(dataset_name, approach):
      return os.path.join(EXPERIMENTS, dataset_name, approach)

  def discover_datasets():
      d = os.path.join(DATA_DIR, "datasets")
      return [f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))]

  Every script imports from shared/config.py.
  Paths work on all three environments with zero changes.
  Nothing hardcoded anywhere else in the codebase.

---

## MLflow UI

  Local (after git pull of mlruns/):
    mlflow ui
    Open: http://localhost:5000

  On Kaggle (no localhost browser):
    Training scripts export CSV at end of each run:
    pd.DataFrame(rows).to_csv(results_path)
    OR open Gradio evaluation panel — renders comparison table inline.

---

## QUICK REFERENCE

  # Install tunnel extension
  code --install-extension ms-vscode.remote-tunnels

  # Kaggle bootstrap (once per session)
  !pip install -q vscode-colab
  import vscode_colab; vscode_colab.login(); vscode_colab.connect(name="sprout-gpu")

  # VS Code: Ctrl+Shift+P → Remote Tunnels → sprout-gpu

  # In Kaggle terminal (via VS Code):
  git clone https://github.com/YOUR/sprout-router-classifier.git
  cd sprout-router-classifier && pip install -r requirements.txt
  python launcher/app.py                            # open Gradio in browser

  # Direct CLI (alternative to Gradio):
  python phases/phase_3_classical_ml/train_single.py \
      --dataset-name v1_baseline --vec tfidf_char --clf svm

  # End of session
  bash sync.sh
