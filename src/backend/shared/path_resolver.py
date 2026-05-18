"""
Path resolver — environment-aware path helpers.

Auto-detects Kaggle / Colab / local environment.
All scripts import from here — never hardcode paths elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

IS_KAGGLE: bool = os.path.exists("/kaggle/input")
IS_COLAB: bool = os.path.exists("/content/drive")
IS_LOCAL: bool = not IS_KAGGLE and not IS_COLAB

if IS_KAGGLE:
    BASE_DIR = Path("/kaggle/working/sprout-router-classifier")
elif IS_COLAB:
    BASE_DIR = Path("/content/sprout-router-classifier")
else:
    # Walk up from this file: shared/ → backend/ → src/ → project root
    BASE_DIR = Path(__file__).parent.parent.parent.parent

DATA_DIR: Path = BASE_DIR / "data"
EXPERIMENTS: Path = BASE_DIR / "experiments"
RESULTS_DIR: Path = BASE_DIR / "results"
MLFLOW_URI: str = str(BASE_DIR / "mlruns")


def get_dataset_path(name: str) -> Path:
    """Return path to data/datasets/{name}/. Creates it if missing."""
    p = DATA_DIR / "datasets" / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_experiment_path(dataset_name: str, approach: str) -> Path:
    """Return path to experiments/{dataset_name}/{approach}/.

    approach: "classical", "transformers", or "eda_plots"
    """
    p = EXPERIMENTS / dataset_name / approach
    p.mkdir(parents=True, exist_ok=True)
    return p


def discover_datasets() -> list[str]:
    """Return sorted list of dataset folder names under data/datasets/."""
    d = DATA_DIR / "datasets"
    if not d.exists():
        return []
    return sorted(f.name for f in d.iterdir() if f.is_dir())
