"""
Evaluation plots — ROC curve and confusion matrix per model.

Both functions save a PNG to the model's output directory and return the path.
They also log the figure to MLflow if an active run exists.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

log = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    output_dir: Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Save a confusion matrix PNG.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        output_dir: Directory to save the PNG.
        title: Figure title (shown above the matrix).

    Returns:
        Path to the saved PNG file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_labels = ["0 (mini)", "1 (4o)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )

    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()

    out_path = output_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    try_mlflow_log_figure(fig, "confusion_matrix.png")
    log.info(f"  Confusion matrix saved → {out_path}")
    return out_path


def plot_roc_curve(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    roc_auc: float,
    output_dir: Path,
    title: str = "ROC Curve",
) -> Path:
    """Save a ROC curve PNG.

    Args:
        y_true: Ground-truth labels.
        y_proba: Predicted probabilities for label=1.
        roc_auc: Pre-computed ROC-AUC score (shown in legend).
        output_dir: Directory to save the PNG.
        title: Figure title.

    Returns:
        Path to the saved PNG file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=1)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, color="#1f77b4", label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], lw=1, color="grey", linestyle="--", label="Random")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()

    out_path = output_dir / "roc_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    try_mlflow_log_figure(fig, "roc_curve.png")
    log.info(f"  ROC curve saved → {out_path}")
    return out_path


def try_mlflow_log_figure(fig: object, artifact_name: str) -> None:
    """Log a matplotlib figure to MLflow if an active run exists."""
    try:
        import mlflow

        if mlflow.active_run():
            mlflow.log_figure(fig, artifact_name)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        pass
