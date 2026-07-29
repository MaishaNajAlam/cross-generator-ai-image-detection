"""
metrics.py — Evaluation metrics and visualization helper utilities.
"""

from pathlib import Path
from typing import Dict, Optional, Union, Sequence
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0=Real, 1=Fake).
    y_pred : array-like
        Predicted binary labels.
    y_prob : array-like, optional
        Predicted probabilities for class 1 (Fake).

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, f1, and optionally roc_auc.
    """
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            result["roc_auc"] = float("nan")
    return result


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Union[str, Path]
) -> None:
    """
    Plot and save a confusion matrix heatmap.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    if HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            ax=ax,
        )
    else:
        cax = ax.matshow(cm, cmap="Blues")
        fig.colorbar(cax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="red")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Real", "Fake"])
        ax.set_yticklabels(["Real", "Fake"])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_matrix(
    matrix: np.ndarray,
    generators: Sequence[str],
    metric_name: str,
    save_path: Union[str, Path]
) -> None:
    """
    Plot and save the 5x5 cross-generator transfer matrix heatmap.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))

    if HAS_SEABORN:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            xticklabels=list(generators),
            yticklabels=list(generators),
            ax=ax,
            vmin=0,
            vmax=1,
        )
    else:
        cax = ax.matshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)
        fig.colorbar(cax)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="black")
        ax.set_xticks(range(len(generators)))
        ax.set_yticks(range(len(generators)))
        ax.set_xticklabels(list(generators), rotation=45)
        ax.set_yticklabels(list(generators))

    ax.set_xlabel("Test Generator")
    ax.set_ylabel("Train Generator")
    ax.set_title(f"Cross-Generator Transfer Matrix ({metric_name})")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
