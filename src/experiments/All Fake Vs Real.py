"""
exp1_binary.py — Experiment 1: Binary Baseline (Real vs All Fake).
Trains models on all generators combined and evaluates binary classification performance.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import pandas as pd
import numpy as np

try:
    from src.experiments.data_loader import load_subset
    from src.experiments.models import get_models
    from src.experiments.metrics import evaluate_binary, plot_confusion_matrix
except ImportError:
    from .data_loader import load_subset
    from .models import get_models
    from .metrics import evaluate_binary, plot_confusion_matrix

DEFAULT_OUTPUT_DIR = Path("outputs/results/exp1_binary")


def run_exp1(
    features_root: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    feature_set: str = "combined"
) -> Dict[str, Any]:
    """
    Run Experiment 1: Binary baseline evaluation.

    Parameters
    ----------
    features_root : Path or str, optional
        Path to features root directory.
    output_dir : Path or str, optional
        Path to save CSV tables and confusion matrices.
    feature_set : str
        Feature set to evaluate (default "combined").

    Returns
    -------
    dict
        Contains results_df, best_model_name, best_model, and output_dir.
    """
    out_path = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    out_path.mkdir(parents=True, exist_ok=True)

    print("Loading data for Exp 1 (All Generators)...")
    X_train, y_train, _ = load_subset("train", feature_set=feature_set, features_root=features_root)
    X_val, y_val, _     = load_subset("val",   feature_set=feature_set, features_root=features_root)
    X_test, y_test, _   = load_subset("test",  feature_set=feature_set, features_root=features_root)

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    models = get_models()
    results = []

    best_model_name = None
    best_val_f1 = -1.0
    best_model = None

    for name, pipeline in models.items():
        print(f"\n--- Training {name} ---")
        pipeline.fit(X_train, y_train)

        # Validation evaluation
        y_val_pred = pipeline.predict(X_val)
        y_val_prob = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else None
        val_metrics = evaluate_binary(y_val, y_val_pred, y_val_prob)

        # Test evaluation
        y_test_pred = pipeline.predict(X_test)
        y_test_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        test_metrics = evaluate_binary(y_test, y_test_pred, y_test_prob)

        print(f"[{name}] Val F1: {val_metrics['f1']:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | Test F1: {test_metrics['f1']:.4f}")

        # Save confusion matrix for test set
        cm_path = out_path / f"confusion_matrix_{name}.png"
        plot_confusion_matrix(y_test, y_test_pred, f"Exp 1: {name} Confusion Matrix (Test)", cm_path)

        results.append({
            "model": name,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics.get("roc_auc", np.nan),
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics.get("roc_auc", np.nan),
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_name = name
            best_model = pipeline

    df_results = pd.DataFrame(results)
    csv_path = out_path / "exp1_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved Experiment 1 results to {csv_path}")
    print(f"Best model based on Validation F1: {best_model_name} (Val F1: {best_val_f1:.4f})")

    return {
        "results_df": df_results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "output_dir": str(out_path)
    }


if __name__ == "__main__":
    run_exp1()
