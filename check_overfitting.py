"""
check_overfitting.py — Diagnostic script to evaluate overfitting risk across classifiers.
Supports both Google Colab (Google Drive paths) and local execution.
Compares Train F1 vs. Validation F1 vs. Test F1 and reports the Overfitting Gap.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Dual-mode path setup (Local vs Google Colab)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Detect Google Colab environment
IN_COLAB = "google.colab" in sys.modules

try:
    from src.config import FEATURE_ROOT, FEATURE_VERSION
    DEFAULT_FEATURES_ROOT = FEATURE_ROOT / FEATURE_VERSION
except ImportError:
    try:
        from config import FEATURE_ROOT, FEATURE_VERSION
        DEFAULT_FEATURES_ROOT = FEATURE_ROOT / FEATURE_VERSION
    except ImportError:
        # Fallback default path for Colab / Drive setup
        DEFAULT_FEATURES_ROOT = Path("/content/drive/MyDrive/ml_project/processed/features/v1")

try:
    from src.experiments.data_loader import load_subset
    from src.experiments.models import get_models
    from src.experiments.metrics import evaluate_binary
except ImportError:
    from experiments.data_loader import load_subset
    from experiments.models import get_models
    from experiments.metrics import evaluate_binary


def run_overfitting_check(features_root: Path = DEFAULT_FEATURES_ROOT):
    print("=" * 85)
    print("      OVERFITTING DIAGNOSTIC CHECK (Train vs Val vs Test Performance)      ")
    print("=" * 85)

    print(f"\nEnvironment: {'Google Colab' if IN_COLAB else 'Local'}")
    print(f"Features Root: {features_root}")

    print("\nLoading dataset splits...")
    try:
        X_train, y_train, _ = load_subset("train", feature_set="combined", features_root=features_root)
        X_val, y_val, _     = load_subset("val",   feature_set="combined", features_root=features_root)
        X_test, y_test, _   = load_subset("test",  feature_set="combined", features_root=features_root)
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset from {features_root}: {e}")
        print("\nIf running in Colab, ensure Google Drive is mounted and features_root is correctly specified:")
        print("  from check_overfitting import run_overfitting_check")
        print("  run_overfitting_check(features_root='/content/drive/MyDrive/path_to_features')")
        return

    print(f"Loaded successfully:")
    print(f"  • Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  • Val:   {X_val.shape[0]} samples, {X_val.shape[1]} features")
    print(f"  • Test:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"  • Feature-to-Sample Ratio (Train): {X_train.shape[1] / X_train.shape[0]:.2f} features per sample")

    models = get_models()
    report_rows = []

    print("\nEvaluating classifiers for overfitting...")
    print("-" * 85)

    for name, pipeline in models.items():
        print(f"Training {name}...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)
        print("Done!")

        # 1. Train Evaluation
        y_train_pred = pipeline.predict(X_train)
        y_train_prob = pipeline.predict_proba(X_train)[:, 1] if hasattr(pipeline, "predict_proba") else None
        train_m = evaluate_binary(y_train, y_train_pred, y_train_prob)

        # 2. Validation Evaluation
        y_val_pred = pipeline.predict(X_val)
        y_val_prob = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else None
        val_m = evaluate_binary(y_val, y_val_pred, y_val_prob)

        # 3. Test Evaluation
        y_test_pred = pipeline.predict(X_test)
        y_test_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        test_m = evaluate_binary(y_test, y_test_pred, y_test_prob)

        # Calculate gaps
        train_val_gap = train_m["f1"] - val_m["f1"]
        train_test_gap = train_m["f1"] - test_m["f1"]

        # Overfitting Risk Assessment
        if train_val_gap > 0.10:
            status = "🔴 SEVERE OVERFITTING"
        elif train_val_gap > 0.05:
            status = "🟡 MODERATE OVERFITTING"
        else:
            status = "🟢 GOOD GENERALIZATION"

        report_rows.append({
            "Model": name,
            "Train Acc": f"{train_m['accuracy']:.4f}",
            "Train F1": f"{train_m['f1']:.4f}",
            "Val Acc": f"{val_m['accuracy']:.4f}",
            "Val F1": f"{val_m['f1']:.4f}",
            "Test Acc": f"{test_m['accuracy']:.4f}",
            "Test F1": f"{test_m['f1']:.4f}",
            "Train-Val F1 Gap": f"{train_val_gap:+.4f}",
            "Status": status
        })

    # Display Report Table
    df_report = pd.DataFrame(report_rows)
    print("\n" + "=" * 85)
    print("                        OVERFITTING DIAGNOSTIC SUMMARY                       ")
    print("=" * 85)
    print(df_report.to_string(index=False))
    print("=" * 85)

    print("\nInterpretation Guide:")
    print("  • Train-Val F1 Gap > 0.10  ==> Model has memorized training noise (High Variance / Overfitting)")
    print("  • Train-Val F1 Gap <= 0.05 ==> Model generalizes well between train and validation splits")
    print("  • If gap is large, try reducing feature dimension (PCA/Mutual Info) or applying max_depth regularization.")

    return df_report


if __name__ == "__main__":
    run_overfitting_check()
