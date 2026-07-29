"""
exp2_diagonal.py — Experiment 2: Within-Generator Performance (Diagonal).
Trains and evaluates models on matching train/test generator pairs (Real + Gen_X).
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from src.config import GENERATORS
except ImportError:
    try:
        from config import GENERATORS
    except ImportError:
        GENERATORS = ["Real", "SD21", "SDXL", "SD3", "DALLE3", "Midjourney"]

try:
    from src.experiments.data_loader import load_subset
    from src.experiments.models import get_models
    from src.experiments.metrics import evaluate_binary
except ImportError:
    from .data_loader import load_subset
    from .models import get_models
    from .metrics import evaluate_binary

DEFAULT_OUTPUT_DIR = Path("outputs/results/exp2_diagonal")
FAKE_GENERATORS = [g for g in GENERATORS if g != "Real"]


def run_exp2(
    features_root: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    model_name: str = "LogisticRegression",
    feature_set: str = "combined"
) -> Dict[str, Any]:
    """
    Run Experiment 2: Within-generator evaluation.

    Parameters
    ----------
    features_root : Path or str, optional
        Path to features directory.
    output_dir : Path or str, optional
        Path to output directory.
    model_name : str
        Model to evaluate (defaults to LogisticRegression if LightGBM not installed).
    feature_set : str
        Feature set to use (defaults to 'combined').

    Returns
    -------
    dict
        Contains results_df and output_dir.
    """
    out_path = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    out_path.mkdir(parents=True, exist_ok=True)

    results = []

    for gen in FAKE_GENERATORS:
        print(f"\n--- Running Exp 2 for Generator: Real vs {gen} ---")
        X_train, y_train, _ = load_subset("train", generators=["Real", gen], feature_set=feature_set, features_root=features_root)
        X_test, y_test, _   = load_subset("test",  generators=["Real", gen], feature_set=feature_set, features_root=features_root)

        print(f"Train size: {len(y_train)} ({sum(y_train==0)} Real / {sum(y_train==1)} {gen})")
        print(f"Test size:  {len(y_test)} ({sum(y_test==0)} Real / {sum(y_test==1)} {gen})")

        models = get_models()
        target_model = model_name if model_name in models else list(models.keys())[0]

        pipeline = models[target_model]
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        metrics = evaluate_binary(y_test, y_pred, y_prob)

        print(f"[{gen}] Model: {target_model} | Acc: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

        results.append({
            "generator": gen,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics.get("roc_auc", np.nan),
        })

    df_results = pd.DataFrame(results)
    csv_path = out_path / "exp2_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved Experiment 2 results to {csv_path}")

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    if HAS_SEABORN:
        sns.barplot(data=df_results, x="generator", y="f1", palette="viridis", ax=ax)
    else:
        ax.bar(df_results["generator"], df_results["f1"], color="teal")

    ax.set_ylim(0, 1.05)
    ax.set_title(f"Within-Generator Classification F1 Score")
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Fake Generator")
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

    chart_path = out_path / "diagonal_bar_chart.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar chart to {chart_path}")

    return {
        "results_df": df_results,
        "output_dir": str(out_path)
    }


if __name__ == "__main__":
    run_exp2()
