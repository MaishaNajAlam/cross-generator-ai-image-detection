"""
exp2_diagonal.py — Experiment 2: Within-Generator Performance (Diagonal).
Trains and evaluates models on matching train/test generator pairs (Real + Gen_X).
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any, List
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


def plot_individual_diagonal(df_model: pd.DataFrame, model_name: str, save_path: Path) -> None:
    """
    Plot and save a bar chart for a single model's diagonal results.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if HAS_SEABORN:
        sns.barplot(data=df_model, x="generator", y="f1", palette="viridis", ax=ax)
    else:
        ax.bar(df_model["generator"], df_model["f1"], color="teal")

    ax.set_ylim(0, 1.05)
    ax.set_title(f"Within-Generator Classification F1 Score ({model_name})")
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Fake Generator")
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_exp2(
    features_root: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    model_name: Union[str, List[str]] = "LogisticRegression",
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
    model_name : str or list of str
        Model(s) to evaluate. If "all", runs all available models.
    feature_set : str
        Feature set to use (defaults to 'combined').

    Returns
    -------
    dict
        Contains results_df and output_dir.
    """
    out_path = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    out_path.mkdir(parents=True, exist_ok=True)

    models = get_models()
    
    if model_name == "all":
        models_to_run = list(models.keys())
    elif isinstance(model_name, list):
        models_to_run = model_name
    else:
        models_to_run = [model_name]

    # Filter valid models
    valid_models_to_run = [m for m in models_to_run if m in models]
    if not valid_models_to_run:
        raise ValueError(f"None of the specified models {model_name} are available: {list(models.keys())}")

    results = []

    for gen in FAKE_GENERATORS:
        print(f"\n--- Running Exp 2 for Generator: Real vs {gen} ---")
        X_train, y_train, _ = load_subset("train", generators=["Real", gen], feature_set=feature_set, features_root=features_root)
        X_test, y_test, _   = load_subset("test",  generators=["Real", gen], feature_set=feature_set, features_root=features_root)

        print(f"Train size: {len(y_train)} ({sum(y_train==0)} Real / {sum(y_train==1)} {gen})")
        print(f"Test size:  {len(y_test)} ({sum(y_test==0)} Real / {sum(y_test==1)} {gen})")

        for m_name in valid_models_to_run:
            pipeline = models[m_name]
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
            metrics = evaluate_binary(y_test, y_pred, y_prob)

            print(f"[{gen} | {m_name}] Acc: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

            results.append({
                "model": m_name,
                "generator": gen,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics.get("roc_auc", np.nan),
            })

    df_results = pd.DataFrame(results)
    
    # Save results
    # 1. Combined CSV of all models
    csv_path = out_path / "exp2_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved combined Experiment 2 results to {csv_path}")

    # 2. Individual CSVs and individual plots
    for m_name in valid_models_to_run:
        df_model = df_results[df_results["model"] == m_name]
        model_dir = out_path / m_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual CSV (dropping model name column to match old format exactly)
        df_model_save = df_model.drop(columns=["model"])
        df_model_save.to_csv(model_dir / "exp2_results.csv", index=False)
        
        # Save individual bar chart
        plot_individual_diagonal(df_model, m_name, model_dir / "diagonal_bar_chart.png")
        
        # If it's the only model, or if we want a copy at the root level for backwards compatibility
        if len(valid_models_to_run) == 1 or m_name == valid_models_to_run[0]:
            plot_individual_diagonal(df_model, m_name, out_path / "diagonal_bar_chart.png")

    # 3. Combined comparison bar chart if multiple models
    if len(valid_models_to_run) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        if HAS_SEABORN:
            sns.barplot(data=df_results, x="generator", y="f1", hue="model", palette="viridis", ax=ax)
        else:
            x = np.arange(len(FAKE_GENERATORS))
            width = 0.8 / len(valid_models_to_run)
            for offset, m_name in enumerate(valid_models_to_run):
                model_df = df_results[df_results["model"] == m_name]
                ax.bar(x + offset * width - (len(valid_models_to_run)-1)*width/2, model_df["f1"], width, label=m_name)
            ax.set_xticks(x)
            ax.set_xticklabels(FAKE_GENERATORS)
            ax.legend()
        
        ax.set_ylim(0, 1.05)
        ax.set_title("Within-Generator Classification F1 Score Comparison")
        ax.set_ylabel("F1 Score")
        ax.set_xlabel("Fake Generator")
        
        if HAS_SEABORN:
            for p in ax.patches:
                height = p.get_height()
                if not np.isnan(height) and height > 0:
                    ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=8)
        
        comparison_chart_path = out_path / "diagonal_bar_chart_comparison.png"
        fig.savefig(comparison_chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined comparison bar chart to {comparison_chart_path}")

    return {
        "results_df": df_results,
        "output_dir": str(out_path)
    }


if __name__ == "__main__":
    run_exp2()
