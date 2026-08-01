"""
exp3_transfer.py — Experiment 3: Cross-Generator Transfer Matrix.
Evaluates model generalization by training on Real + Gen_A and testing on Real + Gen_B.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import pandas as pd
import numpy as np

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
    from src.experiments.metrics import evaluate_binary, plot_transfer_matrix
except ImportError:
    from .data_loader import load_subset
    from .models import get_models
    from .metrics import evaluate_binary, plot_transfer_matrix

DEFAULT_OUTPUT_DIR = Path("outputs/results/exp3_transfer")
FAKE_GENERATORS = [g for g in GENERATORS if g != "Real"]


def run_exp3(
    features_root: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    model_name: Union[str, List[str]] = "all",
    feature_set: str = "combined"
) -> Dict[str, Any]:
    """
    Run Experiment 3: Cross-generator transfer matrix evaluation.

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
        Contains matrix_f1, matrix_acc, df_f1, df_acc, and output_dir.
        If multiple models are specified, returns a dictionary keyed by model names,
        along with an 'output_dir' key.
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

    valid_models_to_run = [m for m in models_to_run if m in models]
    if not valid_models_to_run:
        raise ValueError(f"None of the specified models {model_name} are available: {list(models.keys())}")

    n_gens = len(FAKE_GENERATORS)
    
    # Initialize dictionaries to store matrices for each model
    model_matrices = {}
    for m_name in valid_models_to_run:
        model_matrices[m_name] = {
            "matrix_f1": np.zeros((n_gens, n_gens)),
            "matrix_acc": np.zeros((n_gens, n_gens)),
        }

    print(f"Computing 5x5 Cross-Generator Transfer Matrix using models: {valid_models_to_run}...")

    for i, train_gen in enumerate(FAKE_GENERATORS):
        print(f"\nTraining on Real + {train_gen}...")
        X_train, y_train, _ = load_subset("train", generators=["Real", train_gen], feature_set=feature_set, features_root=features_root)

        # Fit all models
        fitted_pipelines = {}
        for m_name in valid_models_to_run:
            pipeline = models[m_name]
            pipeline.fit(X_train, y_train)
            fitted_pipelines[m_name] = pipeline

        for j, test_gen in enumerate(FAKE_GENERATORS):
            X_test, y_test, _ = load_subset("test", generators=["Real", test_gen], feature_set=feature_set, features_root=features_root)

            for m_name in valid_models_to_run:
                pipeline = fitted_pipelines[m_name]
                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
                metrics = evaluate_binary(y_test, y_pred, y_prob)

                model_matrices[m_name]["matrix_f1"][i, j]  = metrics["f1"]
                model_matrices[m_name]["matrix_acc"][i, j] = metrics["accuracy"]
                print(f"  [{m_name}] -> Test on {test_gen}: F1 = {metrics['f1']:.3f}, Acc = {metrics['accuracy']:.3f}")

    results_to_return = {}

    for m_name in valid_models_to_run:
        m_f1 = model_matrices[m_name]["matrix_f1"]
        m_acc = model_matrices[m_name]["matrix_acc"]

        df_f1  = pd.DataFrame(m_f1,  index=FAKE_GENERATORS, columns=FAKE_GENERATORS)
        df_acc = pd.DataFrame(m_acc, index=FAKE_GENERATORS, columns=FAKE_GENERATORS)

        # Create model-specific subdirectory
        model_dir = out_path / m_name
        model_dir.mkdir(parents=True, exist_ok=True)

        df_f1.to_csv(model_dir / "transfer_matrix_f1.csv")
        df_acc.to_csv(model_dir / "transfer_matrix_acc.csv")

        plot_transfer_matrix(m_f1,  FAKE_GENERATORS, f"F1 Score - {m_name}",  model_dir / "transfer_matrix_f1.png")
        plot_transfer_matrix(m_acc, FAKE_GENERATORS, f"Accuracy - {m_name}", model_dir / "transfer_matrix_acc.png")

        # Save to root dir if it's the only model, or if we want a copy at the root level for backwards compatibility
        if len(valid_models_to_run) == 1 or m_name == valid_models_to_run[0]:
            df_f1.to_csv(out_path / "transfer_matrix_f1.csv")
            df_acc.to_csv(out_path / "transfer_matrix_acc.csv")
            plot_transfer_matrix(m_f1,  FAKE_GENERATORS, f"F1 Score - {m_name}",  out_path / "transfer_matrix_f1.png")
            plot_transfer_matrix(m_acc, FAKE_GENERATORS, f"Accuracy - {m_name}", out_path / "transfer_matrix_acc.png")

        results_to_return[m_name] = {
            "matrix_f1": m_f1,
            "matrix_acc": m_acc,
            "df_f1": df_f1,
            "df_acc": df_acc,
        }

    print(f"\nSaved transfer matrices and heatmaps to {out_path}")

    # For backward compatibility, if a single model is run, return its results directly in the dict
    if len(valid_models_to_run) == 1:
        single_model_results = results_to_return[valid_models_to_run[0]]
        return {
            "matrix_f1": single_model_results["matrix_f1"],
            "matrix_acc": single_model_results["matrix_acc"],
            "df_f1": single_model_results["df_f1"],
            "df_acc": single_model_results["df_acc"],
            "output_dir": str(out_path)
        }
    else:
        return {
            "models": results_to_return,
            "output_dir": str(out_path)
        }


if __name__ == "__main__":
    run_exp3()
