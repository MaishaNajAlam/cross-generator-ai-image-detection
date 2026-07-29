"""
exp3_transfer.py — Experiment 3: Cross-Generator Transfer Matrix.
Evaluates model generalization by training on Real + Gen_A and testing on Real + Gen_B.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
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
    model_name: str = "LogisticRegression",
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
    model_name : str
        Model to evaluate (defaults to LogisticRegression if LightGBM not installed).
    feature_set : str
        Feature set to use (defaults to 'combined').

    Returns
    -------
    dict
        Contains matrix_f1, matrix_acc, df_f1, df_acc, and output_dir.
    """
    out_path = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    out_path.mkdir(parents=True, exist_ok=True)

    n_gens = len(FAKE_GENERATORS)
    matrix_f1  = np.zeros((n_gens, n_gens))
    matrix_acc = np.zeros((n_gens, n_gens))

    models = get_models()
    target_model = model_name if model_name in models else list(models.keys())[0]

    print(f"Computing 5x5 Cross-Generator Transfer Matrix using {target_model}...")

    for i, train_gen in enumerate(FAKE_GENERATORS):
        print(f"\nTraining on Real + {train_gen}...")
        X_train, y_train, _ = load_subset("train", generators=["Real", train_gen], feature_set=feature_set, features_root=features_root)

        pipeline = models[target_model]
        pipeline.fit(X_train, y_train)

        for j, test_gen in enumerate(FAKE_GENERATORS):
            X_test, y_test, _ = load_subset("test", generators=["Real", test_gen], feature_set=feature_set, features_root=features_root)

            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
            metrics = evaluate_binary(y_test, y_pred, y_prob)

            matrix_f1[i, j]  = metrics["f1"]
            matrix_acc[i, j] = metrics["accuracy"]
            print(f"  -> Test on {test_gen}: F1 = {metrics['f1']:.3f}, Acc = {metrics['accuracy']:.3f}")

    df_f1  = pd.DataFrame(matrix_f1,  index=FAKE_GENERATORS, columns=FAKE_GENERATORS)
    df_acc = pd.DataFrame(matrix_acc, index=FAKE_GENERATORS, columns=FAKE_GENERATORS)

    df_f1.to_csv(out_path / "transfer_matrix_f1.csv")
    df_acc.to_csv(out_path / "transfer_matrix_acc.csv")

    plot_transfer_matrix(matrix_f1,  FAKE_GENERATORS, f"F1 Score - {target_model}",  out_path / "transfer_matrix_f1.png")
    plot_transfer_matrix(matrix_acc, FAKE_GENERATORS, f"Accuracy - {target_model}", out_path / "transfer_matrix_acc.png")

    print(f"\nSaved transfer matrices and heatmaps to {out_path}")

    return {
        "matrix_f1": matrix_f1,
        "matrix_acc": matrix_acc,
        "df_f1": df_f1,
        "df_acc": df_acc,
        "output_dir": str(out_path)
    }


if __name__ == "__main__":
    run_exp3()
