"""
tune_models.py — Standalone Hyperparameter & Feature Count Tuning Script (Phase 1)

Performs 5-fold Stratified Cross-Validation on X_train only to tune feature selection count K,
feature scoring method (f_classif vs mutual_info_classif), and model hyperparameters for LightGBM
and Random Forest. Applies a parsimony rule (score within 0.002 of best favoring simpler models)
and saves best parameters to outputs/results/best_hyperparameters.json.
"""

import os
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer

# Import config and data loader with fallback support
try:
    from src.config import RANDOM_SEED, DRIVE_ROOT
    from src.experiments.data_loader import load_subset
except ImportError:
    try:
        from config import RANDOM_SEED, DRIVE_ROOT
        from experiments.data_loader import load_subset
    except ImportError:
        RANDOM_SEED = 42
        DRIVE_ROOT = Path("/content/drive/MyDrive/ml_project")
        from src.experiments.data_loader import load_subset

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def score_func_map(name: str):
    if name == "f_classif":
        return f_classif
    elif name == "mutual_info_classif":
        return mutual_info_classif
    else:
        raise ValueError(f"Unknown score function: {name}")


def tune_pipeline(X_train: np.ndarray, y_train: np.ndarray, model_type: str, seed: int = RANDOM_SEED):
    """
    Run 5-fold CV grid search over K, score_func, and classifier hyperparams.
    Applies 0.002 parsimony rule.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    num_features = X_train.shape[1]
    
    k_candidates = [k for k in [100, 200, 300, 500, 1000] if k <= num_features]
    if not k_candidates:
        k_candidates = [num_features]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif, k=k_candidates[0])),
        ("clf", None)
    ])

    if model_type == "RandomForest":
        pipe.set_params(clf=RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))
        param_grid = {
            "selector__score_func": [f_classif, mutual_info_classif],
            "selector__k": k_candidates,
            "clf__max_depth": [10, 20, 30, None],
            "clf__min_samples_leaf": [1, 2, 4, 8]
        }
    elif model_type == "LightGBM":
        if not HAS_LIGHTGBM:
            print("LightGBM not installed. Skipping LightGBM tuning.")
            return None
        pipe.set_params(clf=LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=seed, n_jobs=-1, verbose=-1))
        param_grid = {
            "selector__score_func": [f_classif, mutual_info_classif],
            "selector__k": k_candidates,
            "clf__num_leaves": [15, 31, 63],
            "clf__max_depth": [-1, 5, 10, 15],
            "clf__colsample_bytree": [0.6, 0.8, 1.0]
        }
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=make_scorer(f1_score, average="binary"),
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    print(f"\n--- Starting 5-fold CV Grid Search for {model_type} ---")
    grid_search.fit(X_train, y_train)

    cv_results = grid_search.cv_results_
    best_score = grid_search.best_score_
    print(f"Absolute Best CV F1 Score for {model_type}: {best_score:.4f}")

    # Parsimony rule: candidate within 0.002 of best_score
    threshold = best_score - 0.002
    qualifying_indices = [
        i for i, score in enumerate(cv_results["mean_test_score"]) if score >= threshold
    ]

    # Select candidate with minimal k, then simplest model params
    best_idx = None
    min_k = float("inf")

    for idx in qualifying_indices:
        params = cv_results["params"][idx]
        k_val = params["selector__k"]
        if k_val < min_k:
            min_k = k_val
            best_idx = idx

    if best_idx is None:
        best_idx = grid_search.best_index_

    selected_params = cv_results["params"][best_idx]
    selected_score = cv_results["mean_test_score"][best_idx]
    print(f"Selected Parsimonious CV F1 Score: {selected_score:.4f} (K={selected_params['selector__k']})")

    # Format result parameters
    sf_name = "mutual_info_classif" if selected_params["selector__score_func"] == mutual_info_classif else "f_classif"
    
    parsed_config = {
        "score_func": sf_name,
        "k": int(selected_params["selector__k"]),
        "cv_f1_score": float(selected_score)
    }

    if model_type == "RandomForest":
        parsed_config["max_depth"] = selected_params["clf__max_depth"]
        parsed_config["min_samples_leaf"] = int(selected_params["clf__min_samples_leaf"])
        parsed_config["n_estimators"] = 300
    elif model_type == "LightGBM":
        parsed_config["num_leaves"] = int(selected_params["clf__num_leaves"])
        parsed_config["max_depth"] = int(selected_params["clf__max_depth"])
        parsed_config["colsample_bytree"] = float(selected_params["clf__colsample_bytree"])
        parsed_config["n_estimators"] = 500
        parsed_config["learning_rate"] = 0.05

    return parsed_config


def main():
    from src.experiments.data_loader import FEATURES_ROOT
    print(f"Using features path: {FEATURES_ROOT}")
    print("Loading X_train & y_train dataset...")
    X_train, y_train, _ = load_subset("train", feature_set="combined")
    print(f"Loaded train data: X shape = {X_train.shape}, y shape = {y_train.shape}")

    best_hyperparams = {
        "LogisticRegression": {
            "penalty": "l1",
            "C": 0.01,
            "solver": "liblinear"
        },
        "SVM_RBF": {
            "C": 1.0,
            "kernel": "rbf"
        }
    }

    rf_config = tune_pipeline(X_train, y_train, "RandomForest")
    if rf_config:
        best_hyperparams["RandomForest"] = rf_config
        # Store global/shared feature selection from top model
        best_hyperparams["feature_selection"] = {
            "score_func": rf_config["score_func"],
            "k": rf_config["k"]
        }

    lgbm_config = tune_pipeline(X_train, y_train, "LightGBM")
    if lgbm_config:
        best_hyperparams["LightGBM"] = lgbm_config
        # If LightGBM achieves better or equal score, use its feature count/score_func
        if "feature_selection" not in best_hyperparams or lgbm_config["cv_f1_score"] >= rf_config.get("cv_f1_score", 0):
            best_hyperparams["feature_selection"] = {
                "score_func": lgbm_config["score_func"],
                "k": lgbm_config["k"]
            }

    # Ensure output directory exists
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "best_hyperparameters.json"
    with open(output_file, "w") as f:
        json.dump(best_hyperparams, f, indent=4)

    print(f"\n[SUCCESS] Saved tuned hyperparameter configuration to: {output_file}")
    print(json.dumps(best_hyperparams, indent=4))


if __name__ == "__main__":
    main()
