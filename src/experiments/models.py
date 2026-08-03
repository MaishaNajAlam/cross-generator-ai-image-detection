"""
models.py — Sklearn Pipelines for model training and comparison.
Scaling inside Pipeline prevents data leakage across splits.

Provides:
  - get_baseline_models(): Unregularized baseline classifiers for diagnostic checks.
  - get_final_models(): Tuned production classifiers (loading best_hyperparameters.json if present).
  - get_models(): Alias to get_final_models() for backward compatibility.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    from src.config import RANDOM_SEED
except ImportError:
    try:
        from config import RANDOM_SEED
    except ImportError:
        RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_JSON_PATH = PROJECT_ROOT / "outputs" / "results" / "best_hyperparameters.json"


def get_baseline_models(seed: int = RANDOM_SEED) -> Dict[str, Pipeline]:
    """
    Return dictionary of named unregularized baseline sklearn Pipelines.
    Used for diagnostic overfit comparison.
    """
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=seed)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
        ]),
    }

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                   random_state=seed, n_jobs=-1, verbose=-1)),
        ])
    except ImportError:
        pass

    return models


def get_final_models(seed: int = RANDOM_SEED, json_path: Optional[Path] = None) -> Dict[str, Pipeline]:
    """
    Return dictionary of tuned production sklearn Pipelines.
    Loads optimal hyperparameters from JSON if available, otherwise uses defaults.
    """
    config = {}
    target_json = json_path if json_path is not None else DEFAULT_JSON_PATH
    if target_json.exists():
        try:
            with open(target_json, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not read {target_json}: {e}. Falling back to default tuned parameters.")

    # RF config
    rf_cfg = config.get("RandomForest", {})
    rf_k = rf_cfg.get("k", 100)
    rf_max_depth = rf_cfg.get("max_depth", 8)
    rf_min_samples_leaf = rf_cfg.get("min_samples_leaf", 4)
    rf_n_estimators = rf_cfg.get("n_estimators", 300)
    rf_max_features = rf_cfg.get("max_features", "sqrt")

    # LGBM config
    lgb_cfg = config.get("LightGBM", {})
    lgb_k = lgb_cfg.get("k", 100)
    lgb_n_estimators = lgb_cfg.get("n_estimators", 300)
    lgb_learning_rate = lgb_cfg.get("learning_rate", 0.03)
    lgb_num_leaves = lgb_cfg.get("num_leaves", 15)
    lgb_max_depth = lgb_cfg.get("max_depth", 6)
    lgb_colsample_bytree = lgb_cfg.get("colsample_bytree", 0.5)
    lgb_subsample = lgb_cfg.get("subsample", 0.8)
    lgb_min_child_samples = lgb_cfg.get("min_child_samples", 20)

    # Logistic Regression (L1 regularized)
    lr_cfg = config.get("LogisticRegression", {})
    lr_penalty = lr_cfg.get("penalty", "l1")
    lr_c = lr_cfg.get("C", 0.01)
    lr_solver = lr_cfg.get("solver", "saga")

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty=lr_penalty, C=lr_c, solver=lr_solver,
                                       max_iter=1000, random_state=seed)),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=seed)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("select_k", SelectKBest(score_func=f_classif, k=rf_k)),
            ("clf", RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth,
                                           min_samples_leaf=rf_min_samples_leaf, max_features=rf_max_features,
                                           random_state=seed, n_jobs=-1)),
        ]),
    }

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("select_k", SelectKBest(score_func=f_classif, k=lgb_k)),
            ("clf", LGBMClassifier(n_estimators=lgb_n_estimators, learning_rate=lgb_learning_rate,
                                   num_leaves=lgb_num_leaves, max_depth=lgb_max_depth,
                                   colsample_bytree=lgb_colsample_bytree, subsample=lgb_subsample,
                                   min_child_samples=lgb_min_child_samples,
                                   random_state=seed, n_jobs=-1, verbose=-1)),
        ])
    except ImportError:
        pass

    return models


def get_models(seed: int = RANDOM_SEED) -> Dict[str, Pipeline]:
    """
    Backwards-compatible wrapper returning tuned production models.
    """
    return get_final_models(seed=seed)

