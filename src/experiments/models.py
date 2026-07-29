"""
models.py — Sklearn Pipelines for model training and comparison.
Scaling inside Pipeline prevents data leakage across splits.
"""

from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


def get_models(seed: int = RANDOM_SEED) -> Dict[str, Pipeline]:
    """
    Return dictionary of named sklearn Pipelines.
    Each pipeline couples StandardScaler with a classifier.
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
