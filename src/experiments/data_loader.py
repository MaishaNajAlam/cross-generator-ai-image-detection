"""
data_loader.py — Centralized mask-based loader for experimental feature data.
"""

from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np

try:
    from src.config import FEATURE_ROOT, FEATURE_VERSION
    DEFAULT_FEATURES_ROOT = FEATURE_ROOT / FEATURE_VERSION
except ImportError:
    try:
        from config import FEATURE_ROOT, FEATURE_VERSION
        DEFAULT_FEATURES_ROOT = FEATURE_ROOT / FEATURE_VERSION
    except ImportError:
        DEFAULT_FEATURES_ROOT = Path("/content/drive/MyDrive/ml_project/processed/features/v1")

FEATURES_ROOT = DEFAULT_FEATURES_ROOT


def load_subset(
    split: str,
    generators: Optional[list[str]] = None,
    feature_set: str = "combined",
    features_root: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load feature arrays for a given split, optionally filtered by generators.

    Parameters
    ----------
    split : str
        "train" | "val" | "test"
    generators : list[str] or None
        List of generator names to include (e.g. ["Real", "SDXL"]).
        If None, all classes present in split are returned.
    feature_set : str
        Feature set to load ("combined" | "hog" | "lbp" | "glcm" | "dct" | "wavelet" | "color" | "canny").
    features_root : Path or str, optional
        Custom root directory for feature files. Defaults to FEATURES_ROOT.

    Returns
    -------
    X : np.ndarray
        (N, D) float32 feature matrix
    y_binary : np.ndarray
        (N,) int8 labels (0=Real, 1=Fake)
    y_generator : np.ndarray
        (N,) str generator names
    """
    root = Path(features_root) if features_root is not None else FEATURES_ROOT
    split_dir = root / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Features split directory not found: {split_dir}")

    X_all = np.load(split_dir / f"{feature_set}.npy")
    y_all = np.load(split_dir / "labels.npy")
    gen_all = np.load(split_dir / "generators.npy")

    if generators is not None:
        mask = np.isin(gen_all, generators)
        return X_all[mask], y_all[mask], gen_all[mask]

    return X_all, y_all, gen_all
