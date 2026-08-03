"""
data_loader.py — Centralized mask-based loader for experimental feature data.
"""

from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

try:
    from src.config import FEATURE_ROOT, FEATURE_VERSION
    DEFAULT_FEATURES_ROOT = FEATURE_ROOT / FEATURE_VERSION
except ImportError:
    try:
        from config import FEATURE_ROOT, FEATURE_VERSION
        DEFAULT_FEATURES_ROOT = FEATURE_ROOT / FEATURE_VERSION
    except ImportError:
        DEFAULT_FEATURES_ROOT = Path("/content/drive/MyDrive/ml_project_prev/ml_project/processed/features/v1")


def resolve_features_root(provided_root: Optional[Union[str, Path]] = None) -> Path:
    """Resolve features root by checking provided path, defaults, and local project fallbacks."""
    candidates = []
    if provided_root is not None:
        clean_str = str(provided_root).strip()
        if clean_str:
            candidates.append(Path(clean_str))

    candidates.append(DEFAULT_FEATURES_ROOT)

    candidates.extend([
        Path("/content/drive/MyDrive/ml_project_prev/ml_project/processed/features/v1"),
        Path("/content/drive/MyDrive/ml_project/processed/features/v1"),
        PROJECT_ROOT / "data" / "processed" / "features" / "v1",
        PROJECT_ROOT / "processed" / "features" / "v1",
        Path("./processed/features/v1"),
        Path("./data/processed/features/v1"),
    ])

    for cand in candidates:
        try:
            cand_clean = Path(str(cand).strip())
            if cand_clean.exists() and (cand_clean / "train").exists():
                return cand_clean.resolve()
        except Exception:
            continue

    if provided_root is not None:
        return Path(str(provided_root).strip())
    return DEFAULT_FEATURES_ROOT


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
        Custom root directory for feature files. Defaults to resolved FEATURES_ROOT.

    Returns
    -------
    X : np.ndarray
        (N, D) float32 feature matrix
    y_binary : np.ndarray
        (N,) int8 labels (0=Real, 1=Fake)
    y_generator : np.ndarray
        (N,) str generator names
    """
    root = resolve_features_root(features_root)
    split_dir = root / split

    if not split_dir.exists():
        err_msg = (
            f"\n\n[ERROR] Features split directory not found: '{split_dir}'\n"
            f"Troubleshooting Steps:\n"
            f"  1. If running on Google Colab, make sure Google Drive is mounted:\n"
            f"     from google.colab import drive\n"
            f"     drive.mount('/content/drive')\n\n"
            f"  2. Verify that your features exist in Google Drive at:\n"
            f"     /content/drive/MyDrive/ml_project_prev/ml_project/processed/features/v1\n\n"
            f"  3. If your features are stored in a different folder, specify the path when running:\n"
            f"     python check_overfitting_comprehensive.py --features-root /path/to/your/features/v1\n"
        )
        raise FileNotFoundError(err_msg)

    X_all = np.load(split_dir / f"{feature_set}.npy")
    y_all = np.load(split_dir / "labels.npy")
    gen_all = np.load(split_dir / "generators.npy")

    if generators is not None:
        mask = np.isin(gen_all, generators)
        return X_all[mask], y_all[mask], gen_all[mask]

    return X_all, y_all, gen_all

