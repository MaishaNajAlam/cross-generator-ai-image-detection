"""
dataset.py — Load and manage the Phase 1 split CSVs.

Expected CSV schema (train.csv / val.csv / test.csv):
    filename      — e.g. "SDXL_456.jpg"  (stable unique image ID)
    generator     — e.g. "SDXL"
    label_binary  — 0 (Real) or 1 (Fake)

This module:
    - Preserves the CSV row order exactly as written in Phase 1 (no sorting)
    - Reconstructs absolute image paths from IMAGE_ROOT / generator / filename
    - Adds a 'split' column to distinguish train / val / test rows
    - Uses 'filename' as the permanent stable image identifier
"""

import logging
from pathlib import Path

import pandas as pd

from config import SPLITS_DIR, IMAGE_ROOT, GENERATORS

logger = logging.getLogger(__name__)

# Required columns in each split CSV
REQUIRED_COLS = {"filename", "generator", "label_binary"}


def load_splits(splits_dir: Path = SPLITS_DIR) -> pd.DataFrame:
    """
    Load train.csv, val.csv, and test.csv into a single master DataFrame.

    Row order within each CSV is preserved exactly as written in Phase 1.
    The three splits are concatenated in order: train → val → test.

    Parameters
    ----------
    splits_dir : Path
        Directory containing train.csv, val.csv, test.csv.

    Returns
    -------
    pd.DataFrame
        Columns:
            filename      — stable unique image ID (e.g. "SDXL_456.jpg")
            generator     — generator class name
            label_binary  — 0 (Real) or 1 (Fake)
            split         — "train" | "val" | "test"
            image_path    — absolute Path to the image file on disk

        Index is a clean 0-based RangeIndex (row position in combined frame).
    """
    splits_dir = Path(splits_dir)
    split_files = {
        "train": splits_dir / "train.csv",
        "val":   splits_dir / "val.csv",
        "test":  splits_dir / "test.csv",
    }

    frames = []
    for split_name, csv_path in split_files.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Validate required columns
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"{csv_path.name} is missing required columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        df["split"] = split_name
        frames.append(df)
        logger.info("Loaded %s — %d rows", csv_path.name, len(df))

    master = pd.concat(frames, ignore_index=True)  # clean 0-based index

    # Validate generator values
    unknown = set(master["generator"].unique()) - set(GENERATORS)
    if unknown:
        logger.warning("Unknown generator values found: %s", unknown)

    # Validate label values
    invalid_labels = set(master["label_binary"].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"Unexpected label_binary values: {invalid_labels}")

    # Reconstruct full absolute image path: IMAGE_ROOT / generator / filename
    master["image_path"] = master.apply(
        lambda row: IMAGE_ROOT / row["generator"] / row["filename"],
        axis=1,
    )

    # Enforce column order
    master = master[["filename", "generator", "label_binary", "split", "image_path"]]

    logger.info(
        "Master DataFrame: %d total rows | generators: %s | splits: %s",
        len(master),
        dict(master["generator"].value_counts()),
        dict(master["split"].value_counts()),
    )
    return master


def get_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """
    Filter the master DataFrame to rows belonging to a given split.

    Row order is preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame returned by load_splits().
    split : str
        One of "train", "val", "test".

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame. Original integer index is preserved.
    """
    valid = {"train", "val", "test"}
    if split not in valid:
        raise ValueError(f"split must be one of {valid}, got '{split}'")
    return df[df["split"] == split]


def class_balance_report(df: pd.DataFrame) -> str:
    """
    Return a human-readable string summarising class balance per generator.

    Parameters
    ----------
    df : pd.DataFrame
        Master or split DataFrame.

    Returns
    -------
    str
        Formatted balance table.
    """
    lines = [
        f"{'Generator':<15} {'Total':>7} {'Real':>6} {'Fake':>6}",
        "-" * 38,
    ]
    for gen in GENERATORS:
        subset = df[df["generator"] == gen]
        real = (subset["label_binary"] == 0).sum()
        fake = (subset["label_binary"] == 1).sum()
        lines.append(f"{gen:<15} {len(subset):>7} {real:>6} {fake:>6}")
    lines.append("-" * 38)
    lines.append(f"{'TOTAL':<15} {len(df):>7} "
                 f"{(df['label_binary']==0).sum():>6} "
                 f"{(df['label_binary']==1).sum():>6}")
    return "\n".join(lines)
