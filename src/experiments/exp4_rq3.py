"""
exp4_rq3.py — Experiment 4: RQ3 Universal vs. Generator-Specific Forensic Features.

Research Question:
    Are there handcrafted forensic features that remain consistently informative
    across different generators, or are the most important features generator-dependent?

Implementation is split across 4 phases:
    Phase 1  (this module)  — Metadata loading, Generator-wise MI, Full feature statistics
    Phase 2                 — Feature Analysis & Fair Family Ranking
    Phase 3                 — Validate Family Importance via Ablation
    Phase 4                 — Visualizations & Master Orchestrator (run_exp4 entry point)

Design guarantees:
    - Fully dimension-independent: D is always derived from X.shape[1] and
      feature_names.txt. No dimension values are hardcoded anywhere.
    - mi_scores_per_generator.csv and feature_stability.csv are saved immediately
      at the end of Phase 1, before any further analysis.
    - Spearman rank correlation is used exclusively for MI vs. F1 ranking validation.
    - Family ranking always uses Average MI per feature (never sum) to ensure
      fair comparison across families of different sizes.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# ── Config imports (dual-mode: absolute for Colab, relative for local) ────────
try:
    from src.config import GENERATORS, RANDOM_SEED, FEATURE_ROOT, FEATURE_VERSION
    DEFAULT_FEATURES_ROOT = FEATURE_ROOT / FEATURE_VERSION
except ImportError:
    try:
        from config import GENERATORS, RANDOM_SEED, FEATURE_ROOT, FEATURE_VERSION
        DEFAULT_FEATURES_ROOT = FEATURE_ROOT / FEATURE_VERSION
    except ImportError:
        GENERATORS = ["Real", "SD21", "SDXL", "SD3", "DALLE3", "Midjourney"]
        RANDOM_SEED = 42
        DEFAULT_FEATURES_ROOT = Path("/content/drive/MyDrive/ml_project/processed/features/v1")

# ── Data loader import (dual-mode) ───────────────────────────────────────────
try:
    from src.experiments.data_loader import load_subset
except ImportError:
    from .data_loader import load_subset

# ── Constants ─────────────────────────────────────────────────────────────────
FAKE_GENERATORS: List[str] = [g for g in GENERATORS if g != "Real"]
FAMILIES: List[str] = ["hog", "lbp", "glcm", "dct", "wavelet", "color", "canny"]

DEFAULT_OUTPUT_DIR = Path("outputs/results/exp4_rq3")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# PHASE 1 — STAGE 0: Metadata Loading & Feature Index Map
# =============================================================================

def load_feature_metadata(
    features_root: Union[str, Path],
) -> pd.DataFrame:
    """
    Load feature metadata and build a dimension-independent feature → family mapping.

    Reads 'feature_metadata.json' for boundary indices and 'feature_names.txt'
    for human-readable names. Validates that the total feature count D matches
    the shape of the combined.npy feature matrix to catch any metadata mismatches.

    Parameters
    ----------
    features_root : str or Path
        Root directory containing 'train/', 'val/', 'test/', and 'metadata/' subdirs.

    Returns
    -------
    pd.DataFrame
        Shape (D, 3) with columns: feature_name, family, index.
        D is always derived dynamically — never hardcoded.

    Raises
    ------
    FileNotFoundError
        If feature_metadata.json or feature_names.txt are missing.
    AssertionError
        If combined.npy.shape[1] != len(feature_names) — metadata mismatch.
    ValueError
        If any feature index remains unassigned (family == 'unknown').
    """
    root = Path(features_root)
    meta_dir = root / "metadata"

    # ── Load boundary index map ───────────────────────────────────────────────
    meta_path = meta_dir / "feature_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"feature_metadata.json not found: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    # ── Load feature names (one per line, length = D) ────────────────────────
    names_path = meta_dir / "feature_names.txt"
    if not names_path.exists():
        raise FileNotFoundError(f"feature_names.txt not found: {names_path}")

    with open(names_path) as f:
        feature_names = [line.strip() for line in f if line.strip()]

    D = len(feature_names)
    logger.info("Loaded %d feature names from feature_names.txt", D)

    # ── CRITICAL: Validate D against actual feature matrix ───────────────────
    combined_path = root / "train" / "combined.npy"
    if combined_path.exists():
        X_check = np.load(combined_path, mmap_mode="r")
        assert X_check.shape[1] == D, (
            f"Metadata mismatch: combined.npy has {X_check.shape[1]} features "
            f"but feature_names.txt has {D} names. "
            f"Re-run feature extraction to regenerate both files."
        )
        logger.info(
            "Dimension check passed: combined.npy.shape[1] == len(feature_names) == %d", D
        )
    else:
        logger.warning(
            "combined.npy not found at %s — dimension assertion skipped.", combined_path
        )

    # ── Build feature → family mapping from boundary indices ─────────────────
    family_map = ["unknown"] * D
    for fam_name, bounds in meta["families"].items():
        start = bounds["start"]
        end   = bounds["end"] + 1   # end is inclusive in metadata, exclusive in range
        for idx in range(start, min(end, D)):
            family_map[idx] = fam_name

    # ── Validate: no feature should remain unassigned ────────────────────────
    unknown_count = family_map.count("unknown")
    if unknown_count > 0:
        unknown_indices = [i for i, f in enumerate(family_map) if f == "unknown"]
        raise ValueError(
            f"{unknown_count} feature(s) not assigned to any family. "
            f"Indices: {unknown_indices[:20]}{'...' if unknown_count > 20 else ''}. "
            f"Check that feature_metadata.json boundaries cover all {D} features."
        )

    feature_meta_df = pd.DataFrame({
        "feature_name": feature_names,
        "family":       family_map,
        "index":        list(range(D)),
    })

    # ── Log family dimension summary ──────────────────────────────────────────
    family_counts = feature_meta_df["family"].value_counts().sort_index()
    logger.info("Feature family dimensions:")
    for fam, count in family_counts.items():
        logger.info("  %-12s  %d features", fam, count)
    logger.info("  %-12s  %d features (total)", "TOTAL", D)

    return feature_meta_df


# =============================================================================
# PHASE 1 — STAGE 1: Generator-wise Mutual Information
# =============================================================================

def compute_generator_mi(
    features_root: Union[str, Path],
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Compute mutual_info_classif(feature_i, binary_label) for every feature,
    for each generator subset independently.

    For each generator G, the subset loaded is: Real images ∪ G-generated images.
    MI is computed with discrete_features=False (all features are continuous float32).

    Parameters
    ----------
    features_root : str or Path
        Root directory for feature files (must contain train/combined.npy etc.).
    random_seed : int
        Random seed for mutual_info_classif reproducibility.

    Returns
    -------
    pd.DataFrame
        Shape (D, 5): rows = features (indexed by feature_name),
        columns = FAKE_GENERATORS (SD21, SDXL, SD3, DALLE3, Midjourney).
        D is the actual number of features in combined.npy — never hardcoded.
    """
    root = Path(features_root)
    mi_scores: Dict[str, np.ndarray] = {}

    logger.info("Computing generator-wise Mutual Information for %d generators...", len(FAKE_GENERATORS))

    for gen in FAKE_GENERATORS:
        logger.info("  Loading Real + %s subset...", gen)
        X_g, y_g, _ = load_subset(
            split="train",
            generators=["Real", gen],
            feature_set="combined",
            features_root=root,
        )

        D = X_g.shape[1]  # Always dynamic — never hardcoded
        logger.info("    Subset shape: %s  |  Label distribution: Real=%d, Fake=%d",
                    X_g.shape,
                    int((y_g == 0).sum()),
                    int((y_g == 1).sum()))

        logger.info("    Computing MI for %d features...", D)
        mi_scores[gen] = mutual_info_classif(
            X_g,
            y_g,
            discrete_features=False,
            random_state=random_seed,
        )
        logger.info("    Done: max_mi=%.4f, mean_mi=%.4f",
                    float(mi_scores[gen].max()),
                    float(mi_scores[gen].mean()))

    # Assemble into a DataFrame indexed by feature position
    # Feature names will be assigned in the caller (after load_feature_metadata)
    mi_array = np.stack([mi_scores[gen] for gen in FAKE_GENERATORS], axis=1)
    mi_df = pd.DataFrame(
        mi_array,
        columns=FAKE_GENERATORS,
    )
    return mi_df


# =============================================================================
# PHASE 1 — STAGE 5: Full Feature Statistics (Complete D-feature save)
# =============================================================================

def build_full_stability_df(
    feature_meta_df: pd.DataFrame,
    mi_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge feature metadata with per-generator MI scores and compute all
    summary statistics. Assembles the complete feature stability DataFrame.

    Computed metrics:
        mean_mi           — Average MI across all generators (primary universal signal)
        std_mi            — Std of MI across generators (variability proxy)
        max_mi            — Maximum MI observed for any single generator
        stability_score   — mean_mi / (std_mi + 1e-9)  [high = universally informative]
        specificity_ratio — max_mi / (mean_mi + 1e-9)  [high = generator-specific]

    Parameters
    ----------
    feature_meta_df : pd.DataFrame
        Output of load_feature_metadata(): columns [feature_name, family, index].
    mi_df : pd.DataFrame
        Output of compute_generator_mi(): shape (D, 5), columns = FAKE_GENERATORS.
        Index must be aligned with feature_meta_df (both 0-indexed by feature position).

    Returns
    -------
    pd.DataFrame
        Shape (D, 13) with columns:
            feature_name, family, index,
            SD21, SDXL, SD3, DALLE3, Midjourney,
            mean_mi, std_mi, max_mi, stability_score, specificity_ratio
    """
    D = len(feature_meta_df)
    assert len(mi_df) == D, (
        f"Row count mismatch: feature_meta_df has {D} rows, mi_df has {len(mi_df)} rows."
    )

    # Set feature names as index on mi_df for downstream merging
    mi_df_indexed = mi_df.copy()
    mi_df_indexed.index = feature_meta_df["feature_name"].values

    # Compute per-feature summary statistics across generators
    mi_values = mi_df[FAKE_GENERATORS].values   # (D, 5) numpy array

    mean_mi          = mi_values.mean(axis=1)
    std_mi           = mi_values.std(axis=1)
    max_mi           = mi_values.max(axis=1)
    stability_score  = mean_mi / (std_mi + 1e-9)
    specificity_ratio = max_mi / (mean_mi + 1e-9)

    full_stats_df = feature_meta_df.copy().reset_index(drop=True)
    for gen in FAKE_GENERATORS:
        full_stats_df[gen] = mi_df[gen].values

    full_stats_df["mean_mi"]           = mean_mi
    full_stats_df["std_mi"]            = std_mi
    full_stats_df["max_mi"]            = max_mi
    full_stats_df["stability_score"]   = stability_score
    full_stats_df["specificity_ratio"] = specificity_ratio

    logger.info(
        "Built full feature stability DataFrame: shape %s", full_stats_df.shape
    )
    logger.info(
        "  mean_mi stats  — min: %.4f, median: %.4f, max: %.4f",
        float(mean_mi.min()), float(np.median(mean_mi)), float(mean_mi.max())
    )
    logger.info(
        "  stability_score— min: %.4f, median: %.4f, max: %.4f",
        float(stability_score.min()), float(np.median(stability_score)), float(stability_score.max())
    )

    return full_stats_df


# =============================================================================
# PHASE 1 — Master Save: Persist Phase 1 outputs immediately
# =============================================================================

def run_phase1(
    features_root: Union[str, Path],
    output_dir: Union[str, Path],
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute Phase 1 of RQ3 end-to-end and save outputs immediately.

    Steps:
        1. Load feature metadata and build feature → family index map.
        2. Compute mutual_info_classif per generator subset (D × 5).
        3. Compute all summary statistics.
        4. Save mi_scores_per_generator.csv and feature_stability.csv immediately.

    Parameters
    ----------
    features_root : str or Path
        Root directory containing train/combined.npy, metadata/feature_metadata.json,
        and metadata/feature_names.txt.
    output_dir : str or Path
        Directory where Phase 1 CSVs will be saved.
    random_seed : int
        Seed for mutual_info_classif.

    Returns
    -------
    Tuple of:
        feature_meta_df  — (D, 3) feature index map
        mi_df            — (D, 5) per-generator MI scores, indexed by feature_name
        full_stats_df    — (D, 13) complete feature statistics
    """
    root     = Path(features_root)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RQ3 — Phase 1: Metadata, Generator MI & Feature Statistics")
    logger.info("=" * 60)
    logger.info("features_root : %s", root)
    logger.info("output_dir    : %s", out_path)

    # ── Stage 0: Load metadata ────────────────────────────────────────────────
    logger.info("\n[Stage 0] Loading feature metadata...")
    feature_meta_df = load_feature_metadata(root)

    # ── Stage 1: Compute MI per generator ────────────────────────────────────
    logger.info("\n[Stage 1] Computing generator-wise Mutual Information...")
    mi_df_raw = compute_generator_mi(root, random_seed=random_seed)

    # Attach feature names as index (required for downstream Stages 2–4)
    mi_df = mi_df_raw.copy()
    mi_df.index = feature_meta_df["feature_name"].values
    mi_df.index.name = "feature_name"

    # Save mi_scores_per_generator.csv immediately (D × 5 table — fundamental RQ3 table)
    mi_out_path = out_path / "mi_scores_per_generator.csv"
    mi_df.to_csv(mi_out_path)
    logger.info("\nSaved: %s  (shape: %s)", mi_out_path, mi_df.shape)

    # ── Stage 5: Build full statistics (all D features × 13 columns) ─────────
    logger.info("\n[Stage 5] Building full feature stability statistics...")
    full_stats_df = build_full_stability_df(feature_meta_df, mi_df_raw)

    # Save feature_stability.csv immediately (complete, reproducible record)
    stability_out_path = out_path / "feature_stability.csv"
    full_stats_df.to_csv(stability_out_path, index=False)
    logger.info("Saved: %s  (shape: %s)", stability_out_path, full_stats_df.shape)

    # ── Summary log ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1 Complete. Outputs saved to: %s", out_path)
    logger.info("  mi_scores_per_generator.csv  (%d features × %d generators)",
                mi_df.shape[0], mi_df.shape[1])
    logger.info("  feature_stability.csv        (%d features × %d columns)",
                full_stats_df.shape[0], full_stats_df.shape[1])
    logger.info("=" * 60)

    return feature_meta_df, mi_df, full_stats_df


# =============================================================================
# PHASE 2 — Feature Analysis & Family Ranking
# =============================================================================

# ── Stage 2: Universal Feature Selection ─────────────────────────────────────

def extract_universal_features(
    full_stats_df: pd.DataFrame,
    info_threshold: float = 0.05,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Identify features that are informative AND stable across all generators.

    Two-step selection:
        Step 1 — Informativeness filter: keep only features where mean_mi > info_threshold.
                 This removes noise features whose stability_score is artificially high
                 due to near-zero std (e.g., MI=[0.001]*5 gets stability≈∞ but is useless).
        Step 2 — Stability ranking: sort passing features by stability_score descending,
                 return top_n.

    Parameters
    ----------
    full_stats_df : pd.DataFrame
        Output of build_full_stability_df() — shape (D, 13).
    info_threshold : float
        Minimum mean_mi a feature must have to qualify as 'informative'.
        Default: 0.05 (tunable; lower = more permissive).
    top_n : int
        Number of top universal features to return.

    Returns
    -------
    pd.DataFrame
        Top-N universal features sorted by stability_score descending.
        Columns: feature_name, family, mean_mi, std_mi, stability_score,
                 + one column per generator (individual MI scores).
    """
    # Step 1: Informativeness filter
    informative_mask = full_stats_df["mean_mi"] > info_threshold
    informative_df   = full_stats_df[informative_mask].copy()

    n_informative = len(informative_df)
    n_total       = len(full_stats_df)
    logger.info(
        "Universal filter: %d / %d features pass mean_mi > %.3f",
        n_informative, n_total, info_threshold
    )

    if n_informative == 0:
        logger.warning(
            "No features pass info_threshold=%.3f. Consider lowering the threshold.",
            info_threshold
        )
        return pd.DataFrame()

    # Step 2: Stability ranking
    cols = ["feature_name", "family", "mean_mi", "std_mi", "stability_score"] + FAKE_GENERATORS
    universal_df = (
        informative_df
        .sort_values("stability_score", ascending=False)
        .head(top_n)
        [cols]
        .reset_index(drop=True)
    )

    logger.info(
        "Top %d universal features (by stability_score): families represented = %s",
        len(universal_df),
        universal_df["family"].value_counts().to_dict()
    )
    return universal_df


# ── Stage 3: Generator-Specific Feature Detection ────────────────────────────

def extract_generator_specific_features(
    full_stats_df: pd.DataFrame,
    specificity_ratio_threshold: float = 2.0,
    max_mi_threshold: float = 0.10,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Identify features that are strongly informative for one generator but weak for others.

    A feature qualifies as generator-specific if it satisfies BOTH conditions:
        1. max_mi >= max_mi_threshold  — it carries genuine information for ≥1 generator
        2. max_mi / mean_mi >= specificity_ratio_threshold — disproportionately useful
           for one generator vs. the average across all generators

    Why max_mi/mean_mi instead of std_mi alone:
        A feature with MI=[0.5, 0.01, 0.01, 0.01, 0.01] is a true fingerprint.
        Its ratio ≈ 10×. Using std_mi alone could include features that fluctuate
        between two medium values (e.g., 0.3 and 0.5) which are variable but not
        generator-specific.

    Parameters
    ----------
    full_stats_df : pd.DataFrame
        Output of build_full_stability_df() — shape (D, 13).
    specificity_ratio_threshold : float
        Minimum max_mi/mean_mi ratio to qualify. Default: 2.0.
    max_mi_threshold : float
        Minimum max_mi to ensure the feature is genuinely informative. Default: 0.10.
    top_n : int
        Number of top specific features to return.

    Returns
    -------
    pd.DataFrame
        Top-N generator-specific features sorted by max_mi descending.
        Columns: feature_name, family, max_mi, mean_mi, specificity_ratio,
                 + one column per generator (individual MI scores).
    """
    mask = (
        (full_stats_df["max_mi"]           >= max_mi_threshold) &
        (full_stats_df["specificity_ratio"] >= specificity_ratio_threshold)
    )
    specific_pool = full_stats_df[mask].copy()

    n_specific = len(specific_pool)
    n_total    = len(full_stats_df)
    logger.info(
        "Specificity filter: %d / %d features pass "
        "(max_mi >= %.2f AND specificity_ratio >= %.1f)",
        n_specific, n_total, max_mi_threshold, specificity_ratio_threshold
    )

    if n_specific == 0:
        logger.warning(
            "No generator-specific features found. "
            "Consider lowering specificity_ratio_threshold (current=%.1f) "
            "or max_mi_threshold (current=%.2f).",
            specificity_ratio_threshold, max_mi_threshold
        )
        return pd.DataFrame()

    cols = ["feature_name", "family", "max_mi", "mean_mi", "specificity_ratio"] + FAKE_GENERATORS
    specific_df = (
        specific_pool
        .sort_values("max_mi", ascending=False)
        .head(top_n)
        [cols]
        .reset_index(drop=True)
    )

    logger.info(
        "Top %d generator-specific features (by max_mi): families represented = %s",
        len(specific_df),
        specific_df["family"].value_counts().to_dict()
    )
    return specific_df


# ── Stage 4: Feature Family Ranking (Normalized by Average MI per Feature) ───

def compute_family_ranking(
    feature_meta_df: pd.DataFrame,
    mi_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rank the 7 feature families by their average per-feature MI across generators.

    Normalization rule:
        Compare families using AVERAGE MI per feature, not sum.
        This prevents high-dimensional families (e.g. HOG with ~8100 features)
        from dominating the ranking purely due to their size — an artefact of
        the extraction configuration, not forensic informativeness.

    Parameters
    ----------
    feature_meta_df : pd.DataFrame
        Output of load_feature_metadata(): columns [feature_name, family, index].
    mi_df : pd.DataFrame
        Per-generator MI DataFrame, indexed by feature_name. Shape (D, 5).

    Returns
    -------
    Tuple of:
        family_detail_df — long-form table (family × generator), all stats
        family_pivot_df  — wide pivot: families (rows) × generators + avg_across (cols)
                           includes n_features column documenting actual dimensions.
    """
    family_rows = []

    for fam in FAMILIES:
        fam_mask         = feature_meta_df["family"] == fam
        fam_feature_names = feature_meta_df.loc[fam_mask, "feature_name"].tolist()
        n_fam_features   = len(fam_feature_names)

        if n_fam_features == 0:
            logger.warning("Family '%s' has 0 features — skipping.", fam)
            continue

        for gen in FAKE_GENERATORS:
            # mi_df is indexed by feature_name — select the family's features
            fam_mi_values = mi_df.loc[fam_feature_names, gen]

            family_rows.append({
                "family":     fam,
                "generator":  gen,
                "avg_mi":     float(fam_mi_values.mean()),    # PRIMARY metric — normalized
                "median_mi":  float(fam_mi_values.median()),
                "max_mi":     float(fam_mi_values.max()),
                "n_features": n_fam_features,                 # Always document actual dims
            })

    family_detail_df = pd.DataFrame(family_rows)

    # Pivot: families (rows) × generators (cols), values = avg_mi
    family_pivot_df = family_detail_df.pivot(
        index="family", columns="generator", values="avg_mi"
    )[FAKE_GENERATORS].copy()

    # Cross-generator average (primary ranking column)
    family_pivot_df["avg_across_generators"] = family_pivot_df[FAKE_GENERATORS].mean(axis=1)

    # Add n_features column (one value per family — same across all generators)
    n_features_map = (
        family_detail_df.drop_duplicates("family")
        .set_index("family")["n_features"]
    )
    family_pivot_df["n_features"] = family_pivot_df.index.map(n_features_map)

    # Sort by cross-generator average MI (descending)
    family_pivot_df = family_pivot_df.sort_values("avg_across_generators", ascending=False)

    logger.info("Feature family ranking (by avg MI per feature, cross-generator):")
    for fam in family_pivot_df.index:
        logger.info(
            "  %-10s  avg_mi=%.4f  n_features=%d",
            fam,
            family_pivot_df.loc[fam, "avg_across_generators"],
            int(family_pivot_df.loc[fam, "n_features"]),
        )

    return family_detail_df, family_pivot_df


# ── Phase 2 Orchestrator ──────────────────────────────────────────────────────

def run_phase2(
    feature_meta_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    full_stats_df: pd.DataFrame,
    output_dir: Union[str, Path],
    info_threshold: float = 0.05,
    specificity_ratio_threshold: float = 2.0,
    max_mi_threshold: float = 0.10,
    top_n: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute Phase 2 of RQ3: Feature Analysis & Family Ranking.

    Stages covered:
        Stage 2 — Universal feature selection (mean_mi filter + stability ranking)
        Stage 3 — Generator-specific feature detection (ratio filter + max_mi ranking)
        Stage 4 — Family-level ranking by average MI per feature, with n_features

    Parameters
    ----------
    feature_meta_df : pd.DataFrame
        Output of run_phase1() — feature → family index map.
    mi_df : pd.DataFrame
        Output of run_phase1() — (D, 5) per-generator MI, indexed by feature_name.
    full_stats_df : pd.DataFrame
        Output of run_phase1() — (D, 13) complete feature statistics.
    output_dir : str or Path
        Directory where Phase 2 CSVs will be saved.
    info_threshold : float
        Min mean_mi for universal feature filter.
    specificity_ratio_threshold : float
        Min max_mi/mean_mi for generator-specific filter.
    max_mi_threshold : float
        Min max_mi for generator-specific filter.
    top_n : int
        Top N features to retain in universal and specific tables.

    Returns
    -------
    Tuple of:
        universal_df      — Top-N universal features
        specific_df       — Top-N generator-specific features
        family_detail_df  — Long-form family × generator MI stats
        family_pivot_df   — Wide pivot family ranking table (with n_features)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RQ3 — Phase 2: Feature Analysis & Family Ranking")
    logger.info("=" * 60)

    # ── Stage 2: Universal features ───────────────────────────────────────────
    logger.info("\n[Stage 2] Extracting universal features...")
    universal_df = extract_universal_features(
        full_stats_df,
        info_threshold=info_threshold,
        top_n=top_n,
    )
    if not universal_df.empty:
        universal_path = out_path / "universal_features.csv"
        universal_df.to_csv(universal_path, index=False)
        logger.info("Saved: %s  (%d rows)", universal_path, len(universal_df))
        # Verify filter integrity
        assert (universal_df["mean_mi"] > info_threshold).all(), (
            "VERIFICATION FAILED: Some universal features have mean_mi <= info_threshold."
        )
        logger.info("Verification passed: all universal features have mean_mi > %.3f",
                    info_threshold)

    # ── Stage 3: Generator-specific features ──────────────────────────────────
    logger.info("\n[Stage 3] Extracting generator-specific features...")
    specific_df = extract_generator_specific_features(
        full_stats_df,
        specificity_ratio_threshold=specificity_ratio_threshold,
        max_mi_threshold=max_mi_threshold,
        top_n=top_n,
    )
    if not specific_df.empty:
        specific_path = out_path / "generator_specific_features.csv"
        specific_df.to_csv(specific_path, index=False)
        logger.info("Saved: %s  (%d rows)", specific_path, len(specific_df))
        # Verify filter integrity
        assert (specific_df["specificity_ratio"] >= specificity_ratio_threshold).all(), (
            "VERIFICATION FAILED: Some specific features have specificity_ratio < threshold."
        )
        logger.info(
            "Verification passed: all specific features have specificity_ratio >= %.1f",
            specificity_ratio_threshold
        )

    # ── Stage 4: Family ranking ───────────────────────────────────────────────
    logger.info("\n[Stage 4] Computing feature family ranking (normalized avg MI)...")
    family_detail_df, family_pivot_df = compute_family_ranking(feature_meta_df, mi_df)

    family_ranking_path = out_path / "feature_family_ranking.csv"
    family_pivot_df.to_csv(family_ranking_path)
    logger.info("Saved: %s  (%d families × %d cols)",
                family_ranking_path, len(family_pivot_df), len(family_pivot_df.columns))

    # Verify n_features column is present and avg_mi is used (not sum)
    assert "n_features" in family_pivot_df.columns, (
        "VERIFICATION FAILED: n_features column missing from family_pivot_df."
    )
    assert "avg_across_generators" in family_pivot_df.columns, (
        "VERIFICATION FAILED: avg_across_generators column missing from family_pivot_df."
    )
    logger.info("Verification passed: n_features column present, avg_mi ranking in use.")

    logger.info("\n" + "=" * 60)
    logger.info("Phase 2 Complete. Outputs saved to: %s", out_path)
    logger.info("  universal_features.csv            (%d rows)", len(universal_df))
    logger.info("  generator_specific_features.csv   (%d rows)", len(specific_df))
    logger.info("  feature_family_ranking.csv         (%d families)", len(family_pivot_df))
    logger.info("=" * 60)

    return universal_df, specific_df, family_detail_df, family_pivot_df


# =============================================================================
# PHASE 3 — Validate Family Importance via Ablation
# =============================================================================

def validate_family_importance_via_ablation(
    features_root: Union[str, Path],
    output_dir: Union[str, Path],
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Validate MI-based family rankings by measuring actual cross-generator
    classification performance using LightGBM trained on each feature family.

    Design decisions:
        - LightGBM only (best model from RQ1/RQ2; no need to repeat all 4 classifiers)
        - Off-diagonal pairs only: train_gen != test_gen (strictly cross-generator)
        - Uses load_subset(feature_set=family) to load per-family .npy files
        - Average cross-gen F1 is the single summary metric per family

    The off-diagonal average F1 reflects how well a family's forensic signal
    transfers to unseen generators — the core question of cross-generator generalization.

    Parameters
    ----------
    features_root : str or Path
        Root directory for feature files (must contain per-family .npy files,
        e.g. train/hog.npy, train/lbp.npy, etc.).
    output_dir : str or Path
        Directory where feature_family_ablation.csv will be saved.
    random_seed : int
        Seed for LightGBM reproducibility.

    Returns
    -------
    pd.DataFrame
        Shape (7, 2) — columns: family, avg_cross_gen_f1.
        Sorted by avg_cross_gen_f1 descending.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError(
            "LightGBM is required for Phase 3 ablation. "
            "Install with: pip install lightgbm"
        )

    root     = Path(features_root)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RQ3 — Phase 3: Validate Family Importance via Ablation")
    logger.info("=" * 60)
    logger.info("Model      : LightGBM (best from RQ1/RQ2)")
    logger.info("Evaluation : off-diagonal cross-generator F1 (train_gen != test_gen)")
    logger.info("Families   : %s", FAMILIES)

    from sklearn.feature_selection import SelectKBest, f_classif

    lgbm_pipeline = None
    try:
        try:
            from src.experiments.models import get_final_models
        except ImportError:
            from .models import get_final_models
        
        if get_final_models is not None:
            models_dict = get_final_models(seed=random_seed)
            if "LightGBM" in models_dict:
                lgbm_pipeline = models_dict["LightGBM"]
                logger.info("Loaded tuned LightGBM pipeline for ablation.")
    except Exception as e:
        logger.warning("Could not load tuned LightGBM model: %s. Using default baseline.", e)

    if lgbm_pipeline is None:
        lgbm_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                random_state=random_seed,
                n_jobs=-1,
                verbose=-1,
            )),
        ])
        logger.info("Initialized default baseline LightGBM pipeline.")

    # Record the original 'k' parameter of SelectKBest (if present) so we can dynamically cap it per family
    tuned_k = None
    if "select_k" in lgbm_pipeline.named_steps:
        tuned_k = lgbm_pipeline.named_steps["select_k"].k
        logger.info("Tuned select_k features: k = %s", tuned_k)

    ablation_rows = []

    for family in FAMILIES:
        logger.info("\n[Ablation] Family: %s", family.upper())
        cross_f1_list = []

        for train_gen in FAKE_GENERATORS:
            # Load training data: Real + train_gen, using this family's features only
            try:
                X_tr, y_tr, _ = load_subset(
                    split="train",
                    generators=["Real", train_gen],
                    feature_set=family,
                    features_root=root,
                )
            except FileNotFoundError:
                logger.warning(
                    "  Family '%s' train file not found for generator '%s' — skipping.",
                    family, train_gen
                )
                continue

            if tuned_k is not None and "select_k" in lgbm_pipeline.named_steps:
                lgbm_pipeline.named_steps["select_k"].k = min(tuned_k, X_tr.shape[1])

            lgbm_pipeline.fit(X_tr, y_tr)
            logger.info(
                "  Trained on Real + %-12s  (X_train shape: %s)",
                train_gen, X_tr.shape
            )

            for test_gen in FAKE_GENERATORS:
                if test_gen == train_gen:
                    continue  # strictly off-diagonal — no in-distribution evaluation

                try:
                    X_te, y_te, _ = load_subset(
                        split="test",
                        generators=["Real", test_gen],
                        feature_set=family,
                        features_root=root,
                    )
                except FileNotFoundError:
                    logger.warning(
                        "  Family '%s' test file not found for generator '%s' — skipping.",
                        family, test_gen
                    )
                    continue

                y_pred = lgbm_pipeline.predict(X_te)
                f1     = float(f1_score(y_te, y_pred, zero_division=0))
                cross_f1_list.append(f1)
                logger.info(
                    "    Test on %-12s  F1 = %.4f",
                    test_gen, f1
                )

        if cross_f1_list:
            avg_f1 = float(np.mean(cross_f1_list))
        else:
            avg_f1 = float("nan")
            logger.warning("  No valid train/test pairs found for family '%s'.", family)

        logger.info("  %s  avg cross-gen F1 = %.4f  (over %d pairs)",
                    family.upper(), avg_f1, len(cross_f1_list))
        ablation_rows.append({
            "family":           family,
            "avg_cross_gen_f1": avg_f1,
            "n_pairs":          len(cross_f1_list),
        })

    ablation_df = (
        pd.DataFrame(ablation_rows)
        .sort_values("avg_cross_gen_f1", ascending=False)
        .reset_index(drop=True)
    )

    # Save immediately
    ablation_path = out_path / "feature_family_ablation.csv"
    ablation_df.to_csv(ablation_path, index=False)
    logger.info("\nSaved: %s  (%d families)", ablation_path, len(ablation_df))

    return ablation_df


def compute_ranking_correlation(
    family_pivot_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
) -> float:
    """
    Compute Spearman rank correlation between the MI-based family ranking
    and the ablation-based F1 family ranking.

    Spearman rank correlation is used exclusively (not Pearson/Kendall) because:
        - We are comparing two ordinal rankings, not continuous values
        - Spearman is robust to outliers and monotone non-linear relationships
        - It directly measures whether higher avg_mi predicts higher avg_cross_gen_f1

    Interpretation:
        r > 0.6  — MI family ranking is strongly validated by ablation results
        r > 0.4  — Moderate agreement; MI is a reasonable (not perfect) proxy
        r < 0.4  — Weak agreement; MI-based conclusions should be treated cautiously

    Parameters
    ----------
    family_pivot_df : pd.DataFrame
        Output of compute_family_ranking() — must have 'avg_across_generators' column.
        Index = family names.
    ablation_df : pd.DataFrame
        Output of validate_family_importance_via_ablation() — must have
        'family' and 'avg_cross_gen_f1' columns.

    Returns
    -------
    float
        Spearman rank correlation coefficient r in [-1, 1].
        Printed to console and returned for use in run_exp4 summary.
    """
    from scipy.stats import spearmanr

    # Align on shared families
    mi_series = family_pivot_df["avg_across_generators"].rename("avg_mi")
    f1_series = ablation_df.set_index("family")["avg_cross_gen_f1"]

    shared_families = mi_series.index.intersection(f1_series.index)
    if len(shared_families) < 3:
        logger.warning(
            "Only %d shared families for Spearman correlation — result unreliable.",
            len(shared_families)
        )
        return float("nan")

    mi_vals = mi_series.loc[shared_families].values
    f1_vals = f1_series.loc[shared_families].values

    r, p_value = spearmanr(mi_vals, f1_vals)
    r = float(r)

    # Log with interpretation
    if r > 0.6:
        interpretation = "STRONG — MI conclusions are validated by ablation."
    elif r > 0.4:
        interpretation = "MODERATE — MI is a reasonable proxy for F1 ranking."
    else:
        interpretation = "WEAK — MI-based conclusions should be treated cautiously."

    logger.info("\n" + "=" * 60)
    logger.info("Spearman Rank Correlation (MI ranking vs F1 ranking):")
    logger.info("  r = %.4f  (p = %.4f)", r, p_value)
    logger.info("  Interpretation: %s", interpretation)
    logger.info("=" * 60)

    print(f"\nSpearman Rank Correlation (MI ranking vs F1 ranking): r = {r:.4f}  (p = {p_value:.4f})")
    print(f"→ {interpretation}")

    return r


def run_phase3(
    features_root: Union[str, Path],
    output_dir: Union[str, Path],
    family_pivot_df: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, float]:
    """
    Execute Phase 3 of RQ3: Validate Family Importance via Ablation.

    Steps:
        1. Run LightGBM ablation for all 7 families (off-diagonal cross-gen F1).
        2. Save feature_family_ablation.csv.
        3. Compute and log Spearman rank correlation vs. MI family ranking.

    Parameters
    ----------
    features_root : str or Path
        Root directory for feature files (per-family .npy files must exist).
    output_dir : str or Path
        Directory where feature_family_ablation.csv will be saved.
    family_pivot_df : pd.DataFrame
        Output of run_phase2() — required for Spearman correlation.
    random_seed : int
        Seed for LightGBM.

    Returns
    -------
    Tuple of:
        ablation_df   — (7, 3) family ablation results
        spearman_r    — Spearman rank correlation coefficient
    """
    ablation_df = validate_family_importance_via_ablation(
        features_root=features_root,
        output_dir=output_dir,
        random_seed=random_seed,
    )

    spearman_r = compute_ranking_correlation(
        family_pivot_df=family_pivot_df,
        ablation_df=ablation_df,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Phase 3 Complete.")
    logger.info("  feature_family_ablation.csv saved.")
    logger.info("  Spearman r = %.4f", spearman_r)
    logger.info("=" * 60)

    return ablation_df, spearman_r


# =============================================================================
# PHASE 4 — Visualizations & Master Orchestrator
# =============================================================================

def plot_rq3_figures(
    output_dir: Union[str, Path],
    family_pivot_df: pd.DataFrame,
    full_stats_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    family_detail_df: pd.DataFrame,
) -> None:
    """
    Generate and save all RQ3 visualizations to output_dir/figures/.

    Plots produced:
        1. mi_heatmap_families.png   — Heatmap: 7 families × 5 generators (avg MI)
        2. family_avg_mi_bar.png     — Bar: avg MI per family (cross-generator avg)
        3. stability_scatter.png     — Scatter: mean_mi vs std_mi, colored by family
        4. ablation_f1_bar.png       — Bar: avg cross-gen F1 from LightGBM ablation

    Parameters
    ----------
    output_dir : str or Path
    family_pivot_df : pd.DataFrame
        Wide pivot from compute_family_ranking() with avg_mi per family/generator.
    full_stats_df : pd.DataFrame
        Complete (D, 13) feature statistics from build_full_stability_df().
    ablation_df : pd.DataFrame
        Family ablation results from validate_family_importance_via_ablation().
    family_detail_df : pd.DataFrame
        Long-form family × generator table from compute_family_ranking().
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    try:
        import seaborn as sns
        HAS_SEABORN = True
    except ImportError:
        HAS_SEABORN = False
        logger.warning("seaborn not installed — falling back to matplotlib for heatmap.")

    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    FAMILY_ORDER_PLOT = list(
        family_pivot_df.sort_values("avg_across_generators", ascending=False).index
    )

    # ── Plot 1: MI Heatmap — 7 families × 5 generators ───────────────────────
    logger.info("[Plot 1] Generating mi_heatmap_families.png ...")
    heatmap_data = family_pivot_df.loc[FAMILY_ORDER_PLOT, FAKE_GENERATORS]

    fig, ax = plt.subplots(figsize=(9, 5))
    if HAS_SEABORN:
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Avg MI per Feature"},
        )
    else:
        cax = ax.matshow(heatmap_data.values, cmap="YlOrRd")
        fig.colorbar(cax, label="Avg MI per Feature")
        ax.set_xticks(range(len(FAKE_GENERATORS)))
        ax.set_yticks(range(len(FAMILY_ORDER_PLOT)))
        ax.set_xticklabels(FAKE_GENERATORS, rotation=45, ha="left")
        ax.set_yticklabels(FAMILY_ORDER_PLOT)
        for i in range(len(FAMILY_ORDER_PLOT)):
            for j in range(len(FAKE_GENERATORS)):
                ax.text(j, i, f"{heatmap_data.values[i, j]:.4f}",
                        ha="center", va="center", fontsize=8, color="black")

    ax.set_title("Average MI per Feature — Family × Generator",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Generator", fontsize=11)
    ax.set_ylabel("Feature Family", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "mi_heatmap_families.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", fig_dir / "mi_heatmap_families.png")

    # ── Plot 2: Bar — Avg MI per family (cross-generator average) ────────────
    logger.info("[Plot 2] Generating family_avg_mi_bar.png ...")
    avg_col = family_pivot_df.loc[FAMILY_ORDER_PLOT, "avg_across_generators"]
    colors = plt.cm.YlOrRd(
        [0.2 + 0.6 * (v - avg_col.min()) / (avg_col.max() - avg_col.min() + 1e-9)
         for v in avg_col.values]
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        FAMILY_ORDER_PLOT[::-1],
        avg_col[FAMILY_ORDER_PLOT[::-1]],
        color=colors[::-1],
        edgecolor="white",
        height=0.6,
    )
    for bar, val in zip(bars, avg_col[FAMILY_ORDER_PLOT[::-1]].values):
        ax.text(val + 0.0003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Average MI per Feature (cross-generator avg)", fontsize=10)
    ax.set_title(
        "Feature Family Ranking — Average MI per Feature\n"
        "(normalized; families ranked by cross-generator informativeness)",
        fontsize=11, fontweight="bold",
    )
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(fig_dir / "family_avg_mi_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", fig_dir / "family_avg_mi_bar.png")

    # ── Plot 3: Scatter — mean_mi vs std_mi, colored by family ───────────────
    logger.info("[Plot 3] Generating stability_scatter.png ...")
    family_colors = {fam: plt.cm.tab10(i) for i, fam in enumerate(FAMILIES)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for fam in FAMILIES:
        fam_data = full_stats_df[full_stats_df["family"] == fam]
        ax.scatter(
            fam_data["mean_mi"],
            fam_data["std_mi"],
            label=fam,
            color=family_colors[fam],
            alpha=0.4,
            s=12,
            rasterized=True,
        )

    ax.axvline(x=0.05, color="red", linestyle="--", linewidth=1.2,
               label="info_threshold (0.05)")
    ax.set_xlabel("Mean MI (across generators)", fontsize=11)
    ax.set_ylabel("Std MI (across generators)", fontsize=11)
    ax.set_title(
        "Feature Stability Scatter\n(mean_mi vs std_mi — colored by family)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8, markerscale=2, framealpha=0.8)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "stability_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", fig_dir / "stability_scatter.png")

    # ── Plot 4: Bar — Ablation F1 per family ─────────────────────────────────
    logger.info("[Plot 4] Generating ablation_f1_bar.png ...")
    abl_sorted = ablation_df.sort_values("avg_cross_gen_f1", ascending=True)
    f1_min = abl_sorted["avg_cross_gen_f1"].min()
    f1_max = abl_sorted["avg_cross_gen_f1"].max()
    abl_colors = plt.cm.Blues(
        [0.35 + 0.55 * (v - f1_min) / (f1_max - f1_min + 1e-9)
         for v in abl_sorted["avg_cross_gen_f1"].values]
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        abl_sorted["family"].tolist(),
        abl_sorted["avg_cross_gen_f1"].values,
        color=abl_colors,
        edgecolor="white",
        height=0.6,
    )
    for bar, val in zip(bars, abl_sorted["avg_cross_gen_f1"].values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Avg Off-Diagonal Cross-Generator F1 (LightGBM)", fontsize=10)
    ax.set_title(
        "Family-wise Ablation — Cross-Generator Generalization\n"
        "(LightGBM, off-diagonal pairs only)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(0, min(1.0, f1_max * 1.18))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(fig_dir / "ablation_f1_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", fig_dir / "ablation_f1_bar.png")

    logger.info("All 4 figures saved to: %s", fig_dir)


def run_exp4(
    features_root: Union[str, Path],
    output_dir: Union[str, Path],
    info_threshold: float = 0.05,
    specificity_ratio_threshold: float = 2.0,
    max_mi_threshold: float = 0.10,
    top_n: int = 20,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """
    Master entry point for Experiment 4 (RQ3): Universal vs. Generator-Specific Features.

    Chains all four phases end-to-end in a single call:
        Phase 1 — Metadata, generator-wise MI, full feature statistics
                  → Saves: mi_scores_per_generator.csv, feature_stability.csv
        Phase 2 — Feature analysis & family ranking
                  → Saves: universal_features.csv, generator_specific_features.csv,
                            feature_family_ranking.csv
        Phase 3 — LightGBM family ablation + Spearman rank correlation
                  → Saves: feature_family_ablation.csv
        Phase 4 — All RQ3 visualizations
                  → Saves: figures/mi_heatmap_families.png
                            figures/family_avg_mi_bar.png
                            figures/stability_scatter.png
                            figures/ablation_f1_bar.png

    Parameters
    ----------
    features_root : str or Path
        Root directory with train/, test/, val/, and metadata/ subdirs.
    output_dir : str or Path
        Directory where all outputs (CSVs + figures/) will be saved.
    info_threshold : float
        Min mean_mi for universal feature filter. Default: 0.05.
    specificity_ratio_threshold : float
        Min max_mi/mean_mi for generator-specific filter. Default: 2.0.
    max_mi_threshold : float
        Min max_mi for generator-specific filter. Default: 0.10.
    top_n : int
        Top N features to retain in universal and specific tables. Default: 20.
    random_seed : int
        Seed for MI computation and LightGBM.

    Returns
    -------
    dict with keys:
        feature_meta_df, mi_df, full_stats_df, universal_df, specific_df,
        family_detail_df, family_pivot_df, ablation_df, spearman_r, output_dir

    Colab usage
    -----------
    >>> from src.experiments.exp4_rq3 import run_exp4
    >>> results = run_exp4(
    ...     features_root="/content/drive/MyDrive/ml_project/processed/features/v1",
    ...     output_dir="/content/drive/MyDrive/ml_project/outputs/exp4_rq3",
    ... )
    """
    root     = Path(features_root)
    out_path = Path(output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("RQ3 — Experiment 4: Universal vs. Generator-Specific Features")
    logger.info("=" * 60)
    logger.info("features_root               : %s", root)
    logger.info("output_dir                  : %s", out_path)
    logger.info("info_threshold              = %.3f", info_threshold)
    logger.info("specificity_ratio_threshold = %.1f", specificity_ratio_threshold)
    logger.info("max_mi_threshold            = %.2f", max_mi_threshold)
    logger.info("top_n                       = %d",  top_n)
    logger.info("random_seed                 = %d",  random_seed)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    feature_meta_df, mi_df, full_stats_df = run_phase1(
        features_root=root,
        output_dir=out_path,
        random_seed=random_seed,
    )

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    universal_df, specific_df, family_detail_df, family_pivot_df = run_phase2(
        feature_meta_df=feature_meta_df,
        mi_df=mi_df,
        full_stats_df=full_stats_df,
        output_dir=out_path,
        info_threshold=info_threshold,
        specificity_ratio_threshold=specificity_ratio_threshold,
        max_mi_threshold=max_mi_threshold,
        top_n=top_n,
    )

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    ablation_df, spearman_r = run_phase3(
        features_root=root,
        output_dir=out_path,
        family_pivot_df=family_pivot_df,
        random_seed=random_seed,
    )

    # ── Phase 4: Visualizations ───────────────────────────────────────────────
    logger.info("\n[Phase 4] Generating RQ3 visualizations...")
    plot_rq3_figures(
        output_dir=out_path,
        family_pivot_df=family_pivot_df,
        full_stats_df=full_stats_df,
        ablation_df=ablation_df,
        family_detail_df=family_detail_df,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 4 (RQ3) Complete!")
    logger.info("Outputs saved to: %s", out_path)
    logger.info("  mi_scores_per_generator.csv     (D x 5 raw MI)")
    logger.info("  feature_stability.csv           (D x 13 complete stats)")
    logger.info("  universal_features.csv          (Top %d)", top_n)
    logger.info("  generator_specific_features.csv (Top %d)", top_n)
    logger.info("  feature_family_ranking.csv      (7 families, avg MI)")
    logger.info("  feature_family_ablation.csv     (7 families, cross-gen F1)")
    logger.info("  figures/ (4 plots)")
    logger.info("Key result: Spearman r = %.4f", spearman_r)
    logger.info("=" * 60)

    return {
        "feature_meta_df":  feature_meta_df,
        "mi_df":            mi_df,
        "full_stats_df":    full_stats_df,
        "universal_df":     universal_df,
        "specific_df":      specific_df,
        "family_detail_df": family_detail_df,
        "family_pivot_df":  family_pivot_df,
        "ablation_df":      ablation_df,
        "spearman_r":       spearman_r,
        "output_dir":       str(out_path),
    }


# =============================================================================
# Standalone execution guard (for local testing)
# =============================================================================

if __name__ == "__main__":
    results = run_exp4(
        features_root=DEFAULT_FEATURES_ROOT,
        output_dir=DEFAULT_OUTPUT_DIR,
    )

    print("\n" + "=" * 60)
    print("RQ3 — Final Summary")
    print("=" * 60)
    print("\nFeature Family Ranking (avg MI per feature):")
    print(results["family_pivot_df"].to_string())
    print("\nTop Universal Features:")
    print(results["universal_df"][
        ["feature_name", "family", "mean_mi", "stability_score"]
    ].to_string(index=False))
    print("\nTop Generator-Specific Features:")
    print(results["specific_df"][
        ["feature_name", "family", "max_mi", "specificity_ratio"]
    ].to_string(index=False))
    print("\nFamily Ablation (off-diagonal cross-gen F1):")
    print(results["ablation_df"][
        ["family", "avg_cross_gen_f1", "n_pairs"]
    ].to_string(index=False))
    print(f"\nSpearman r (MI vs F1 ranking) = {results['spearman_r']:.4f}")
    print(f"All outputs saved to: {results['output_dir']}")
