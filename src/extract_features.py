"""
extract_features.py — Production feature extraction runner.

This script is the ONLY code that should be run to perform feature extraction.
The Colab notebook calls this script via !python; it does not contain this logic.

Usage (inside Google Colab after mounting Drive and cloning the repo):

    !python /content/repo/src/extract_features.py \\
        --splits_dir  "/content/drive/MyDrive/ML_Project/processed/splits" \\
        --image_root  "/content/drive/MyDrive/ML_Project/images" \\
        --out_root    "/content/drive/MyDrive/ML_Project/processed/features" \\
        --version     v1

All CLI arguments are optional and default to the values in config.py.

Output layout under <out_root>/<version>/:
    full/            — all images combined
    train/ val/ test/ — per-split subsets
    metadata/        — feature_metadata.json, feature_names.txt,
                       feature_stats.csv, preprocessing.json,
                       timing_report.json, manifest.json,
                       feature_report.txt, failed_images.txt,
                       feature_extraction.log
    figures/         — PCA scatter, class balance, variance plots
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# ── Ensure src/ is on sys.path when called from repo root ─────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CANNY_HIGH, CANNY_LOW, COLOR_HIST_BINS, DCT_BLOCK,
    FEATURE_ROOT, FEATURE_VERSION, GENERATORS,
    GLCM_ANGLES, GLCM_DISTANCES, GLCM_PROPS,
    HOG_CELLS_PER_BLOCK, HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL,
    IMAGE_ROOT, IMAGE_SIZE, LBP_METHOD, LBP_N_BINS, LBP_POINTS, LBP_RADIUS,
    N_JOBS, RANDOM_SEED, SPLITS_DIR, WAVELET_LEVELS, WAVELET_NAME,
)
from dataset import load_splits
from features import (
    FAMILY_ORDER,
    build_feature_metadata,
    build_feature_names,
    combine_features,
)
from utils import prepare_image


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    """Configure root logger to write to both console and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt,
                        handlers=handlers)
    return logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_output_dirs(out_root: Path) -> dict:
    """Create all output subdirectories and return a dict of paths."""
    dirs = {
        "full":     out_root / "full",
        "train":    out_root / "train",
        "val":      out_root / "val",
        "test":     out_root / "test",
        "metadata": out_root / "metadata",
        "figures":  out_root / "figures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ─────────────────────────────────────────────────────────────────────────────
# Per-image extraction (called in parallel)
# ─────────────────────────────────────────────────────────────────────────────

def extract_one(row_tuple) -> dict:
    """
    Extract features for a single image row.

    Parameters
    ----------
    row_tuple : namedtuple
        A row from df.itertuples(). Must have:
        .Index, .filename, .generator, .label_binary, .split, .image_path

    Returns
    -------
    dict with keys:
        "ok"       — bool: True if successful
        "filename" — str
        "features" — dict from combine_features(), or None on failure
        "timings"  — dict of per-family extraction times, or {}
        "error"    — str error message on failure, or None
    """
    row = row_tuple
    try:
        rgb_u8, gray_u8, gray_f32, rgb_f32 = prepare_image(Path(row.image_path))

        # Time each family
        t_start = {}
        t_end   = {}

        t0 = time.perf_counter()
        from features import (
            extract_hog, extract_lbp, extract_glcm,
            extract_dct, extract_wavelet, extract_color_histogram, extract_canny,
        )

        results = {}
        for name, fn, arg in [
            ("hog",     extract_hog,              gray_f32),
            ("lbp",     extract_lbp,              gray_f32),
            ("glcm",    extract_glcm,             gray_u8),
            ("dct",     extract_dct,              gray_f32),
            ("wavelet", extract_wavelet,           gray_f32),
            ("color",   extract_color_histogram,  rgb_f32),
            ("canny",   extract_canny,            gray_u8),
        ]:
            t_s = time.perf_counter()
            results[name] = fn(arg)
            t_end[name] = time.perf_counter() - t_s

        import numpy as np
        results["combined"] = np.concatenate(
            [results[n] for n in FAMILY_ORDER], axis=0
        ).astype(np.float32)

        return {
            "ok":       True,
            "filename": row.filename,
            "features": results,
            "timings":  t_end,
            "error":    None,
        }

    except Exception as exc:
        return {
            "ok":       False,
            "filename": row.filename,
            "features": None,
            "timings":  {},
            "error":    f"{type(exc).__name__}: {exc}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_split_arrays(
    out_dir: Path,
    combined: np.ndarray,
    family_arrays: dict,
    labels: np.ndarray,
    generators: np.ndarray,
    splits: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Save feature arrays for a subset defined by a boolean mask."""
    idx = np.where(mask)[0]
    np.save(out_dir / "combined.npy",  combined[idx].astype(np.float32))
    np.save(out_dir / "labels.npy",    labels[idx].astype(np.int8))
    np.save(out_dir / "generators.npy", generators[idx])
    np.save(out_dir / "splits.npy",    splits[idx])
    for name in FAMILY_ORDER:
        np.save(out_dir / f"{name}.npy", family_arrays[name][idx].astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_root   = Path(args.out_root) / args.version
    splits_dir = Path(args.splits_dir)
    image_root = Path(args.image_root)

    # ── Setup logging ─────────────────────────────────────────────────────────
    log_path = out_root / "metadata" / "feature_extraction.log"
    logger   = setup_logging(log_path)
    logger.info("=" * 60)
    logger.info("Feature Extraction Pipeline — version %s", args.version)
    logger.info("=" * 60)
    logger.info("Output root : %s", out_root)
    logger.info("Splits dir  : %s", splits_dir)
    logger.info("Image root  : %s", image_root)

    # ── Create directories ────────────────────────────────────────────────────
    dirs = make_output_dirs(out_root)

    # ── Write preprocessing.json immediately (config snapshot) ────────────────
    preprocessing = {
        "image_size":       list(IMAGE_SIZE),
        "color_space":      "RGB",
        "gray_conversion":  "OpenCV ITU-R 601 weighted",
        "normalize":        False,
        "hog_pixels_per_cell": list(HOG_PIXELS_PER_CELL),
        "hog_cells_per_block": list(HOG_CELLS_PER_BLOCK),
        "hog_orientations":    HOG_ORIENTATIONS,
        "lbp_radius":    LBP_RADIUS,
        "lbp_points":    LBP_POINTS,
        "lbp_method":    LBP_METHOD,
        "lbp_n_bins":    LBP_N_BINS,
        "glcm_distances": GLCM_DISTANCES,
        "glcm_angles":    GLCM_ANGLES,
        "glcm_props":     GLCM_PROPS,
        "dct_block":      DCT_BLOCK,
        "wavelet_name":   WAVELET_NAME,
        "wavelet_levels": WAVELET_LEVELS,
        "color_hist_bins": COLOR_HIST_BINS,
        "canny_low":      CANNY_LOW,
        "canny_high":     CANNY_HIGH,
        "random_seed":    RANDOM_SEED,
        "version":        args.version,
        "extraction_date": datetime.now().strftime("%Y-%m-%d"),
    }
    with open(dirs["metadata"] / "preprocessing.json", "w") as f:
        json.dump(preprocessing, f, indent=2)
    logger.info("preprocessing.json written.")

    # ── Load split CSVs ───────────────────────────────────────────────────────
    logger.info("Loading split CSVs from %s ...", splits_dir)
    df = load_splits(splits_dir)
    logger.info("Total images in DataFrame: %d", len(df))
    logger.info("Class balance:\n%s",
                df.groupby(["generator", "label_binary"]).size().to_string())

    # ── DRY RUN — 3 images ───────────────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("DRY RUN: extracting features from 3 images to compute dimensions ...")
    dry_df = df.head(3)
    dry_result = None
    for row in dry_df.itertuples():
        try:
            rgb_u8, gray_u8, gray_f32, rgb_f32 = prepare_image(Path(row.image_path))
            dry_result = combine_features(gray_f32, gray_u8, rgb_f32)
            break
        except Exception as exc:
            logger.warning("Dry-run image %s failed: %s", row.filename, exc)

    if dry_result is None:
        logger.error("All dry-run images failed. Cannot determine feature dimensions.")
        sys.exit(1)

    # Auto-compute dimensions from dry run
    feature_meta  = build_feature_metadata(dry_result)
    feature_names = build_feature_names(dry_result)
    TOTAL_DIM     = feature_meta["total_dims"]

    logger.info("Feature dimensions (auto-computed):")
    for name, info in feature_meta["families"].items():
        logger.info("  %-10s %d dims  [%d:%d]", name, info["dims"], info["start"], info["end"])
    logger.info("  TOTAL      %d dims", TOTAL_DIM)
    logger.info("  Feature names list length: %d", len(feature_names))

    assert len(feature_names) == TOTAL_DIM, (
        f"Feature name count ({len(feature_names)}) != total dims ({TOTAL_DIM})"
    )
    logger.info("Dry-run passed. Proceeding to full extraction.")

    # ── FULL EXTRACTION ───────────────────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Starting full extraction: %d images, n_jobs=%d ...", len(df), N_JOBS)

    rows       = list(df.itertuples())
    t_pipeline_start = time.time()

    raw_results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
        delayed(extract_one)(row) for row in tqdm(rows, desc="Extracting", unit="img")
    )

    t_pipeline_end = time.time()
    runtime_sec    = t_pipeline_end - t_pipeline_start
    logger.info("Extraction loop completed in %.1f seconds (%.1f min)",
                runtime_sec, runtime_sec / 60)

    # ── Separate successful and failed ────────────────────────────────────────
    succeeded = [r for r in raw_results if r["ok"]]
    failed    = [r for r in raw_results if not r["ok"]]

    logger.info("Processed: %d | Failed: %d", len(succeeded), len(failed))

    # Write failed_images.txt
    failed_path = dirs["metadata"] / "failed_images.txt"
    with open(failed_path, "w") as f:
        f.write(f"# Failed images — {datetime.now().isoformat()}\n")
        f.write(f"# Total failed: {len(failed)}\n\n")
        for rec in failed:
            f.write(f"FILE:  {rec['filename']}\n")
            f.write(f"ERROR: {rec['error']}\n")
            f.write(f"TIME:  {datetime.now().isoformat()}\n\n")
    logger.info("failed_images.txt written (%d entries).", len(failed))

    if len(succeeded) == 0:
        logger.error("No images were successfully processed. Aborting.")
        sys.exit(1)

    # ── Build arrays — preserve CSV row order ─────────────────────────────────
    # Map filename → result for ordered assembly
    result_map = {r["filename"]: r for r in succeeded}

    combined_list   = []
    family_lists    = {name: [] for name in FAMILY_ORDER}
    labels_list     = []
    generators_list = []
    splits_list     = []
    filenames_list  = []

    skipped = 0
    for row in rows:
        if row.filename not in result_map:
            skipped += 1
            continue
        rec = result_map[row.filename]
        combined_list.append(rec["features"]["combined"])
        for name in FAMILY_ORDER:
            family_lists[name].append(rec["features"][name])
        labels_list.append(int(row.label_binary))
        generators_list.append(row.generator)
        splits_list.append(row.split)
        filenames_list.append(row.filename)

    N = len(combined_list)
    logger.info("Assembling arrays for %d images (skipped %d failed).", N, skipped)

    combined_arr   = np.stack(combined_list, axis=0).astype(np.float32)
    family_arrays  = {
        name: np.stack(family_lists[name], axis=0).astype(np.float32)
        for name in FAMILY_ORDER
    }
    labels_arr     = np.array(labels_list, dtype=np.int8)
    generators_arr = np.array(generators_list, dtype="U20")
    splits_arr     = np.array(splits_list,     dtype="U5")

    # ── ASSERTIONS ────────────────────────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Running assertions ...")

    assert combined_arr.shape == (N, TOTAL_DIM), (
        f"Shape mismatch: expected ({N}, {TOTAL_DIM}), got {combined_arr.shape}"
    )
    nan_count = int(np.isnan(combined_arr).sum())
    inf_count = int(np.isinf(combined_arr).sum())
    assert nan_count == 0,  f"combined.npy contains {nan_count} NaN values"
    assert inf_count == 0,  f"combined.npy contains {inf_count} Inf values"

    for name in FAMILY_ORDER:
        assert family_arrays[name].shape[0] == N, (
            f"{name}.npy row count mismatch: {family_arrays[name].shape[0]} vs {N}"
        )

    # Spot-check 50 random rows: combined == concatenation of families
    rng        = np.random.default_rng(RANDOM_SEED)
    check_idxs = rng.choice(N, size=min(50, N), replace=False)
    for idx in check_idxs:
        expected = np.concatenate([family_arrays[name][idx] for name in FAMILY_ORDER])
        assert np.allclose(combined_arr[idx], expected, atol=1e-6), (
            f"Row {idx}: combined != concatenation of families"
        )

    logger.info("All assertions passed. combined shape: %s, NaN: 0, Inf: 0",
                combined_arr.shape)

    # ── SAVE: full/ ───────────────────────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Saving full/ arrays ...")
    np.save(dirs["full"] / "combined.npy",   combined_arr)
    np.save(dirs["full"] / "labels.npy",     labels_arr)
    np.save(dirs["full"] / "generators.npy", generators_arr)
    np.save(dirs["full"] / "splits.npy",     splits_arr)
    for name in FAMILY_ORDER:
        np.save(dirs["full"] / f"{name}.npy", family_arrays[name])
    logger.info("full/ arrays saved.")

    # ── SAVE: train/ val/ test/ ───────────────────────────────────────────────
    logger.info("Saving per-split arrays ...")
    for split_name in ("train", "val", "test"):
        mask = splits_arr == split_name
        save_split_arrays(
            dirs[split_name], combined_arr, family_arrays,
            labels_arr, generators_arr, splits_arr, mask,
        )
        logger.info("  %s: %d rows saved.", split_name, int(mask.sum()))

    # ── SAVE: metadata/ ───────────────────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Saving metadata files ...")

    # metadata.csv
    meta_df = pd.DataFrame({
        "filename":     filenames_list,
        "generator":    generators_arr.tolist(),
        "label_binary": labels_arr.tolist(),
        "split":        splits_arr.tolist(),
    })
    meta_df.to_csv(dirs["metadata"] / "metadata.csv", index=False)

    # feature_metadata.json  — boundaries + dims, all auto-computed
    feature_meta["version"]        = args.version
    feature_meta["extraction_date"] = datetime.now().strftime("%Y-%m-%d")
    with open(dirs["metadata"] / "feature_metadata.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    # feature_names.txt
    with open(dirs["metadata"] / "feature_names.txt", "w") as f:
        f.write("\n".join(feature_names) + "\n")

    # feature_stats.csv  — per-column statistics of the combined array
    stats_df = pd.DataFrame({
        "feature_name": feature_names,
        "mean":  combined_arr.mean(axis=0).tolist(),
        "std":   combined_arr.std(axis=0).tolist(),
        "min":   combined_arr.min(axis=0).tolist(),
        "max":   combined_arr.max(axis=0).tolist(),
        "var":   combined_arr.var(axis=0).tolist(),
    })
    stats_df.to_csv(dirs["metadata"] / "feature_stats.csv", index=False)

    # timing_report.json  — mean extraction time per family
    all_timings = {name: [] for name in FAMILY_ORDER}
    for rec in succeeded:
        for name, t in rec["timings"].items():
            if name in all_timings:
                all_timings[name].append(t)
    timing_summary = {
        name: {
            "mean_sec": round(float(np.mean(times)), 5),
            "std_sec":  round(float(np.std(times)), 5),
        }
        for name, times in all_timings.items() if times
    }
    with open(dirs["metadata"] / "timing_report.json", "w") as f:
        json.dump(timing_summary, f, indent=2)

    # manifest.json
    manifest = {
        "dataset":             "MS COCOAI",
        "images_total":        len(df),
        "images_processed":    N,
        "images_failed":       len(failed),
        "feature_version":     args.version,
        "image_size":          list(IMAGE_SIZE),
        "total_dimensions":    TOTAL_DIM,
        "dtype":               "float32",
        "random_seed":         RANDOM_SEED,
        "created":             datetime.now().strftime("%Y-%m-%d"),
        "runtime_minutes":     round(runtime_sec / 60, 2),
        "generators":          GENERATORS,
        "split_counts": {
            split_name: int((splits_arr == split_name).sum())
            for split_name in ("train", "val", "test")
        },
    }
    with open(dirs["metadata"] / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # feature_report.txt
    report_lines = [
        "=" * 50,
        f"  Feature Extraction Report — {args.version}",
        "=" * 50,
        f"Images total      : {len(df)}",
        f"Images processed  : {N}",
        f"Images failed     : {len(failed)}",
        f"Total dimensions  : {TOTAL_DIM}",
        "",
        "Family breakdown:",
    ]
    for name, info in feature_meta["families"].items():
        report_lines.append(f"  {name:<12} : {info['dims']:>5} dims")
    report_lines += [
        "",
        f"NaN values        : 0",
        f"Inf values        : 0",
        "",
        "Split counts:",
    ]
    for split_name in ("train", "val", "test"):
        cnt = int((splits_arr == split_name).sum())
        report_lines.append(f"  {split_name:<6} : {cnt}")
    report_lines += [
        "",
        f"Runtime           : {runtime_sec/60:.1f} min",
        f"Extraction date   : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 50,
    ]
    with open(dirs["metadata"] / "feature_report.txt", "w") as f:
        f.write("\n".join(report_lines) + "\n")

    logger.info("All metadata files saved.")
    logger.info("-" * 40)
    logger.info("Phase 2 feature extraction COMPLETE.")
    logger.info("Output: %s", out_root)
    logger.info("Total dims: %d | Processed: %d | Failed: %d",
                TOTAL_DIM, N, len(failed))
    logger.info("Runtime: %.1f min", runtime_sec / 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2 Feature Extraction — Cross-Generator AI Image Detection"
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=str(SPLITS_DIR),
        help="Directory containing train.csv, val.csv, test.csv",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=str(IMAGE_ROOT),
        help="Root directory containing per-generator image folders",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(FEATURE_ROOT),
        help="Root output directory (version subfolder will be created inside)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=FEATURE_VERSION,
        help="Feature version tag (e.g. v1, v2). Creates <out_root>/<version>/",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
