"""
features.py — Forensic feature extractors for AI-generated image detection.

Seven feature families:
    1. HOG     — Histogram of Oriented Gradients   (shape & edge structure)
    2. LBP     — Local Binary Patterns             (local texture)
    3. GLCM    — Grey-Level Co-occurrence Matrix   (statistical texture)
    4. DCT     — Discrete Cosine Transform         (frequency domain)
    5. Wavelet — Haar multi-scale decomposition    (multi-scale structure)
    6. Color   — Per-channel colour histogram      (colour distribution)
    7. Canny   — Edge density & contour statistics (edge structure)

Design rules:
    - Every extractor is a pure function: no I/O, no side effects, no global state.
    - Every extractor returns a 1-D float32 numpy array.
    - Every extractor validates its output immediately (NaN, Inf, shape).
    - No StandardScaler is applied here. Raw features are saved to disk.
      Scaling happens inside sklearn.Pipeline at training time.
    - Feature dimensions are auto-computed (never hardcoded).
    - Running the same inputs twice always produces identical outputs.

Feature family order is fixed in FAMILY_ORDER — this defines the concatenation
order for the combined vector and the layout recorded in feature_metadata.json.
"""

import logging
import warnings
from typing import Dict

import cv2
import numpy as np
import pywt
from scipy.fft import dctn
from skimage.feature import (
    graycomatrix,
    graycoprops,
    hog,
    local_binary_pattern,
)

from config import (
    CANNY_HIGH,
    CANNY_LOW,
    COLOR_HIST_BINS,
    DCT_BLOCK,
    GLCM_ANGLES,
    GLCM_DISTANCES,
    GLCM_PROPS,
    HOG_CELLS_PER_BLOCK,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    LBP_METHOD,
    LBP_N_BINS,
    LBP_POINTS,
    LBP_RADIUS,
    WAVELET_LEVELS,
    WAVELET_NAME,
)

logger = logging.getLogger(__name__)

# Fixed concatenation order — must not change between extraction runs.
FAMILY_ORDER = ["hog", "lbp", "glcm", "dct", "wavelet", "color", "canny"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate(vec: np.ndarray, name: str) -> np.ndarray:
    """Validate a 1-D float32 feature vector. Raise on NaN/Inf, warn on zero variance."""
    assert vec.ndim == 1, f"{name}: expected 1-D array, got shape {vec.shape}"
    assert not np.isnan(vec).any(),  f"{name}: contains NaN values"
    assert not np.isinf(vec).any(),  f"{name}: contains Inf values"
    if vec.var() == 0.0:
        logger.warning("%s: zero variance — feature may carry no information", name)
    return vec


def _safe_stats(arr: np.ndarray):
    """Return (mean, std) of a flattened array, guarding against empty arrays."""
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std())


# ─────────────────────────────────────────────────────────────────────────────
# 1. HOG — Histogram of Oriented Gradients
# ─────────────────────────────────────────────────────────────────────────────

def extract_hog(gray_f32: np.ndarray) -> np.ndarray:
    """
    Extract HOG features.

    Parameters
    ----------
    gray_f32 : np.ndarray
        float32 grayscale image, shape (H, W), values in [0, 1].

    Returns
    -------
    np.ndarray
        1-D float32 HOG descriptor.
        Dimension depends on IMAGE_SIZE and HOG_* config parameters.
        For 128×128 with pixels_per_cell=(8,8), cells_per_block=(2,2),
        orientations=9: dim = 15×15×4×9 = 8100.
    """
    fd = hog(
        gray_f32,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=False,
        feature_vector=True,
        channel_axis=None,
    )
    return _validate(fd.astype(np.float32), "hog")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LBP — Local Binary Patterns
# ─────────────────────────────────────────────────────────────────────────────

def extract_lbp(gray_f32: np.ndarray) -> np.ndarray:
    """
    Extract a normalised LBP histogram.

    Parameters
    ----------
    gray_f32 : np.ndarray
        float32 grayscale image, shape (H, W), values in [0, 1].

    Returns
    -------
    np.ndarray
        1-D float32 normalised histogram of length LBP_N_BINS.
    """
    lbp_map = local_binary_pattern(
        gray_f32,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method=LBP_METHOD,
    )
    # Number of uniform patterns = P*(P-1)+3 = 8*7+3 = 59 → we histogram into LBP_N_BINS
    hist, _ = np.histogram(lbp_map.ravel(), bins=LBP_N_BINS, range=(0, LBP_POINTS + 2))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total   # L1 normalisation → sum to 1
    return _validate(hist, "lbp")


# ─────────────────────────────────────────────────────────────────────────────
# 3. GLCM — Grey-Level Co-occurrence Matrix
# ─────────────────────────────────────────────────────────────────────────────

def extract_glcm(gray_u8: np.ndarray) -> np.ndarray:
    """
    Extract GLCM texture properties.

    Computes properties for each (distance, angle) pair, then concatenates.

    Parameters
    ----------
    gray_u8 : np.ndarray
        uint8 grayscale image, shape (H, W), values in [0, 255].

    Returns
    -------
    np.ndarray
        1-D float32 vector of length:
        len(GLCM_PROPS) × len(GLCM_DISTANCES) × len(GLCM_ANGLES)
        = 5 × 2 × 4 = 40 (with default config).
    """
    glcm = graycomatrix(
        gray_u8,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=256,
        symmetric=True,
        normed=True,
    )
    features = []
    for prop in GLCM_PROPS:
        result = graycoprops(glcm, prop)   # shape: (n_distances, n_angles)
        features.append(result.ravel())

    vec = np.concatenate(features).astype(np.float32)
    return _validate(vec, "glcm")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DCT — Discrete Cosine Transform
# ─────────────────────────────────────────────────────────────────────────────

def extract_dct(gray_f32: np.ndarray) -> np.ndarray:
    """
    Extract low-frequency DCT coefficients.

    Computes the 2-D DCT of the full image, then retains the top-left
    DCT_BLOCK × DCT_BLOCK coefficients (low-frequency energy).

    Parameters
    ----------
    gray_f32 : np.ndarray
        float32 grayscale image, shape (H, W), values in [0, 1].

    Returns
    -------
    np.ndarray
        1-D float32 vector of length DCT_BLOCK² (default: 64).
    """
    dct_full = dctn(gray_f32, norm="ortho")          # shape: (H, W)
    block    = dct_full[:DCT_BLOCK, :DCT_BLOCK]      # top-left N×N
    vec      = block.ravel().astype(np.float32)
    return _validate(vec, "dct")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Wavelet — Haar multi-scale decomposition
# ─────────────────────────────────────────────────────────────────────────────

def extract_wavelet(gray_f32: np.ndarray) -> np.ndarray:
    """
    Extract Haar wavelet statistics across multiple decomposition levels.

    For each sub-band (approximation + detail bands at each level),
    computes mean and standard deviation.

    Parameters
    ----------
    gray_f32 : np.ndarray
        float32 grayscale image, shape (H, W), values in [0, 1].

    Returns
    -------
    np.ndarray
        1-D float32 vector.
        Length = 2 stats × (1 approx + 3 detail types × WAVELET_LEVELS)
               = 2 × (1 + 3×3) = 20 (with WAVELET_LEVELS=3).
    """
    coeffs = pywt.wavedec2(gray_f32, wavelet=WAVELET_NAME, level=WAVELET_LEVELS)
    # coeffs = [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]

    features = []

    # Approximation coefficients at the deepest level
    mean, std = _safe_stats(coeffs[0])
    features.extend([mean, std])

    # Detail coefficient tuples at each level (from deepest to shallowest)
    for detail_tuple in coeffs[1:]:
        for sub_band in detail_tuple:   # LH, HL, HH
            m, s = _safe_stats(sub_band)
            features.extend([m, s])

    vec = np.array(features, dtype=np.float32)
    return _validate(vec, "wavelet")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Color Histogram
# ─────────────────────────────────────────────────────────────────────────────

def extract_color_histogram(rgb_f32: np.ndarray) -> np.ndarray:
    """
    Extract a normalised per-channel colour histogram.

    Parameters
    ----------
    rgb_f32 : np.ndarray
        float32 RGB image, shape (H, W, 3), values in [0, 1].

    Returns
    -------
    np.ndarray
        1-D float32 concatenated histogram of length 3 × COLOR_HIST_BINS
        (default: 96). Each channel histogram is L1-normalised independently.
    """
    features = []
    for ch in range(3):
        hist, _ = np.histogram(
            rgb_f32[:, :, ch].ravel(),
            bins=COLOR_HIST_BINS,
            range=(0.0, 1.0),
        )
        hist = hist.astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        features.append(hist)

    vec = np.concatenate(features).astype(np.float32)
    return _validate(vec, "color")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Canny — Edge density & contour statistics
# ─────────────────────────────────────────────────────────────────────────────

def extract_canny(gray_u8: np.ndarray) -> np.ndarray:
    """
    Extract scalar edge and contour statistics from a Canny edge map.

    Six statistics are computed:
        1. edge_density         — fraction of pixels that are edges
        2. n_contours_norm      — number of contours / image area
        3. mean_contour_area    — mean contour area / image area
        4. std_contour_area     — std of contour areas / image area
        5. mean_contour_perim   — mean contour perimeter / image perimeter
        6. std_contour_perim    — std of contour perimeters / image perimeter

    Parameters
    ----------
    gray_u8 : np.ndarray
        uint8 grayscale image, shape (H, W), values in [0, 255].

    Returns
    -------
    np.ndarray
        1-D float32 vector of length 6.
    """
    edges = cv2.Canny(gray_u8, CANNY_LOW, CANNY_HIGH)

    H, W = edges.shape
    image_area      = float(H * W)
    image_perimeter = float(2 * (H + W))

    # Stat 1: edge density
    edge_density = float((edges > 0).sum()) / image_area

    # Find contours on the edge map
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    n_contours = len(contours)
    if n_contours > 0:
        areas = np.array([cv2.contourArea(c) for c in contours], dtype=np.float64)
        perims = np.array([cv2.arcLength(c, True) for c in contours], dtype=np.float64)
        mean_area  = float(areas.mean())  / image_area
        std_area   = float(areas.std())   / image_area
        mean_perim = float(perims.mean()) / image_perimeter
        std_perim  = float(perims.std())  / image_perimeter
    else:
        mean_area = std_area = mean_perim = std_perim = 0.0

    vec = np.array([
        edge_density,
        n_contours / image_area,
        mean_area,
        std_area,
        mean_perim,
        std_perim,
    ], dtype=np.float32)

    return _validate(vec, "canny")


# ─────────────────────────────────────────────────────────────────────────────
# Combiner — single entry point for the runner
# ─────────────────────────────────────────────────────────────────────────────

def combine_features(
    gray_f32: np.ndarray,
    gray_u8:  np.ndarray,
    rgb_f32:  np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Extract all seven feature families and return them as a dict.

    The 'combined' key holds the full concatenated vector. Its dimension D
    is whatever the concatenation produces — auto-recorded downstream.

    Parameters
    ----------
    gray_f32 : np.ndarray  float32 grayscale (H, W)  [0, 1]
    gray_u8  : np.ndarray  uint8   grayscale (H, W)  [0, 255]
    rgb_f32  : np.ndarray  float32 RGB       (H, W, 3) [0, 1]

    Returns
    -------
    dict with keys: "hog", "lbp", "glcm", "dct", "wavelet", "color", "canny", "combined"
    Each value is a 1-D float32 np.ndarray.
    """
    result = {
        "hog":     extract_hog(gray_f32),
        "lbp":     extract_lbp(gray_f32),
        "glcm":    extract_glcm(gray_u8),
        "dct":     extract_dct(gray_f32),
        "wavelet": extract_wavelet(gray_f32),
        "color":   extract_color_histogram(rgb_f32),
        "canny":   extract_canny(gray_u8),
    }
    result["combined"] = np.concatenate(
        [result[name] for name in FAMILY_ORDER], axis=0
    ).astype(np.float32)
    return result


def build_feature_metadata(sample_result: Dict[str, np.ndarray]) -> dict:
    """
    Build the feature_metadata.json dict from a single dry-run result.

    This is called once after the dry-run to compute all boundaries
    automatically — no hardcoding.

    Parameters
    ----------
    sample_result : dict
        Output of combine_features() for one image.

    Returns
    -------
    dict
        Suitable for json.dump(), records start/end/dim for every family
        and total_dims.
    """
    cursor = 0
    families = {}
    for name in FAMILY_ORDER:
        dim = len(sample_result[name])
        families[name] = {
            "start": cursor,
            "end":   cursor + dim - 1,
            "dims":  dim,
        }
        cursor += dim

    return {
        "families":   families,
        "total_dims": cursor,
    }


def build_feature_names(sample_result: Dict[str, np.ndarray]) -> list:
    """
    Build a human-readable name for every feature dimension.

    These names are used in SHAP plots, Mutual Information tables, etc.
    so that outputs show 'wavelet_LL3_mean' instead of 'Feature 421'.

    Parameters
    ----------
    sample_result : dict
        Output of combine_features() for one image.

    Returns
    -------
    list of str
        One name per dimension of the combined vector.
    """
    names = []

    # 1. HOG
    n_hog = len(sample_result["hog"])
    names += [f"hog_{i:04d}" for i in range(n_hog)]

    # 2. LBP
    n_lbp = len(sample_result["lbp"])
    names += [f"lbp_uniform_{i:02d}" for i in range(n_lbp)]

    # 3. GLCM — pattern: glcm_{prop}_d{distance}_a{angle_deg}
    for prop in GLCM_PROPS:
        for dist in GLCM_DISTANCES:
            for angle_rad in GLCM_ANGLES:
                angle_deg = int(round(np.degrees(angle_rad)))
                names.append(f"glcm_{prop}_d{dist}_a{angle_deg}")

    # 4. DCT
    n_dct = DCT_BLOCK * DCT_BLOCK
    names += [f"dct_{i:02d}" for i in range(n_dct)]

    # 5. Wavelet — pattern: wavelet_{band}_{level}_{stat}
    wavelet_names = []
    level_tag = WAVELET_LEVELS
    # Approximation at deepest level
    for stat in ("mean", "std"):
        wavelet_names.append(f"wavelet_LL{level_tag}_{stat}")
    # Detail bands: LH, HL, HH at each level (deepest→shallowest)
    band_tags = ["LH", "HL", "HH"]
    for lvl in range(WAVELET_LEVELS, 0, -1):
        for band in band_tags:
            for stat in ("mean", "std"):
                wavelet_names.append(f"wavelet_{band}{lvl}_{stat}")
    names += wavelet_names

    # 6. Color histogram — pattern: color_{channel}_bin{bin}
    channel_tags = ["R", "G", "B"]
    for ch in channel_tags:
        for b in range(COLOR_HIST_BINS):
            names.append(f"color_{ch}_bin{b:02d}")

    # 7. Canny
    names += [
        "canny_edge_density",
        "canny_n_contours_norm",
        "canny_mean_contour_area",
        "canny_std_contour_area",
        "canny_mean_contour_perim",
        "canny_std_contour_perim",
    ]

    # Sanity check: names count must equal combined vector length
    expected = len(sample_result["combined"])
    assert len(names) == expected, (
        f"Feature name count mismatch: got {len(names)} names "
        f"but combined vector has {expected} dims"
    )
    return names
