"""
config.py — Single source of truth for all feature extraction parameters.
Every other module imports from here. Changing one value updates the whole pipeline.
Bump FEATURE_VERSION to preserve previous experiment outputs.
"""

from pathlib import Path

# ── Google Drive paths ────────────────────────────────────────────────────────
# These are the runtime paths when running inside Google Colab.
DRIVE_ROOT   = Path("/content/drive/MyDrive/ml_project")
IMAGE_ROOT   = DRIVE_ROOT / "images"
SPLITS_DIR   = DRIVE_ROOT / "processed" / "splits"
FEATURE_ROOT = DRIVE_ROOT / "processed" / "features"

# Generator class names — order is fixed and used everywhere.
GENERATORS = ["Real", "SD21", "SDXL", "SD3", "DALLE3", "Midjourney"]

# ── Versioning ────────────────────────────────────────────────────────────────
# Changing any parameter below → bump FEATURE_VERSION → old run is preserved.
FEATURE_VERSION = "v1"

# ── Reproducibility ───────────────────────────────────────────────────────────
# Used by all phases (training, SHAP, cross-validation, etc.)
RANDOM_SEED = 42

# ── Image preprocessing ───────────────────────────────────────────────────────
IMAGE_SIZE  = (128, 128)   # (H, W) — fixed for the entire project
COLOR_SPACE = "RGB"
# IMPORTANT: Raw features are saved. StandardScaler is applied *inside*
# sklearn.Pipeline at training time to prevent data leakage.
NORMALIZE = False

# ── HOG — Histogram of Oriented Gradients ─────────────────────────────────────
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS    = 9

# ── LBP — Local Binary Patterns ───────────────────────────────────────────────
LBP_RADIUS = 1
LBP_POINTS = 8
LBP_METHOD = "uniform"
LBP_N_BINS = 10

# ── GLCM — Grey-Level Co-occurrence Matrix ────────────────────────────────────
GLCM_DISTANCES = [1, 2]
GLCM_ANGLES    = [0.0, 0.7854, 1.5708, 2.3562]   # 0, π/4, π/2, 3π/4
GLCM_PROPS     = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
# Dimension = len(GLCM_PROPS) × len(GLCM_DISTANCES) × len(GLCM_ANGLES) = 5×2×4 = 40

# ── DCT — Discrete Cosine Transform ──────────────────────────────────────────
DCT_BLOCK = 8   # Top-left DCT_BLOCK × DCT_BLOCK coefficients → DCT_BLOCK² dims = 64

# ── Wavelet — Haar multi-scale decomposition ──────────────────────────────────
WAVELET_NAME   = "haar"
WAVELET_LEVELS = 3
# Sub-bands per level: LL (approx), LH, HL, HH; stats per band: mean + std
# Dim = (1 approx + 3 detail levels × 3 bands) × 2 stats = 20

# ── Color Histogram ───────────────────────────────────────────────────────────
COLOR_HIST_BINS = 32   # Bins per channel; 3 channels → 3 × 32 = 96 dims

# ── Canny Edge ────────────────────────────────────────────────────────────────
CANNY_LOW  = 50
CANNY_HIGH = 150
# 6 scalar statistics extracted (see features.py for details)

# ── Runtime ───────────────────────────────────────────────────────────────────
N_JOBS = -1   # All available CPU cores (passed to joblib.Parallel)
