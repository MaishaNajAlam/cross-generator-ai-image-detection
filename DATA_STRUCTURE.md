# Cross-Generator AI Image Detection — Data Structure Guide

This guide describes the directory structure, file meanings, data formats, and usage instructions for the dataset and extracted features stored on Google Drive. 

All scripts are configured to read and write to the Google Drive folder located at:
`/content/drive/MyDrive/ml_project/`

---

## 1. Directory Tree Overview

```text
ml_project/
├── images/                             # Raw dataset images
│   ├── Real/                           # MS COCO real images (Real_0.jpg, Real_799.jpg)
│   ├── SD21/                           # Stable Diffusion v2.1 fake images
│   ├── SDXL/                           # Stable Diffusion XL fake images
│   ├── SD3/                            # Stable Diffusion v3 fake images
│   ├── DALLE3/                         # DALL-E 3 fake images
│   └── Midjourney/                     # Midjourney v6 fake images
│
├── processed/
│   ├── splits/                         # Deterministic dataset splits (Phase 1)
│   │   ├── train.csv                   # 3,360 images (560 per generator)
│   │   ├── val.csv                     # 720 images (120 per generator)
│   │   └── test.csv                    # 720 images (120 per generator)
│   │
│   └── features/                       # Extracted forensic features (Phase 2)
│       └── v1/                         # Versioned run (e.g. v1, v2)
│           ├── full/                   # Entire dataset concatenated
│           ├── train/                  # Train-split subsets
│           ├── val/                    # Val-split subsets
│           ├── test/                   # Test-split subsets
│           ├── metadata/               # Manifest, boundaries, stats, logs
│           └── figures/                # Verification plots
```

---

## 2. File-by-File Breakdown (Google Drive)

### 📂 `processed/splits/`
These files are the single source of truth for train/validation/test membership.
*   **`train.csv`**, **`val.csv`**, **`test.csv`**: Contains three columns:
    *   `filename`: The permanent unique ID (e.g. `SDXL_456.jpg`).
    *   `generator`: The source generator class (`Real`, `SD21`, `SDXL`, `SD3`, `DALLE3`, `Midjourney`).
    *   `label_binary`: `0` for Real, `1` for Fake.

---

### 📂 `processed/features/v1/full/` (Combined Dataset)
Contains features and labels for all $N = 4,800$ images combined, keeping the exact CSV order (train $\rightarrow$ val $\rightarrow$ test). All feature matrices are saved as raw values (no scaling applied yet).
*   **`combined.npy`**: Float32 matrix of shape `(4800, D)` containing all 7 concatenated feature families.
*   **`hog.npy`**: HOG feature matrix of shape `(4800, 8100)`.
*   **`lbp.npy`**: LBP histogram matrix of shape `(4800, 10)`.
*   **`glcm.npy`**: GLCM texture metrics matrix of shape `(4800, 40)`.
*   **`dct.npy`**: Top-left DCT frequency coefficients matrix of shape `(4800, 64)`.
*   **`wavelet.npy`**: Haar wavelet mean/std sub-band stats of shape `(4800, 20)`.
*   **`color.npy`**: Normalized RGB color histograms of shape `(4800, 96)`.
*   **`canny.npy`**: Edge density and contour stats of shape `(4800, 6)`.
*   **`labels.npy`**: Int8 array of shape `(4800,)` containing the binary target labels (`0` or `1`).
*   **`generators.npy`**: String array of shape `(4800,)` containing the generator name.
*   **`splits.npy`**: String array of shape `(4800,)` containing split membership (`train`, `val`, or `test`).

---

### 📂 `processed/features/v1/{train, val, test}/`
Each directory contains the exact same set of `.npy` arrays as listed above, but filtered to its specific split using index masks:
*   **`train/`**: Sub-arrays of size $N_{train} = 3,360$ rows.
*   **`val/`**: Sub-arrays of size $N_{val} = 720$ rows.
*   **`test/`**: Sub-arrays of size $N_{test} = 720$ rows.

---

### 📂 `processed/features/v1/metadata/`
Contains configurations, dimensions, diagnostics, and human-readable run files.
*   **`metadata.csv`**: Map of row order, containing columns `filename`, `generator`, `label_binary`, `split` for the arrays.
*   **`feature_metadata.json`**: Records the exact **start** and **end** indices for each feature family inside the `combined.npy` matrix. Used for slicing without re-extraction:
    ```json
    {
      "families": {
        "hog":     {"start": 0,    "end": 8099, "dims": 8100},
        "lbp":     {"start": 8100, "end": 8109, "dims": 10},
        "glcm":    {"start": 8110, "end": 8149, "dims": 40},
        "dct":     {"start": 8150, "end": 8213, "dims": 64},
        "wavelet": {"start": 8214, "end": 8233, "dims": 20},
        "color":   {"start": 8234, "end": 8329, "dims": 96},
        "canny":   {"start": 8330, "end": 8335, "dims": 6}
      },
      "total_dims": 8336
    }
    ```
*   **`feature_names.txt`**: One name per dimension (e.g., `glcm_contrast_d1_a0`) matching columns of `combined.npy`. Essential for **SHAP** and **feature importance**.
*   **`feature_stats.csv`**: Contains `mean`, `std`, `min`, `max`, `var` for all $D$ columns to check for near-zero variance.
*   **`preprocessing.json`**: Copy of the extraction configurations (image size, wavelet name, HOG parameters).
*   **`timing_report.json`**: Average extraction speed per family (in seconds per image).
*   **`manifest.json`**: High-level execution summary (total processed, failed, splits, execution date).
*   **`failed_images.txt`**: List of corrupted/missing images skipped during the extraction run.
*   **`feature_report.txt`**: Human-readable text summary of the run.

---

## 3. How Downstream Teammates Can Load & Use Features

Teammates do not need to load raw images. They can perform all machine learning tasks using standard Python.

### Load Train Data and Target Labels
```python
import numpy as np

# Load full combined features
X_train = np.load("/content/drive/MyDrive/ml_project/processed/features/v1/train/combined.npy")
y_train = np.load("/content/drive/MyDrive/ml_project/processed/features/v1/train/labels.npy")

print(X_train.shape)  # Output: (3360, 8336)
print(y_train.shape)  # Output: (3360,)
```

### Ablation Study: Train only on Texture (LBP + GLCM)
```python
import json
import numpy as np

# Load boundaries mapping
with open("/content/drive/MyDrive/ml_project/processed/features/v1/metadata/feature_metadata.json") as f:
    meta = json.load(f)

# Find slicing indices
lbp_meta = meta["families"]["lbp"]
glcm_meta = meta["families"]["glcm"]

# Slice combined array
X_combined = np.load("/content/drive/MyDrive/ml_project/processed/features/v1/train/combined.npy")
X_lbp = X_combined[:, lbp_meta["start"]:lbp_meta["end"]+1]
X_glcm = X_combined[:, glcm_meta["start"]:glcm_meta["end"]+1]

X_texture = np.concatenate([X_lbp, X_glcm], axis=1)
print(X_texture.shape)  # Output: (3360, 50)
```

### Cross-Generator Attribution (Multi-Class classification)
```python
# To attribute which generator created a fake image
y_generators = np.load("/content/drive/MyDrive/ml_project/processed/features/v1/train/generators.npy")
# Train classifier on X_train to predict y_generators
```
