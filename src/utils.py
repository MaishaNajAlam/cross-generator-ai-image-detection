"""
utils.py — Image loading and preprocessing utilities.

All functions are:
    - Deterministic: no randomness, no augmentation
    - Stateless: no side effects, no global state
    - Pathlib-native: accept Path objects or strings

These are the only functions that touch raw image files.
Everything downstream operates on numpy arrays.
"""

from pathlib import Path

import cv2
import numpy as np

from config import IMAGE_SIZE


# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    """
    Load an image from disk and return a standardised RGB uint8 array.

    Pipeline:
        disk → cv2 (BGR) → RGB → resize to IMAGE_SIZE

    Parameters
    ----------
    path : Path or str
        Absolute path to the image file.

    Returns
    -------
    np.ndarray
        uint8 array of shape (H, W, 3) in RGB colour space.
        H and W match IMAGE_SIZE from config.

    Raises
    ------
    FileNotFoundError
        If the file does not exist on disk.
    RuntimeError
        If cv2 cannot decode the image (corrupt file, unsupported format).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2 could not decode image: {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # cv2.resize expects (width, height) → swap IMAGE_SIZE (H, W)
    resized = cv2.resize(rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA)
    return resized  # dtype=uint8, shape=(H, W, 3)


def to_gray(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Convert an RGB uint8 image to a grayscale float32 image in [0, 1].

    Uses OpenCV's standard ITU-R 601 weighted conversion:
        gray = 0.299·R + 0.587·G + 0.114·B

    Parameters
    ----------
    rgb_uint8 : np.ndarray
        uint8 array of shape (H, W, 3) in RGB order.

    Returns
    -------
    np.ndarray
        float32 array of shape (H, W), values in [0, 1].
    """
    gray_u8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    return (gray_u8 / 255.0).astype(np.float32)


def to_gray_uint8(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Convert an RGB uint8 image to a grayscale uint8 image.

    Needed by extractors that require uint8 input (GLCM, Canny).

    Parameters
    ----------
    rgb_uint8 : np.ndarray
        uint8 array of shape (H, W, 3) in RGB order.

    Returns
    -------
    np.ndarray
        uint8 array of shape (H, W), values in [0, 255].
    """
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)


def to_float(img_uint8: np.ndarray) -> np.ndarray:
    """
    Convert a uint8 image to float32 in [0, 1]. No other changes.

    Parameters
    ----------
    img_uint8 : np.ndarray
        uint8 array of any shape.

    Returns
    -------
    np.ndarray
        float32 array, same shape, values in [0, 1].
    """
    return (img_uint8 / 255.0).astype(np.float32)


def prepare_image(path: Path):
    """
    Convenience wrapper: load image and return all required representations.

    Returns
    -------
    tuple : (rgb_uint8, gray_uint8, gray_float32, rgb_float32)
        rgb_uint8   — (H, W, 3) uint8   — used for color histogram
        gray_uint8  — (H, W)    uint8   — used for GLCM, Canny
        gray_f32    — (H, W)    float32 — used for HOG, LBP, DCT, Wavelet
        rgb_f32     — (H, W, 3) float32 — used for color histogram (float version)
    """
    rgb_u8   = load_image(path)
    gray_u8  = to_gray_uint8(rgb_u8)
    gray_f32 = (gray_u8 / 255.0).astype(np.float32)
    rgb_f32  = to_float(rgb_u8)
    return rgb_u8, gray_u8, gray_f32, rgb_f32
