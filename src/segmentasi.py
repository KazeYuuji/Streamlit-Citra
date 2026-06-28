import cv2
import numpy as np
from scipy import ndimage as ndi


def otsu_thresholding(gray_img: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply Otsu thresholding to separate foreground (plastic) from background.

    Returns:
        binary_mask: binary image where 255 = plastic, 0 = background
        threshold_value: the computed Otsu threshold
    """
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    threshold_value, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, threshold_value


def morphological_cleaning(
    binary_mask: np.ndarray,
    kernel_size: int = 5,
    close_iter: int = 2,
    open_iter: int = 1,
) -> np.ndarray:
    """Clean the binary mask using morphological operations.

    Steps:
        1. Closing  — fill small holes inside detected objects
        2. Opening  — remove small noise speckles
        3. Remove small connected-components below area_threshold
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=open_iter)

    return opened


def remove_small_objects(
    binary_mask: np.ndarray, min_area: int = 500
) -> np.ndarray:
    """Remove connected-components smaller than min_area pixels."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    """Fill holes inside foreground regions using flood-fill."""
    return ndi.binary_fill_holes(binary_mask.astype(bool)).astype(np.uint8) * 255


def segment_plastic_waste(
    image: np.ndarray,
    kernel_size: int = 5,
    close_iter: int = 2,
    open_iter: int = 1,
    min_area: int = 500,
) -> dict:
    """Full pipeline: Otsu → morphological clean → filter by area.

    Returns a dict with:
        - original      : input RGB image
        - gray          : grayscale version
        - raw_otsu      : binary mask before cleaning
        - cleaned_mask  : binary mask after morphological ops
        - overlay       : original image with plastic regions highlighted
        - threshold_val : Otsu threshold value
        - plastic_pixels: number of pixels classified as plastic
        - total_pixels  : total image pixels
        - area_percent  : percentage of image covered by plastic
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    raw_otsu, thresh_val = otsu_thresholding(gray)
    cleaned = morphological_cleaning(raw_otsu, kernel_size, close_iter, open_iter)
    cleaned = remove_small_objects(cleaned, min_area)
    cleaned = fill_holes(cleaned)

    overlay = image.copy()
    overlay[cleaned == 255] = (0, 0, 255)

    plastic_px = int(np.sum(cleaned == 255))
    total_px = cleaned.size
    return {
        "original": image,
        "gray": gray,
        "raw_otsu": raw_otsu,
        "cleaned_mask": cleaned,
        "overlay": overlay,
        "threshold_val": round(float(thresh_val), 2),
        "plastic_pixels": plastic_px,
        "total_pixels": total_px,
        "area_percent": round(plastic_px / total_px * 100, 3),
    }
