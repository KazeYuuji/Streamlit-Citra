import cv2
import numpy as np


def rotate(image: np.ndarray, angle: float = 45) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated


def flip(image: np.ndarray, direction: str = "horizontal") -> np.ndarray:
    if direction == "horizontal":
        return cv2.flip(image, 1)
    elif direction == "vertical":
        return cv2.flip(image, 0)
    elif direction == "both":
        return cv2.flip(image, -1)
    else:
        raise ValueError(f"arah tidak dikenal: {direction}")


def crop(image: np.ndarray, x: int = None, y: int = None, w: int = None, h: int = None) -> np.ndarray:
    img_h, img_w = image.shape[:2]
    if x is None:
        x = img_w // 4
    if y is None:
        y = img_h // 4
    if w is None:
        w = img_w // 2
    if h is None:
        h = img_h // 2
    x_end = min(x + w, img_w)
    y_end = min(y + h, img_h)
    return image[y:y_end, x:x_end].copy()


def scale(image: np.ndarray, factor: float = 2.0) -> np.ndarray:
    h, w = image.shape[:2]
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    interpolation = cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def negate(image: np.ndarray) -> np.ndarray:
    return 255 - image
