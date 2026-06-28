import cv2
import numpy as np


def image_info(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    n_channels = image.shape[2] if len(image.shape) == 3 else 1
    info = {
        "dimensi": f"{w} x {h} piksel",
        "tinggi_px": h,
        "lebar_px": w,
        "channel": n_channels,
        "tipe_data": str(image.dtype),
    }
    if n_channels == 3:
        b, g, r = cv2.split(image)
        info.update({
            "red_min": int(r.min()), "red_max": int(r.max()), "red_mean": round(float(r.mean()), 2),
            "green_min": int(g.min()), "green_max": int(g.max()), "green_mean": round(float(g.mean()), 2),
            "blue_min": int(b.min()), "blue_max": int(b.max()), "blue_mean": round(float(b.mean()), 2),
        })
    return info


def pixel_matrix(image: np.ndarray, x: int = 0, y: int = 0, size: int = 8) -> np.ndarray:
    h, w = image.shape[:2]
    x_end = min(x + size, w)
    y_end = min(y + size, h)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return gray[y:y_end, x:x_end]


def save_pixel_matrix(matrix: np.ndarray, path: str):
    with open(path, "w") as f:
        f.write(f"Pixel Matrix ({matrix.shape[0]}x{matrix.shape[1]}):\n\n")
        for row in matrix:
            f.write("  " + " ".join(f"{p:3d}" for p in row) + "\n")


def split_rgb(image: np.ndarray) -> dict:
    r, g, b = cv2.split(image)
    zeros = np.zeros_like(r)
    return {
        "red": cv2.merge([zeros, zeros, r]),
        "green": cv2.merge([zeros, g, zeros]),
        "blue": cv2.merge([b, zeros, zeros]),
        "red_raw": r,
        "green_raw": g,
        "blue_raw": b,
    }


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def rgb_to_cmy(image: np.ndarray) -> dict:
    normalized = image.astype(np.float32) / 255.0
    cmy = 1.0 - normalized
    c, m, y = cv2.split(cmy)
    zeros = np.zeros_like(c, dtype=np.uint8)
    return {
        "cyan": (c * 255).astype(np.uint8),
        "magenta": (m * 255).astype(np.uint8),
        "yellow": (y * 255).astype(np.uint8),
        "cyan_rgb": cv2.merge([zeros, (c * 255).astype(np.uint8), (c * 255).astype(np.uint8)]),
        "magenta_rgb": cv2.merge([(m * 255).astype(np.uint8), zeros, (m * 255).astype(np.uint8)]),
        "yellow_rgb": cv2.merge([(y * 255).astype(np.uint8), (y * 255).astype(np.uint8), zeros]),
    }


def get_pixel_at(image: np.ndarray, x: int, y: int) -> dict:
    h, w = image.shape[:2]
    if x >= w or y >= h or x < 0 or y < 0:
        return {"error": "koordinat di luar citra"}
    pixel = image[y, x]
    if len(image.shape) == 3:
        return {
            "x": x, "y": y,
            "R": int(pixel[2]), "G": int(pixel[1]), "B": int(pixel[0]),
            "RGB": f"({int(pixel[2])}, {int(pixel[1])}, {int(pixel[0])})",
        }
    return {"x": x, "y": y, "gray": int(pixel)}
