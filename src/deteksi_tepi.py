import cv2
import numpy as np


def _gradient_magnitude(grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
    magnitude = np.sqrt(grad_x.astype(np.float64) ** 2 + grad_y.astype(np.float64) ** 2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def sobel(gray_img: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    return _gradient_magnitude(grad_x, grad_y)


def sobel_detail(gray_img: np.ndarray) -> dict:
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = _gradient_magnitude(grad_x, grad_y)
    return {
        "magnitude": magnitude,
        "grad_x": np.clip(np.abs(grad_x), 0, 255).astype(np.uint8),
        "grad_y": np.clip(np.abs(grad_y), 0, 255).astype(np.uint8),
    }


def prewitt(gray_img: np.ndarray) -> np.ndarray:
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    grad_x = cv2.filter2D(gray_img.astype(np.float64), -1, kernel_x)
    grad_y = cv2.filter2D(gray_img.astype(np.float64), -1, kernel_y)
    return _gradient_magnitude(grad_x, grad_y)


def robert_cross(gray_img: np.ndarray) -> np.ndarray:
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    grad_x = cv2.filter2D(gray_img.astype(np.float64), -1, kernel_x)
    grad_y = cv2.filter2D(gray_img.astype(np.float64), -1, kernel_y)
    return _gradient_magnitude(grad_x, grad_y)


def laplacian(gray_img: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=3)
    return np.clip(np.abs(lap), 0, 255).astype(np.uint8)


def laplacian_detail(gray_img: np.ndarray) -> dict:
    lap = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=3)
    lap_abs = np.abs(lap)
    return {
        "magnitude": np.clip(lap_abs, 0, 255).astype(np.uint8),
        "raw": lap,
    }
