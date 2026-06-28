import cv2
import numpy as np
from pathlib import Path


def format_info(image_path: str) -> dict:
    path = Path(image_path)
    ext = path.suffix.lower()
    img = cv2.imread(str(path))
    if img is None:
        return {"error": "gagal membaca citra"}
    h, w = img.shape[:2]
    return {
        "nama_file": path.name,
        "format": ext,
        "dimensi": f"{w} x {h} piksel",
        "ukuran_disk_kb": round(path.stat().st_size / 1024, 2),
    }


def sampling(image: np.ndarray, scales: list = None) -> dict:
    if scales is None:
        scales = [1.0, 0.5, 0.25, 0.125]
    h, w = image.shape[:2]
    results = {}
    for scale in scales:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        results[str(scale)] = {"image": resized, "dimensi": f"{new_w} x {new_h}", "scale": scale}
    return results


def quantization(gray_img: np.ndarray, bit_depths: list = None) -> dict:
    if bit_depths is None:
        bit_depths = [8, 6, 4, 2, 1]
    results = {}
    for bits in bit_depths:
        levels = 2 ** bits
        quantized = np.floor(gray_img.astype(np.float32) / (256 / levels)) * (256 / levels)
        if bits == 1:
            quantized[quantized > 0] = 255
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        results[f"{bits}_bit"] = {"image": quantized, "levels": levels, "bits": bits}
    return results


def neighbourhood_analysis(gray_img: np.ndarray, x: int, y: int) -> dict:
    h, w = gray_img.shape
    result = {"center": (x, y), "center_value": int(gray_img[y, x])}

    neighbour_4 = {}
    directions_4 = [("atas", 0, -1), ("bawah", 0, 1), ("kiri", -1, 0), ("kanan", 1, 0)]
    for name, dx, dy in directions_4:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            neighbour_4[name] = {"koordinat": (nx, ny), "nilai": int(gray_img[ny, nx])}
        else:
            neighbour_4[name] = {"koordinat": (nx, ny), "nilai": "di luar citra"}
    result["4_neighbour"] = neighbour_4

    neighbour_8 = {}
    directions_8 = [
        ("atas-kiri", -1, -1), ("atas", 0, -1), ("atas-kanan", 1, -1),
        ("kiri", -1, 0), ("kanan", 1, 0),
        ("bawah-kiri", -1, 1), ("bawah", 0, 1), ("bawah-kanan", 1, 1),
    ]
    for name, dx, dy in directions_8:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            neighbour_8[name] = {"koordinat": (nx, ny), "nilai": int(gray_img[ny, nx])}
        else:
            neighbour_8[name] = {"koordinat": (nx, ny), "nilai": "di luar citra"}
    result["8_neighbour"] = neighbour_8

    return result


def draw_neighbourhood_figure(gray_img: np.ndarray, x: int, y: int) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Hubungan Antar Piksel — Pusat ({x}, {y})", fontsize=13, fontweight="bold")

    for ax, title, directions in zip(
        axes,
        ["4-Neighbourhood", "8-Neighbourhood"],
        [
            [(-1, 0), (1, 0), (0, -1), (0, 1)],
            [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
        ],
    ):
        h, w = gray_img.shape
        crop_size = 5
        x_start = max(0, x - crop_size)
        x_end = min(w, x + crop_size + 1)
        y_start = max(0, y - crop_size)
        y_end = min(h, y + crop_size + 1)
        patch = gray_img[y_start:y_end, x_start:x_end]

        ax.imshow(patch, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=11)

        cx, cy = x - x_start, y - y_start
        ax.plot(cx, cy, "ro", markersize=10, label=f"Pusat ({x},{y}) = {int(gray_img[y, x])}")

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                ax.plot(nx - x_start, ny - y_start, "bo", markersize=8)
                ax.annotate(str(int(gray_img[ny, nx])), (nx - x_start, ny - y_start),
                            textcoords="offset points", xytext=(3, 3), fontsize=7, color="blue")

        ax.legend(fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    fig.canvas.draw()
    img_rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_rgb = img_rgb.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return img_rgb
