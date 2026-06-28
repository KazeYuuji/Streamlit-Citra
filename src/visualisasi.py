import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def _save_figure(fig, save_path=None, dpi=150):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    fig.canvas.draw()
    img_rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_rgb = img_rgb.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return img_rgb


def plot_rgb_channels(channels: dict, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Representasi Citra — Model Warna RGB", fontsize=14, fontweight="bold")

    axes[0, 0].imshow(channels["original"])
    axes[0, 0].set_title("Citra Asli (RGB)", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(channels["red"])
    axes[0, 1].set_title("Channel Red", fontsize=11)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(channels["green"])
    axes[1, 0].set_title("Channel Green", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(channels["blue"])
    axes[1, 1].set_title("Channel Blue", fontsize=11)
    axes[1, 1].axis("off")

    plt.tight_layout()
    return _save_figure(fig, save_path)


def plot_grayscale_cmy(grayscale: np.ndarray, cmy: dict, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Representasi Citra — Grayscale & CMY", fontsize=14, fontweight="bold")

    axes[0, 0].imshow(grayscale, cmap="gray")
    axes[0, 0].set_title("Grayscale", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cmy["cyan_rgb"])
    axes[0, 1].set_title("CMY — Cyan", fontsize=11)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(cmy["magenta_rgb"])
    axes[1, 0].set_title("CMY — Magenta", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cmy["yellow_rgb"])
    axes[1, 1].set_title("CMY — Yellow", fontsize=11)
    axes[1, 1].axis("off")

    plt.tight_layout()
    return _save_figure(fig, save_path)


def plot_sampling(sampling_results: dict, save_path=None):
    scales = sorted(sampling_results.keys(), key=lambda k: float(k), reverse=True)
    n = len(scales)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle("Digitalisasi — Sampling (Resolusi Berbeda)", fontsize=14, fontweight="bold")
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, scale_key in enumerate(scales):
        data = sampling_results[scale_key]
        axes[i].imshow(data["image"])
        axes[i].set_title(f"Skala {scale_key}\n{data['dimensi']}", fontsize=9)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return _save_figure(fig, save_path)


def plot_quantization(quant_results: dict, save_path=None):
    bits_order = sorted(quant_results.keys(), key=lambda k: int(k.split("_")[0]), reverse=True)
    n = len(bits_order)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle("Digitalisasi — Kuantisasi (Level Warna Berbeda)", fontsize=14, fontweight="bold")
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, key in enumerate(bits_order):
        data = quant_results[key]
        axes[i].imshow(data["image"], cmap="gray")
        axes[i].set_title(f"{data['bits']}-bit ({data['levels']} level)", fontsize=10)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return _save_figure(fig, save_path)


def plot_geometric_operations(original: np.ndarray, ops: dict, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Operasi Geometri Citra", fontsize=14, fontweight="bold")

    items = [
        (0, 0, "Citra Asli", original),
        (0, 1, "Rotasi 45\u00b0", ops.get("rotate")),
        (0, 2, "Flipping Horizontal", ops.get("flip_h")),
        (1, 0, "Cropping", ops.get("crop")),
        (1, 1, "Scaling 2x", ops.get("scale")),
        (1, 2, "Negasi (Invers Warna)", ops.get("negate")),
    ]

    for row, col, title, img in items:
        axes[row, col].imshow(img)
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis("off")

    plt.tight_layout()
    return _save_figure(fig, save_path)


def plot_edge_detection(edges: dict, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Deteksi Tepi", fontsize=14, fontweight="bold")

    items = [
        (0, 0, "Sobel", edges.get("sobel")),
        (0, 1, "Prewitt", edges.get("prewitt")),
        (1, 0, "Robert Cross", edges.get("robert")),
        (1, 1, "Laplacian", edges.get("laplacian")),
    ]

    for row, col, title, img in items:
        axes[row, col].imshow(img, cmap="gray")
        axes[row, col].set_title(title, fontsize=11)
        axes[row, col].axis("off")

    plt.tight_layout()
    return _save_figure(fig, save_path)


def plot_segmentation_results(result: dict, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Segmentasi Sampah Plastik Makro — Otsu + Morfologi", fontsize=14, fontweight="bold")

    axes[0, 0].imshow(result["original"])
    axes[0, 0].set_title("Citra Asli", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(result["gray"], cmap="gray")
    axes[0, 1].set_title("Grayscale", fontsize=11)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(result["raw_otsu"], cmap="gray")
    axes[1, 0].set_title(f"Otsu Threshold (T={result['threshold_val']})", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(result["cleaned_mask"], cmap="gray")
    axes[1, 1].set_title("Setelah Morfologi + Filter Luas", fontsize=11)
    axes[1, 1].axis("off")

    plt.tight_layout()
    return _save_figure(fig, save_path)


def plot_overlay_comparison(result: dict, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Perbandingan Citra Asli vs Deteksi Plastik", fontsize=13, fontweight="bold")

    axes[0].imshow(result["original"])
    axes[0].set_title("Citra Asli", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(result["overlay"])
    axes[1].set_title("Overlay — Area Terdeteksi (Merah)", fontsize=11)
    axes[1].axis("off")

    plt.tight_layout()
    return _save_figure(fig, save_path)


def print_report(info: dict, sampling_data: dict, quant_data: dict,
                 geo_ops: list, edge_names: list, result: dict, volume: dict):
    sep = "=" * 65
    print(f"\n{sep}")
    print("   LAPORAN UAS PENGOLAHAN CITRA DIGITAL")
    print(f"   Topik: Segmentasi & Perhitungan Volume Sampah Plastik Makro pada Citra Sungai")
    print(f"{sep}")

    # 1. Representasi Citra
    print(f"\n{'-' * 65}")
    print("   1. REPRESENTASI CITRA")
    print(f"{'-' * 65}")
    print(f"   Dimensi citra          : {info['dimensi']}")
    print(f"   Model warna            : RGB ({info['channel']} channel)")
    print(f"   Tipe data              : {info['tipe_data']}")
    if 'red_mean' in info:
        print(f"   Channel Red            : min={info['red_min']}, max={info['red_max']}, mean={info['red_mean']}")
        print(f"   Channel Green          : min={info['green_min']}, max={info['green_max']}, mean={info['green_mean']}")
        print(f"   Channel Blue           : min={info['blue_min']}, max={info['blue_max']}, mean={info['blue_mean']}")

    # 2. Digitalisasi Citra
    print(f"\n{'-' * 65}")
    print("   2. DIGITALISASI CITRA")
    print(f"{'-' * 65}")
    print(f"   Sampling (resolusi):")
    for key, data in sorted(sampling_data.items(), key=lambda x: float(x[0]), reverse=True):
        print(f"     - Skala {key}: {data['dimensi']} piksel")
    print(f"   Kuantisasi (level warna):")
    for key, data in sorted(quant_data.items(), key=lambda x: int(x[0].split('_')[0]), reverse=True):
        print(f"     - {data['bits']}-bit: {data['levels']} level warna")

    # 3. Operasi Geometri
    print(f"\n{'-' * 65}")
    print("   3. OPERASI ARITMATIKA & GEOMETRI CITRA")
    print(f"{'-' * 65}")
    for op in geo_ops:
        print(f"   - {op}")

    # 4. Deteksi Tepi
    print(f"\n{'-' * 65}")
    print("   4. DETEKSI TEPI")
    print(f"{'-' * 65}")
    for name in edge_names:
        print(f"   - {name}")

    # 5. Segmentasi
    print(f"\n{'-' * 65}")
    print("   5. SEGMENTASI CITRA")
    print(f"{'-' * 65}")
    print(f"   Metode                 : Otsu Thresholding + Operasi Morfologi")
    print(f"   Threshold Otsu         : {result['threshold_val']}")
    print(f"   Piksel Plastik         : {result['plastic_pixels']:,}")
    print(f"   Total Piksel           : {result['total_pixels']:,}")
    print(f"   Luas Area (%-cover)    : {result['area_percent']} %")

    print(f"\n{'-' * 65}")
    print("   ESTIMASI VOLUME & MASSA")
    print(f"{'-' * 65}")
    print(f"   Rasio px->cm           : {volume['pixel_to_cm_ratio']} cm/px")
    print(f"   Tebal rata-rata        : {volume['avg_thickness_cm']} cm")
    print(f"   Luas                   : {volume['area_cm2']} cm2  |  {volume['area_m2']} m2")
    print(f"   Volume                 : {volume['volume_cm3']} cm3  |  {volume['volume_m3']} m3  |  {volume['volume_liters']} liter")
    print(f"   Estimasi massa         : {volume['estimated_mass_kg']} kg")
    print(f"{sep}\n")
