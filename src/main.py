import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from representasi import (
    image_info, pixel_matrix, save_pixel_matrix,
    split_rgb, rgb_to_grayscale, rgb_to_cmy, get_pixel_at,
)
from digitalisasi import (
    format_info, sampling, quantization,
    neighbourhood_analysis, draw_neighbourhood_figure,
)
from geometri import rotate, flip, crop, scale, negate
from deteksi_tepi import (
    sobel, sobel_detail, prewitt, robert_cross,
    laplacian, laplacian_detail,
)
from segmentasi import segment_plastic_waste
from volume_calc import estimate_volume_from_mask
from visualisasi import (
    plot_rgb_channels, plot_grayscale_cmy, plot_sampling,
    plot_quantization, plot_geometric_operations,
    plot_edge_detection, plot_segmentation_results,
    plot_overlay_comparison, print_report,
)


def _ensure_dir(path: str):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _imwrite(path: str, img: np.ndarray):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if len(img.shape) == 2:
        cv2.imwrite(path, img)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(
        description="Aplikasi Pengolahan Citra Digital — Segmentasi & Volume Sampah Plastik Sungai"
    )
    parser.add_argument("-i", "--image", required=True, help="Path ke citra input")
    parser.add_argument("-o", "--output", default="output", help="Folder output (default: output)")
    parser.add_argument("--kernel", type=int, default=5, help="Ukuran kernel morfologi (default: 5)")
    parser.add_argument("--min-area", type=int, default=500, help="Area minimum objek dalam piksel (default: 500)")
    parser.add_argument("--px-ratio", type=float, default=0.05, help="Rasio cm per piksel (default: 0.05)")
    parser.add_argument("--thickness", type=float, default=0.3, help="Tebal rata-rata plastik dalam cm (default: 0.3)")
    parser.add_argument("--no-display", action="store_true", help="Jangan tampilkan plot interaktif")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        print(f"[ERROR] File tidak ditemukan: {args.image}", file=sys.stderr)
        sys.exit(1)

    out_dir = _ensure_dir(args.output)
    print("[INFO] Membaca citra...")
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[ERROR] Gagal membaca citra: {args.image}", file=sys.stderr)
        sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ============================================================
    # 1. REPRESENTASI CITRA
    # ============================================================
    print("[INFO] 1. Representasi Citra...")
    rep_dir = _ensure_dir(out_dir / "01_representasi")

    info = image_info(img_rgb)
    channels = split_rgb(img_rgb)
    gray_img = rgb_to_grayscale(img_rgb)
    cmy = rgb_to_cmy(img_rgb)
    channels["original"] = img_rgb

    # Pixel matrix section
    px = pixel_matrix(gray_img, 0, 0, 10)
    save_pixel_matrix(px, str(rep_dir / "pixel_matrix.txt"))

    # Center pixel info
    cy, cx = gray_img.shape[0] // 2, gray_img.shape[1] // 2
    pixel_info = get_pixel_at(img_rgb, cx, cy)

    # Save RGB channel images
    _imwrite(str(rep_dir / "original.png"), img_rgb)
    _imwrite(str(rep_dir / "red_channel.png"), channels["red"])
    _imwrite(str(rep_dir / "green_channel.png"), channels["green"])
    _imwrite(str(rep_dir / "blue_channel.png"), channels["blue"])
    _imwrite(str(rep_dir / "grayscale.png"), gray_img)
    _imwrite(str(rep_dir / "cmy_cyan.png"), cmy["cyan_rgb"])
    _imwrite(str(rep_dir / "cmy_magenta.png"), cmy["magenta_rgb"])
    _imwrite(str(rep_dir / "cmy_yellow.png"), cmy["yellow_rgb"])

    # Visualization figures
    plot_rgb_channels(channels, str(rep_dir / "rgb_channels.png"))
    plot_grayscale_cmy(gray_img, cmy, str(rep_dir / "grayscale_cmy.png"))

    # ============================================================
    # 2. DIGITALISASI CITRA
    # ============================================================
    print("[INFO] 2. Digitalisasi Citra...")
    dig_dir = _ensure_dir(out_dir / "02_digitalisasi")

    fmt_info = format_info(str(img_path))

    # Sampling
    sampling_results = sampling(img_rgb)
    for key, data in sampling_results.items():
        _imwrite(str(dig_dir / f"sampling_{key}.png"), data["image"])
    plot_sampling(sampling_results, str(dig_dir / "sampling_grid.png"))

    # Quantization
    quant_results = quantization(gray_img)
    for key, data in quant_results.items():
        _imwrite(str(dig_dir / f"quantization_{key}.png"), data["image"])
    plot_quantization(quant_results, str(dig_dir / "quantization_grid.png"))

    # Neighbourhood analysis
    neighbour_info = neighbourhood_analysis(gray_img, cx, cy)
    neigh_fig = draw_neighbourhood_figure(gray_img, cx, cy)
    _imwrite(str(dig_dir / "neighbourhood.png"), neigh_fig)

    # ============================================================
    # 3. OPERASI GEOMETRI CITRA
    # ============================================================
    print("[INFO] 3. Operasi Geometri Citra...")
    geo_dir = _ensure_dir(out_dir / "03_geometri")

    ops = {}
    ops["rotate"] = rotate(img_rgb, 45)
    ops["flip_h"] = flip(img_rgb, "horizontal")
    ops["flip_v"] = flip(img_rgb, "vertical")
    ops["crop"] = crop(img_rgb)
    ops["scale"] = scale(img_rgb, 1.5)
    ops["negate"] = negate(img_rgb)

    for name, img_op in ops.items():
        _imwrite(str(geo_dir / f"{name}.png"), img_op)

    plot_geometric_operations(img_rgb, ops, str(geo_dir / "geometric_grid.png"))

    geo_op_names = [
        "Rotasi 45 derajat", "Flipping Horizontal", "Flipping Vertikal",
        "Cropping (1/4 tengah)", "Scaling 1.5x", "Negasi (Invers Warna)",
    ]

    # ============================================================
    # 4. DETEKSI TEPI
    # ============================================================
    print("[INFO] 4. Deteksi Tepi...")
    edge_dir = _ensure_dir(out_dir / "04_edge_detection")

    edges = {}
    edges["sobel"] = sobel(gray_img)
    edges["prewitt"] = prewitt(gray_img)
    edges["robert"] = robert_cross(gray_img)
    edges["laplacian"] = laplacian(gray_img)

    # Sobel detail (grad_x, grad_y)
    sobel_d = sobel_detail(gray_img)
    _imwrite(str(edge_dir / "sobel_grad_x.png"), sobel_d["grad_x"])
    _imwrite(str(edge_dir / "sobel_grad_y.png"), sobel_d["grad_y"])

    # Laplacian detail
    lap_d = laplacian_detail(gray_img)
    _imwrite(str(edge_dir / "laplacian_detail.png"), lap_d["magnitude"])

    for name, img_e in edges.items():
        _imwrite(str(edge_dir / f"{name}.png"), img_e)

    plot_edge_detection(edges, str(edge_dir / "edge_detection_grid.png"))

    edge_names = ["Sobel", "Prewitt", "Robert Cross", "Laplacian of Gaussian"]

    # ============================================================
    # 5. SEGMENTASI CITRA (Otsu + Morfologi)
    # ============================================================
    print("[INFO] 5. Segmentasi Citra (Otsu + Morfologi)...")
    seg_dir = _ensure_dir(out_dir / "05_segmentasi")

    result = segment_plastic_waste(
        img_rgb,
        kernel_size=args.kernel,
        min_area=args.min_area,
    )

    # Volume estimation
    volume = estimate_volume_from_mask(
        result["cleaned_mask"],
        pixel_to_cm_ratio=args.px_ratio,
        avg_thickness_cm=args.thickness,
    )

    # Save segmentation images
    _imwrite(str(seg_dir / "01_segmentation_grid.png"),
             plot_segmentation_results(result))
    _imwrite(str(seg_dir / "02_overlay.png"),
             plot_overlay_comparison(result))
    _imwrite(str(seg_dir / "03_cleaned_mask.png"), result["cleaned_mask"])
    _imwrite(str(seg_dir / "04_overlay_image.png"), result["overlay"])

    # ============================================================
    # PRINT LAPORAN
    # ============================================================
    print_report(
        info, sampling_results, quant_results,
        geo_op_names, edge_names, result, volume,
    )

    # ============================================================
    # SAVE LAPORAN TO FILE
    # ============================================================
    report_path = out_dir / "laporan.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        import io
        old_stdout = sys.stdout
        sys.stdout = f
        print_report(
            info, sampling_results, quant_results,
            geo_op_names, edge_names, result, volume,
        )
        sys.stdout = old_stdout

    print(f"[INFO] Semua output tersimpan di: {out_dir.resolve()}")
    print(f"[INFO] Laporan tersimpan di: {report_path}")


if __name__ == "__main__":
    main()
