import sys
from pathlib import Path
import io

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from representasi import (
    image_info, pixel_matrix, split_rgb,
    rgb_to_grayscale, rgb_to_cmy, get_pixel_at,
)
from digitalisasi import (
    sampling, quantization, neighbourhood_analysis,
    draw_neighbourhood_figure,
)
from geometri import rotate, flip, crop, scale, negate
from deteksi_tepi import (
    sobel, prewitt, robert_cross, laplacian,
)
from segmentasi import segment_plastic_waste
from volume_calc import estimate_volume_from_mask
from visualisasi import (
    plot_rgb_channels, plot_grayscale_cmy, plot_sampling,
    plot_quantization, plot_geometric_operations,
    plot_edge_detection, plot_segmentation_results,
    plot_overlay_comparison,
)


st.set_page_config(
    page_title="Pengolahan Citra Digital - Segmentasi Plastik Sungai",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .stApp { background-color: #f8f9fa; }
    .block-container { max-width: 1200px; }
    .report-title { font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem; }
    .metric-card { background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)


def process_image(img_rgb, kernel_size, min_area):
    gray = rgb_to_grayscale(img_rgb)

    info = image_info(img_rgb)
    channels = split_rgb(img_rgb)
    channels["original"] = img_rgb
    cmy = rgb_to_cmy(img_rgb)
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    pixel_info = get_pixel_at(img_rgb, cx, cy)

    sampling_results = sampling(img_rgb)
    quant_results = quantization(gray)
    neighbour_info = neighbourhood_analysis(gray, cx, cy)
    neigh_fig = draw_neighbourhood_figure(gray, cx, cy)

    geo_ops = {
        "rotate": rotate(img_rgb, 45),
        "flip_h": flip(img_rgb, "horizontal"),
        "flip_v": flip(img_rgb, "vertical"),
        "crop": crop(img_rgb),
        "scale": scale(img_rgb, 1.5),
        "negate": negate(img_rgb),
    }

    edges = {
        "sobel": sobel(gray),
        "prewitt": prewitt(gray),
        "robert": robert_cross(gray),
        "laplacian": laplacian(gray),
    }

    result = segment_plastic_waste(img_rgb, kernel_size=kernel_size, min_area=min_area)
    volume = estimate_volume_from_mask(result["cleaned_mask"])

    return {
        "img_rgb": img_rgb,
        "gray": gray,
        "info": info,
        "channels": channels,
        "cmy": cmy,
        "pixel_info": pixel_info,
        "sampling": sampling_results,
        "quantization": quant_results,
        "neighbour_info": neighbour_info,
        "neigh_fig": neigh_fig,
        "geo_ops": geo_ops,
        "edges": edges,
        "result": result,
        "volume": volume,
    }


st.title("🌊 PCD - Segmentasi Sampah Plastik pada Citra Sungai")
st.markdown("Aplikasi pengolahan citra digital untuk mendeteksi dan menghitung volume sampah plastik makro pada citra sungai menggunakan **Otsu Thresholding** dan **Operasi Morfologi**.")

with st.sidebar:
    st.header("Input Citra")
    uploaded_file = st.file_uploader("Upload gambar sungai", type=["jpg", "jpeg", "png", "bmp"])

    use_default = st.checkbox("Gunakan contoh bawaan", value=True if not uploaded_file else False)

    st.divider()
    st.header("Parameter")
    kernel_size = st.slider("Ukuran kernel morfologi", 3, 15, 5, 2)
    min_area = st.number_input("Area minimum objek (piksel)", 100, 5000, 500, 100)
    px_ratio = st.number_input("Rasio cm/piksel", 0.01, 0.5, 0.05, 0.01)
    thickness = st.number_input("Tebal plastik (cm)", 0.1, 2.0, 0.3, 0.1)

    run_btn = st.button("Proses Citra", type="primary", use_container_width=True)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Gagal membaca file. Gunakan format JPG atau PNG.")
        st.stop()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
elif use_default:
    default_path = Path(__file__).parent / "data" / "river_plastic_01.jpg"
    if not default_path.exists():
        st.error(f"File contoh tidak ditemukan: {default_path}")
        st.stop()
    img_bgr = cv2.imread(str(default_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
else:
    img_rgb = None

if img_rgb is not None and run_btn:
    data = process_image(img_rgb, kernel_size, min_area)

    with st.container():
        st.subheader("Citra Asli")
        st.image(data["img_rgb"], use_container_width=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Representasi Citra", "2. Digitalisasi Citra",
        "3. Operasi Geometri", "4. Deteksi Tepi",
        "5. Segmentasi & Volume",
    ])

    # ==================== TAB 1 ====================
    with tab1:
        st.subheader("Representasi Citra")
        st.write("Citra sebagai matriks piksel, model warna RGB/CMY, dan konversi grayscale.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dimensi", data["info"]["dimensi"])
        with col2:
            st.metric("Channel", f'{data["info"]["channel"]} (RGB)')
        with col3:
            st.metric("Tipe Data", data["info"]["tipe_data"])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("R mean", data["info"].get("red_mean", "-"))
        with c2:
            st.metric("G mean", data["info"].get("green_mean", "-"))
        with c3:
            st.metric("B mean", data["info"].get("blue_mean", "-"))

        st.markdown("**Channel RGB**")
        st.image(plot_rgb_channels(data["channels"]), use_container_width=True)

        st.markdown("**Grayscale & CMY**")
        st.image(plot_grayscale_cmy(data["gray"], data["cmy"]), use_container_width=True)

        with st.expander("Lihat matriks piksel (10x10)"):
            px = pixel_matrix(data["gray"], 0, 0, 10)
            st.code("\n".join("  ".join(f"{v:3d}" for v in row) for row in px))

        with st.expander("Lihat informasi piksel pusat"):
            pi = data["pixel_info"]
            if "error" not in pi:
                st.json({
                    "x": pi["x"], "y": pi["y"],
                    "R": pi["R"], "G": pi["G"], "B": pi["B"],
                })
            else:
                st.write(pi["error"])

    # ==================== TAB 2 ====================
    with tab2:
        st.subheader("Digitalisasi Citra")
        st.write("Sampling (resolusi), kuantisasi (level warna), dan hubungan antar piksel.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Sampling — Resolusi Berbeda**")
            st.image(plot_sampling(data["sampling"]), use_container_width=True)
        with c2:
            st.markdown("**Kuantisasi — Level Warna Berbeda**")
            st.image(plot_quantization(data["quantization"]), use_container_width=True)

        with st.expander("Detail sampling"):
            for key, d in sorted(data["sampling"].items(), key=lambda x: float(x[0]), reverse=True):
                st.write(f"- Skala {key}: {d['dimensi']} piksel")

        with st.expander("Detail kuantisasi"):
            for key, d in sorted(data["quantization"].items(), key=lambda x: int(x[0].split("_")[0]), reverse=True):
                st.write(f"- {d['bits']}-bit: {d['levels']} level warna")

        st.markdown("**Hubungan Antar Piksel (4 & 8 Neighbourhood)**")
        st.image(data["neigh_fig"], use_container_width=True)

        with st.expander("Lihat data neighbour"):
            ni = data["neighbour_info"]
            st.write("**4-Neighbour:**")
            for arah, info in ni["4_neighbour"].items():
                st.write(f"  {arah}: {info}")
            st.write("**8-Neighbour:**")
            for arah, info in ni["8_neighbour"].items():
                st.write(f"  {arah}: {info}")

    # ==================== TAB 3 ====================
    with tab3:
        st.subheader("Operasi Geometri Citra")
        st.write("Rotasi, flipping, cropping, scaling, dan negasi.")

        st.image(plot_geometric_operations(data["img_rgb"], data["geo_ops"]), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(data["geo_ops"]["rotate"], caption="Rotasi 45°", use_container_width=True)
        with col2:
            st.image(data["geo_ops"]["flip_h"], caption="Flipping Horizontal", use_container_width=True)
        with col3:
            st.image(data["geo_ops"]["flip_v"], caption="Flipping Vertikal", use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(data["geo_ops"]["crop"], caption="Cropping", use_container_width=True)
        with col2:
            st.image(data["geo_ops"]["scale"], caption="Scaling 1.5x", use_container_width=True)
        with col3:
            st.image(data["geo_ops"]["negate"], caption="Negasi (Invers)", use_container_width=True)

    # ==================== TAB 4 ====================
    with tab4:
        st.subheader("Deteksi Tepi")
        st.write("Empat metode deteksi tepi: Sobel, Prewitt, Robert Cross, dan Laplacian.")

        st.image(plot_edge_detection(data["edges"]), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.image(data["edges"]["sobel"], caption="Sobel", use_container_width=True)
        with col2:
            st.image(data["edges"]["prewitt"], caption="Prewitt", use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.image(data["edges"]["robert"], caption="Robert Cross", use_container_width=True)
        with col2:
            st.image(data["edges"]["laplacian"], caption="Laplacian", use_container_width=True)

    # ==================== TAB 5 ====================
    with tab5:
        st.subheader("Segmentasi Citra — Otsu + Morfologi")
        st.write("Deteksi sampah plastik menggunakan Otsu thresholding dan operasi morfologi.")

        st.image(plot_segmentation_results(data["result"]), use_container_width=True)
        st.image(plot_overlay_comparison(data["result"]), use_container_width=True)

        vol = data["volume"]
        res = data["result"]

        st.markdown("### Hasil Estimasi Volume & Massa")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Area Plastik", f"{res['area_percent']}%")
        with m2:
            st.metric("Threshold Otsu", str(res["threshold_val"]))
        with m3:
            st.metric("Piksel Plastik", f"{res['plastic_pixels']:,}")
        with m4:
            st.metric("Total Piksel", f"{res['total_pixels']:,}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Luas", f"{vol['area_m2']} m²")
        with c2:
            st.metric("Volume", f"{vol['volume_liters']} liter")
        with c3:
            st.metric("Estimasi Massa", f"{vol['estimated_mass_kg']} kg")

        with st.expander("Detail estimasi"):
            st.json(vol)

elif img_rgb is not None:
    st.info("Upload gambar atau gunakan contoh bawaan, lalu klik **Proses Citra**.")
else:
    st.info("Upload gambar sungai di sidebar, atau gunakan contoh bawaan.")

st.divider()
st.caption("UAS Pengolahan Citra Digital — Segmentasi & Perhitungan Volume Sampah Plastik Makro pada Citra Sungai")
