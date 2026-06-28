import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import traceback

st.set_page_config(
    page_title="Segmentasi Sampah Plastik Sungai",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_image(uploaded_file):
    try:
        bytes_data = uploaded_file.getvalue()
        pil_img = Image.open(BytesIO(bytes_data))
        opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return pil_img, opencv_img
    except Exception as e:
        st.error(f"Gagal memuat gambar: {str(e)}")
        st.stop()

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def rgb_to_cmy(r, g, b):
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    c = 1 - r_norm
    m = 1 - g_norm
    y = 1 - b_norm
    return (c * 255).astype(np.uint8), (m * 255).astype(np.uint8), (y * 255).astype(np.uint8)

def apply_sobel(img_gray):
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_prewitt(img_gray):
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    prewitt_x = cv2.filter2D(img_gray, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(img_gray, cv2.CV_64F, kernel_y)
    magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_robert(img_gray):
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    robert_x = cv2.filter2D(img_gray, cv2.CV_64F, kernel_x)
    robert_y = cv2.filter2D(img_gray, cv2.CV_64F, kernel_y)
    magnitude = np.sqrt(robert_x**2 + robert_y**2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_laplacian(img_gray):
    lap = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)
    return cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def segment_otsu_morphology(img_gray, morph_op="closing", kernel_size=5, iterations=1):
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    morph_map = {
        "erosion":    cv2.erode(binary, kernel, iterations=iterations),
        "dilation":   cv2.dilate(binary, kernel, iterations=iterations),
        "opening":    cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations),
        "closing":    cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations),
        "gradient":   cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel, iterations=iterations),
    }
    return binary, morph_map.get(morph_op, binary)

def analyze_segmentation(binary_mask, pixel_to_cm=0.026458):
    total_pixels = binary_mask.size
    plastic_pixels = int(np.sum(binary_mask > 0))
    non_plastic = total_pixels - plastic_pixels
    coverage_pct = (plastic_pixels / total_pixels) * 100
    area_cm2 = plastic_pixels * (pixel_to_cm ** 2)
    est_volume_ml = area_cm2 * 0.5
    return {
        "total_pixels": total_pixels,
        "plastic_pixels": plastic_pixels,
        "non_plastic_pixels": non_plastic,
        "coverage_pct": coverage_pct,
        "area_cm2": area_cm2,
        "est_volume_ml": est_volume_ml,
    }

def colormap_overlay(original_bgr, binary_mask, color=(0, 0, 255), alpha=0.5):
    overlay = original_bgr.copy()
    colored_mask = np.zeros_like(original_bgr, dtype=np.uint8)
    colored_mask[binary_mask > 0] = color
    overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
    return overlay

def make_pixel_matrix_display(patch):
    patch_8bit = (patch / patch.max() * 255).astype(np.uint8) if patch.max() > 0 else patch
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(patch_8bit, cmap="gray" if len(patch_8bit.shape) == 2 else None)
    for i in range(patch_8bit.shape[0]):
        for j in range(patch_8bit.shape[1]):
            val = patch_8bit[i, j]
            txt = f"{val:.0f}" if len(patch_8bit.shape) == 2 else str(val[:3])
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="red")
    ax.set_title("Pixel Matrix (Top-Left 16x16)")
    ax.axis("off")
    return fig

def plot_rgb_histogram(img_rgb):
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ("red", "green", "blue")
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, alpha=0.8)
    ax.set_xlim([0, 256])
    ax.set_title("RGB Histogram")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    return fig

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    st.title("🌊 Segmentasi & Perhitungan Volume Sampah Plastik Makro pada Citra Sungai")
    st.markdown("""
    **Menggunakan Metode Otsu Thresholding dan Operasi Morfologi**
    """)
    st.markdown("---")

    # ---------- Sidebar: Upload & Info ----------
    with st.sidebar:
        st.header("📂 Upload Citra")
        uploaded_file = st.file_uploader(
            "Pilih gambar sungai (JPG, PNG, BMP)",
            type=["jpg", "jpeg", "png", "bmp"]
        )

        if uploaded_file is not None:
            pil_img, cv2_img_bgr = load_image(uploaded_file)
            cv2_img_rgb = cv2.cvtColor(cv2_img_bgr, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(cv2_img_bgr, cv2.COLOR_BGR2GRAY)

            st.success(f"✔ **{uploaded_file.name}**")
            h, w = cv2_img_bgr.shape[:2]
            st.info(f"**Dimensi:** {w} x {h} px\n**Format:** {uploaded_file.type}")
            st.image(cv2_img_rgb, caption="Citra Asli", width="stretch")

            # Save to session state
            st.session_state["pil_img"] = pil_img
            st.session_state["cv2_img_bgr"] = cv2_img_bgr
            st.session_state["cv2_img_rgb"] = cv2_img_rgb
            st.session_state["gray_img"] = gray_img
            st.session_state["img_loaded"] = True
        else:
            st.warning("Silakan unggah citra sungai untuk memulai.")
            if "img_loaded" in st.session_state:
                del st.session_state["img_loaded"]

    # ---------- Main content ----------
    if "img_loaded" not in st.session_state:
        st.info("👈 Unggah citra dari sidebar untuk memulai.")
        st.markdown("""
        ### Fitur Aplikasi
        | Modul | Deskripsi |
        |---|---|
        | **1. Representasi Citra** | Matriks pixel, model RGB/CMY, histogram, konversi grayscale |
        | **2. Digitalisasi Citra** | Sampling, kuantisasi, resolusi, tetangga pixel |
        | **3. Operasi Geometri** | Grayscale, rotasi, flipping, cropping, scaling, negasi |
        | **4. Deteksi Tepi** | Sobel, Prewitt, Robert Cross, Laplacian |
        | **5. Segmentasi & Volume** | Otsu thresholding + morfologi + estimasi volume sampah |
        """)
        return

    # Retrieve from session state
    pil_img = st.session_state["pil_img"]
    cv2_img_bgr = st.session_state["cv2_img_bgr"]
    cv2_img_rgb = st.session_state["cv2_img_rgb"]
    gray_img = st.session_state["gray_img"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Representasi Citra",
        "🔢 Digitalisasi Citra",
        "🔄 Operasi Geometri",
        "✏️ Deteksi Tepi",
        "♻️ Segmentasi & Volume"
    ])

    # ========================================================
    # TAB 1: REPRESENTASI CITRA
    # ========================================================
    with tab1:
        st.header("📊 Representasi Citra")
        st.markdown("*Citra sebagai matriks pixel, model warna RGB/CMY, konversi grayscale*")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Citra Asli (RGB)")
            st.image(cv2_img_rgb, caption=f"Dimensi: {cv2_img_rgb.shape[1]} x {cv2_img_rgb.shape[0]} px", width="stretch")

            # RGB Channel Visualization
            st.subheader("Kanal RGB")
            r_ch, g_ch, b_ch = cv2.split(cv2_img_rgb)
            fig_rgb, axes = plt.subplots(1, 3, figsize=(9, 3))
            titles = ["Red Channel", "Green Channel", "Blue Channel"]
            for ax, ch, title, cmap in zip(axes, [r_ch, g_ch, b_ch], titles, ["Reds", "Greens", "Blues"]):
                im = ax.imshow(ch, cmap=cmap, vmin=0, vmax=255)
                ax.set_title(title)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig_rgb)

            # CMY Model
            st.subheader("Model Warna CMY")
            c_ch, m_ch, y_ch = rgb_to_cmy(r_ch, g_ch, b_ch)
            fig_cmy, axes2 = plt.subplots(1, 3, figsize=(9, 3))
            for ax, ch, title in zip(axes2, [c_ch, m_ch, y_ch], ["Cyan", "Magenta", "Yellow"]):
                im = ax.imshow(ch, cmap="gray", vmin=0, vmax=255)
                ax.set_title(title)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig_cmy)

        with col2:
            st.subheader("Citra Grayscale")
            st.image(gray_img, caption="Konversi RGB → Grayscale", width="stretch", channels="GRAY")

            # Grayscale Formula Info
            with st.expander("ℹ️ Rumus Konversi Grayscale"):
                st.latex(r"Gray = 0.299 \times R + 0.587 \times G + 0.114 \times B")

            # Histogram
            st.subheader("Histogram Grayscale")
            fig_hist_g, ax_h = plt.subplots(figsize=(6, 3))
            ax_h.hist(gray_img.ravel(), bins=256, range=[0, 256], color="gray", alpha=0.8)
            ax_h.set_title("Grayscale Histogram")
            ax_h.set_xlabel("Intensity")
            ax_h.set_ylabel("Frequency")
            st.pyplot(fig_hist_g)

            # RGB Histogram
            st.subheader("Histogram RGB")
            st.pyplot(plot_rgb_histogram(cv2_img_rgb))

            # Pixel Matrix Display
            st.subheader("Matriks Pixel (Area 16×16)")
            patch = gray_img[:16, :16]
            st.pyplot(make_pixel_matrix_display(patch))

    # ========================================================
    # TAB 2: DIGITALISASI CITRA
    # ========================================================
    with tab2:
        st.header("🔢 Digitalisasi Citra")
        st.markdown("*Sampling, kuantisasi, resolusi, format, dan hubungan tetangga pixel*")

        cols = st.columns(2)

        with cols[0]:
            st.subheader("Sampling (Resolusi)")
            scale_factor = st.slider("Faktor Sampling (resolusi)", 0.1, 1.0, 0.5, 0.1, key="sampling")
            new_w = int(cv2_img_rgb.shape[1] * scale_factor)
            new_h = int(cv2_img_rgb.shape[0] * scale_factor)
            sampled = cv2.resize(cv2_img_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            st.image(sampled, caption=f"Sampling {scale_factor:.1f}x → {new_w}×{new_h} px", width="stretch")
            st.caption(f"Resolusi asli: {cv2_img_rgb.shape[1]}×{cv2_img_rgb.shape[0]} → {new_w}×{new_h}")

        with cols[1]:
            st.subheader("Kuantisasi (Level Warna)")
            bit_depth = st.slider("Bit Depth (level warna)", 1, 8, 4, 1, key="quant")
            levels = 2 ** bit_depth
            scale_q = 255 / (levels - 1)
            quantized = (np.round(gray_img / scale_q) * scale_q).astype(np.uint8)
            st.image(quantized, caption=f"Kuantisasi {bit_depth} bit ({levels} level)", width="stretch", channels="GRAY")
            st.caption(f"Level warna: {levels} (bit depth: {bit_depth})")

        st.markdown("---")

        # Image format information
        with st.expander("ℹ️ Format Citra Digital"):
            st.markdown("""
            | Format | Tipe | Kompresi | Alpha Channel |
            |---|---|---|---|
            | **JPG/JPEG** | Lossy | Tinggi | Tidak |
            | **PNG** | Lossless | Rendah | Ya |
            | **BMP** | Uncompressed | Tidak | Tidak |
            """)
            st.info(f"Format file yang diupload: **{uploaded_file.type}**")

        st.markdown("---")

        # Pixel Neighborhood
        st.subheader("🔍 Hubungan Tetangga Pixel (4-Neighbour & 8-Neighbour)")
        col_n1, col_n2 = st.columns(2)

        y_coord = st.number_input("Baris (y)", 0, gray_img.shape[0] - 1, gray_img.shape[0] // 2, key="ny")
        x_coord = st.number_input("Kolom (x)", 0, gray_img.shape[1] - 1, gray_img.shape[1] // 2, key="nx")

        # Get pixel value
        center_val = gray_img[y_coord, x_coord]

        # 4-neighbourhood
        n4 = {}
        if y_coord > 0: n4["Atas (N)"] = gray_img[y_coord - 1, x_coord]
        if y_coord < gray_img.shape[0] - 1: n4["Bawah (S)"] = gray_img[y_coord + 1, x_coord]
        if x_coord > 0: n4["Kiri (W)"] = gray_img[y_coord, x_coord - 1]
        if x_coord < gray_img.shape[1] - 1: n4["Kanan (E)"] = gray_img[y_coord, x_coord + 1]

        # 8-neighbourhood
        n8 = dict(n4)
        if y_coord > 0 and x_coord > 0: n8["Atas-Kiri (NW)"] = gray_img[y_coord - 1, x_coord - 1]
        if y_coord > 0 and x_coord < gray_img.shape[1] - 1: n8["Atas-Kanan (NE)"] = gray_img[y_coord - 1, x_coord + 1]
        if y_coord < gray_img.shape[0] - 1 and x_coord > 0: n8["Bawah-Kiri (SW)"] = gray_img[y_coord + 1, x_coord - 1]
        if y_coord < gray_img.shape[0] - 1 and x_coord < gray_img.shape[1] - 1: n8["Bawah-Kanan (SE)"] = gray_img[y_coord + 1, x_coord + 1]

        with col_n1:
            st.markdown("**4-Neighbourhood**")
            neighbor_data_4 = {"Posisi": ["Pusat"] + list(n4.keys()), "Nilai Pixel": [int(center_val)] + [int(v) for v in n4.values()]}
            st.dataframe(neighbor_data_4, use_container_width=True)

            # Visual grid 3x3 for 4-neighbour
            grid4 = np.zeros((3, 3), dtype=np.uint8)
            grid4[1, 1] = center_val
            if "Atas (N)" in n4: grid4[0, 1] = n4["Atas (N)"]
            if "Bawah (S)" in n4: grid4[2, 1] = n4["Bawah (S)"]
            if "Kiri (W)" in n4: grid4[1, 0] = n4["Kiri (W)"]
            if "Kanan (E)" in n4: grid4[1, 2] = n4["Kanan (E)"]
            fig4, ax4 = plt.subplots(figsize=(3, 3))
            ax4.imshow(grid4, cmap="gray", vmin=0, vmax=255)
            for i in range(3):
                for j in range(3):
                    ax4.text(j, i, str(grid4[i, j]), ha="center", va="center", fontsize=12, fontweight="bold",
                             color="red" if (i == 1 and j == 1) else "white")
            ax4.set_title("4-Neighbour Grid")
            ax4.axis("off")
            st.pyplot(fig4)

        with col_n2:
            st.markdown("**8-Neighbourhood**")
            neighbor_data_8 = {"Posisi": ["Pusat"] + list(n8.keys()), "Nilai Pixel": [int(center_val)] + [int(v) for v in n8.values()]}
            st.dataframe(neighbor_data_8, use_container_width=True)

            grid8 = np.zeros((3, 3), dtype=np.uint8)
            grid8[1, 1] = center_val
            pos_map = {
                "Atas (N)": (0, 1), "Bawah (S)": (2, 1), "Kiri (W)": (1, 0), "Kanan (E)": (1, 2),
                "Atas-Kiri (NW)": (0, 0), "Atas-Kanan (NE)": (0, 2),
                "Bawah-Kiri (SW)": (2, 0), "Bawah-Kanan (SE)": (2, 2),
            }
            for name, (r, c) in pos_map.items():
                if name in n8:
                    grid8[r, c] = n8[name]
            fig8, ax8 = plt.subplots(figsize=(3, 3))
            ax8.imshow(grid8, cmap="gray", vmin=0, vmax=255)
            for i in range(3):
                for j in range(3):
                    ax8.text(j, i, str(grid8[i, j]), ha="center", va="center", fontsize=12, fontweight="bold",
                             color="red" if (i == 1 and j == 1) else "white")
            ax8.set_title("8-Neighbour Grid")
            ax8.axis("off")
            st.pyplot(fig8)

    # ========================================================
    # TAB 3: OPERASI GEOMETRI
    # ========================================================
    with tab3:
        st.header("🔄 Operasi Aritmatika & Geometri Citra")

        ops = ["Grayscale", "Rotasi", "Flipping", "Cropping", "Scaling (Resize)", "Negasi"]
        selected_ops = st.multiselect("Pilih operasi yang ingin ditampilkan:", ops, default=["Grayscale", "Rotasi", "Flipping"])

        result_img = cv2_img_rgb.copy()
        show_result = False

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if "Grayscale" in selected_ops:
                st.subheader("Grayscale")
                st.image(gray_img, caption="Hasil Konversi Grayscale", width="stretch", channels="GRAY")
                with st.expander("ℹ️ Penjelasan"):
                    st.markdown("Mengkonversi citra RGB ke grayscale menggunakan rumus: "
                                "$Gray = 0.299R + 0.587G + 0.114B$")

            if "Rotasi" in selected_ops:
                st.subheader("Rotasi")
                angle = st.slider("Sudut Rotasi (°)", -180, 180, 45, 5, key="rot_angle")
                h, w = cv2_img_rgb.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(cv2_img_rgb, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                st.image(rotated, caption=f"Rotasi {angle}°", width="stretch")
                show_result = True
                result_img = rotated

            if "Flipping" in selected_ops:
                st.subheader("Flipping")
                flip_dir = st.radio("Arah Flip:", ["Horizontal", "Vertikal", "Horizontal & Vertikal"], horizontal=True, key="flip_dir")
                flip_code = {"Horizontal": 1, "Vertikal": 0, "Horizontal & Vertikal": -1}[flip_dir]
                flipped = cv2.flip(cv2_img_rgb, flip_code)
                st.image(flipped, caption=f"Flip {flip_dir}", width="stretch")
                show_result = True
                result_img = flipped

        with col_g2:
            if "Cropping" in selected_ops:
                st.subheader("Cropping")
                h, w = cv2_img_rgb.shape[:2]
                x1 = st.number_input("X awal", 0, w - 1, 0, key="crop_x1")
                y1 = st.number_input("Y awal", 0, h - 1, 0, key="crop_y1")
                x2 = st.number_input("X akhir", 0, w - 1, w // 2, key="crop_x2")
                y2 = st.number_input("Y akhir", 0, h - 1, h // 2, key="crop_y2")
                if x1 < x2 and y1 < y2:
                    cropped = cv2_img_rgb[y1:y2, x1:x2].copy()
                    st.image(cropped, caption=f"Crop ({x1},{y1}) - ({x2},{y2})", width="stretch")
                    show_result = True
                    result_img = cropped
                else:
                    st.warning("X awal < X akhir dan Y awal < Y akhir")

            if "Scaling (Resize)" in selected_ops:
                st.subheader("Scaling (Resize)")
                scale_pct = st.slider("Skala (%)", 10, 200, 80, 10, key="scale_pct")
                new_w = int(cv2_img_rgb.shape[1] * scale_pct / 100)
                new_h = int(cv2_img_rgb.shape[0] * scale_pct / 100)
                interp = st.selectbox("Interpolasi", ["INTER_LINEAR", "INTER_NEAREST", "INTER_CUBIC", "INTER_LANCZOS4"], key="interp")
                interp_map = {
                    "INTER_LINEAR": cv2.INTER_LINEAR,
                    "INTER_NEAREST": cv2.INTER_NEAREST,
                    "INTER_CUBIC": cv2.INTER_CUBIC,
                    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
                }
                scaled = cv2.resize(cv2_img_rgb, (new_w, new_h), interpolation=interp_map[interp])
                st.image(scaled, caption=f"Resize {scale_pct}% → {new_w}×{new_h}", width="stretch")
                st.caption(f"Dimensi asli: {w}×{h} → Dimensi baru: {new_w}×{new_h}")
                show_result = True
                result_img = scaled

            if "Negasi" in selected_ops:
                st.subheader("Negasi (Invers Warna)")
                negated = 255 - cv2_img_rgb
                st.image(negated, caption="Negasi Citra", width="stretch")
                show_result = True
                result_img = negated

        if show_result:
            st.markdown("---")
            col_preview = st.columns([1, 2, 1])
            with col_preview[1]:
                st.caption("Preview hasil operasi terakhir")
                st.image(result_img, width="stretch")

    # ========================================================
    # TAB 4: DETEKSI TEPI
    # ========================================================
    with tab4:
        st.header("✏️ Deteksi Tepi")
        st.markdown("*Implementasi metode Sobel, Prewitt, Robert Cross, dan Laplacian*")

        edge_methods = st.multiselect(
            "Pilih metode deteksi tepi:",
            ["Sobel", "Prewitt", "Robert Cross", "Laplacian"],
            default=["Sobel", "Laplacian"]
        )
        thresh_edge = st.slider("Threshold tepi (opsional)", 0, 255, 50, 5, key="edge_thresh")

        cols_edge = st.columns(4)

        edge_results = {}

        if "Sobel" in edge_methods:
            with cols_edge[0]:
                st.subheader("Sobel")
                sobel_res = apply_sobel(gray_img)
                _, sobel_th = cv2.threshold(sobel_res, thresh_edge, 255, cv2.THRESH_BINARY)
                edge_results["Sobel"] = (sobel_res, sobel_th)
                st.image(sobel_res, caption="Sobel Gradient", width="stretch", channels="GRAY")
                st.image(sobel_th, caption=f"Sobel Threshold {thresh_edge}", width="stretch", channels="GRAY")

        if "Prewitt" in edge_methods:
            with cols_edge[1]:
                st.subheader("Prewitt")
                prewitt_res = apply_prewitt(gray_img)
                _, prewitt_th = cv2.threshold(prewitt_res, thresh_edge, 255, cv2.THRESH_BINARY)
                edge_results["Prewitt"] = (prewitt_res, prewitt_th)
                st.image(prewitt_res, caption="Prewitt Gradient", width="stretch", channels="GRAY")
                st.image(prewitt_th, caption=f"Prewitt Threshold {thresh_edge}", width="stretch", channels="GRAY")

        if "Robert Cross" in edge_methods:
            with cols_edge[2]:
                st.subheader("Robert Cross")
                robert_res = apply_robert(gray_img)
                _, robert_th = cv2.threshold(robert_res, thresh_edge, 255, cv2.THRESH_BINARY)
                edge_results["Robert Cross"] = (robert_res, robert_th)
                st.image(robert_res, caption="Robert Gradient", width="stretch", channels="GRAY")
                st.image(robert_th, caption=f"Robert Threshold {thresh_edge}", width="stretch", channels="GRAY")

        if "Laplacian" in edge_methods:
            with cols_edge[3]:
                st.subheader("Laplacian")
                lap_res = apply_laplacian(gray_img)
                _, lap_th = cv2.threshold(lap_res, thresh_edge, 255, cv2.THRESH_BINARY)
                edge_results["Laplacian"] = (lap_res, lap_th)
                st.image(lap_res, caption="Laplacian", width="stretch", channels="GRAY")
                st.image(lap_th, caption=f"Laplacian Threshold {thresh_edge}", width="stretch", channels="GRAY")

        if edge_results:
            st.markdown("---")
            st.subheader("Perbandingan Metode Deteksi Tepi")
            n_methods = len(edge_results)
            comp_cols = st.columns(n_methods)
            for idx, (name, _) in enumerate(edge_results.items()):
                with comp_cols[idx]:
                    st.caption(f"**{name}**")
                    res_img, th_img = edge_results[name]
                    # Combined view
                    combined = np.hstack([res_img, th_img])
                    st.image(combined, caption=f"{name}: Gradient | Threshold", width="stretch", channels="GRAY")

    # ========================================================
    # TAB 5: SEGMENTASI & VOLUME
    # ========================================================
    with tab5:
        st.header("♻️ Segmentasi & Perhitungan Volume Sampah Plastik")
        st.markdown("*Otsu Thresholding, Operasi Morfologi, dan estimasi volume*")

        st.info("""
        **Alur Segmentasi:**
        1. Konversi RGB → Grayscale
        2. Otsu Thresholding untuk segmentasi awal
        3. Operasi morfologi untuk membersihkan hasil segmentasi
        4. Overlay hasil segmentasi pada citra asli
        5. Perhitungan luas area & estimasi volume sampah
        """)

        col_seg1, col_seg2 = st.columns(2)

        with col_seg1:
            st.subheader("1. Otsu Thresholding")
            st.image(gray_img, caption="Citra Grayscale Input", width="stretch", channels="GRAY")

            # Show Otsu threshold value
            _, otsu_test = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_val = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
            st.success(f"Nilai threshold Otsu otomatis: **{otsu_val:.1f}**")

            st.image(otsu_test, caption=f"Otsu Threshold (T={otsu_val:.0f})", width="stretch", channels="GRAY")

            with st.expander("ℹ️ Tentang Otsu Thresholding"):
                st.markdown("""
                Otsu thresholding secara otomatis menentukan nilai threshold optimal
                dengan meminimalkan varians intra-class (weighted variance within classes).
                
                Rumus:  
                $\\sigma^2_w(t) = w_0(t)\\sigma^2_0(t) + w_1(t)\\sigma^2_1(t)$
                
                dimana $w_0$ dan $w_1$ adalah probabilitas kedua kelas yang dipisah oleh threshold $t$.
                """)

        with col_seg2:
            st.subheader("2. Operasi Morfologi")
            morph_op = st.selectbox(
                "Pilih operasi morfologi:",
                ["closing", "opening", "dilation", "erosion", "gradient"],
                format_func=lambda x: x.capitalize(),
                key="morph_op"
            )
            kernel_size = st.slider("Ukuran Kernel", 3, 21, 5, 2, key="kernel_size")
            morph_iter = st.slider("Iterasi", 1, 10, 2, 1, key="morph_iter")

            binary_otsu, morph_result = segment_otsu_morphology(
                gray_img, morph_op=morph_op,
                kernel_size=kernel_size, iterations=morph_iter
            )

            st.image(binary_otsu, caption="Otsu Binary", width="stretch", channels="GRAY")
            st.image(morph_result, caption=f"Otsu + {morph_op.capitalize()}", width="stretch", channels="GRAY")

        st.markdown("---")

        # Overlay and Volume Calculation
        st.subheader("3. Overlay Segmentasi & Perhitungan Volume")

        overlay_color = st.color_picker("Warna overlay segmentasi", "#FF0000")

        hex_color = overlay_color.lstrip("#")
        overlay_bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

        alpha_overlay = st.slider("Transparansi overlay", 0.1, 1.0, 0.5, 0.1, key="alpha_overlay")

        overlay_img = colormap_overlay(cv2_img_bgr, morph_result, color=overlay_bgr, alpha=alpha_overlay)
        overlay_pil = cv2_to_pil(overlay_img)

        col_over1, col_over2 = st.columns(2)

        with col_over1:
            st.image(overlay_pil, caption="Overlay Segmentasi pada Citra Asli", width="stretch")

        with col_over2:
            st.subheader("Hasil Perhitungan Volume")
            pixel_to_cm = st.number_input(
                "Konversi pixel ke cm (resolution)",
                min_value=0.001, max_value=1.0, value=0.026458, format="%.6f",
                help="1 pixel = ? cm. Default 0.026458 cm untuk ~96 DPI"
            )
            analysis = analyze_segmentation(morph_result, pixel_to_cm)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Total Pixel Citra", f"{analysis['total_pixels']:,}")
                st.metric("Pixel Sampah", f"{analysis['plastic_pixels']:,}")
                st.metric("Pixel Non-Sampah", f"{analysis['non_plastic_pixels']:,}")
            with col_m2:
                st.metric("Coverage Area", f"{analysis['coverage_pct']:.2f}%")
                st.metric("Luas Area Sampah", f"{analysis['area_cm2']:.2f} cm²")
                st.metric("Estimasi Volume", f"{analysis['est_volume_ml']:.2f} ml")

            # Bar chart
            fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
            categories = ["Sampah Plastik", "Non-Sampah"]
            values = [analysis['plastic_pixels'], analysis['non_plastic_pixels']]
            colors_bar = ["#e74c3c", "#2ecc71"]
            bars = ax_bar.bar(categories, values, color=colors_bar, edgecolor="black", linewidth=1.5)
            for bar, val in zip(bars, values):
                ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                           f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
            ax_bar.set_ylabel("Jumlah Pixel")
            ax_bar.set_title("Distribusi Pixel: Sampah vs Non-Sampah")
            ax_bar.set_ylim(0, max(values) * 1.15)
            st.pyplot(fig_bar)

        st.markdown("---")

        # Side-by-side comparison
        st.subheader("4. Perbandingan: Asli vs Segmentasi")

        result_rgb_display = cv2_to_pil(cv2_img_bgr)

        # Combine into single comparison
        fig_comp, axes_comp = plt.subplots(1, 3, figsize=(15, 5))
        axes_comp[0].imshow(cv2_img_rgb)
        axes_comp[0].set_title("Citra Asli (RGB)")
        axes_comp[0].axis("off")

        axes_comp[1].imshow(binary_otsu, cmap="gray")
        axes_comp[1].set_title(f"Otsu Threshold (T={otsu_val:.0f})")
        axes_comp[1].axis("off")

        axes_comp[2].imshow(morph_result, cmap="gray")
        axes_comp[2].set_title(f"Otsu + {morph_op.capitalize()}")
        axes_comp[2].axis("off")

        st.pyplot(fig_comp)

        # Download button for segmented result
        buf = BytesIO()
        seg_download = Image.fromarray(morph_result)
        seg_download.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Download Hasil Segmentasi (PNG)",
            data=byte_im,
            file_name="segmentasi_sampah_plastik.png",
            mime="image/png",
        )

    # ========================================================
    # FOOTER
    # ========================================================
    st.markdown("---")
    st.markdown(
        "<center>"
        "🌊 **Segmentasi & Perhitungan Volume Sampah Plastik Makro pada Citra Sungai**<br>"
        "Menggunakan Metode Otsu Thresholding dan Operasi Morfologi<br>"
        "<small>UAS Project Pengolahan Citra Digital</small>"
        "</center>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
