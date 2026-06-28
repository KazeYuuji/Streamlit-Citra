import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import base64

st.set_page_config(
    page_title="Segmentasi Sampah Plastik Sungai",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_image(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    pil_img = Image.open(BytesIO(bytes_data))
    # Convert PIL to OpenCV (BGR)
    opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return pil_img, opencv_img

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
            st.image(cv2_img_rgb, caption="Citra Asli", use_column_width=True)

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

        st.info("""
        **🔍 APA YANG DIPELAJARI DI TAB INI?**
        Di tab ini kita akan melihat bahwa **citra digital hanyalah sekumpulan angka (matriks)**.
        Setiap angka mewakili tingkat kecerahan (0 = hitam, 255 = putih). Kita juga akan
        mempelajari model warna RGB dan CMY, serta bagaimana citra berwarna diubah menjadi
        grayscale.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Citra Asli (RGB)")
            st.image(cv2_img_rgb, caption=f"Dimensi: {cv2_img_rgb.shape[1]} x {cv2_img_rgb.shape[0]} px", use_column_width=True)

            st.info("""
            **📌 Penjelasan:** Citra digital adalah **matriks 2 dimensi** yang setiap elemennya disebut **pixel** (picture element). 
            Citra ini memiliki **3 kanal warna**: Red (R), Green (G), Blue (B). Dimensi `{w}×{h}` berarti ada **{h} baris** dan **{w} kolom** pixel. 
            Total ada **{h * w:,} pixel**, dan setiap pixel menyimpan 3 angka (R, G, B) sehingga total data = **{h * w * 3:,} angka**.
            """.format(w=cv2_img_rgb.shape[1], h=cv2_img_rgb.shape[0]))

            # RGB Channel Visualization
            st.subheader("Kanal RGB Terpisah")
            r_ch, g_ch, b_ch = cv2.split(cv2_img_rgb)
            fig_rgb, axes = plt.subplots(1, 3, figsize=(9, 3))
            titles = ["Red Channel", "Green Channel", "Blue Channel"]
            for ax, ch, title, cmap in zip(axes, [r_ch, g_ch, b_ch], titles, ["Reds", "Greens", "Blues"]):
                im = ax.imshow(ch, cmap=cmap, vmin=0, vmax=255)
                ax.set_title(title)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig_rgb)

            st.info("""
            **📌 Penjelasan:** Citra RGB dapat dipecah menjadi 3 kanal. Setiap kanal adalah matriks 2D dengan nilai 0–255.
            - **Red Channel:** Semakin terang (putih) = semakin banyak warna merah di area tersebut.
            - **Green Channel:** Semakin terang = semakin banyak warna hijau.
            - **Blue Channel:** Semakin terang = semakin banyak warna biru.
            
            **Colorbar** disamping menunjukkan pemetaan: 0 = hitam (tanpa warna tsb), 255 = putih (penuh warna tsb).
            Pada citra sungai, air biasanya dominan di kanal **Biru** dan **Hijau**, sedangkan sampah plastik 
            biasanya terang di **semua kanal** (karena berwarna putih/terang).
            """)

            # CMY Model
            st.subheader("Model Warna CMY (Cyan-Magenta-Yellow)")
            c_ch, m_ch, y_ch = rgb_to_cmy(r_ch, g_ch, b_ch)
            fig_cmy, axes2 = plt.subplots(1, 3, figsize=(9, 3))
            for ax, ch, title in zip(axes2, [c_ch, m_ch, y_ch], ["Cyan", "Magenta", "Yellow"]):
                im = ax.imshow(ch, cmap="gray", vmin=0, vmax=255)
                ax.set_title(title)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig_cmy)

            st.info("""
            **📌 Penjelasan:** **CMY (Cyan-Magenta-Yellow)** adalah model warna **subtraktif**, kebalikan dari RGB (aditif).
            - **RGB** digunakan oleh **layar** (memancarkan cahaya) — tambah cahaya = tambah terang.
            - **CMY** digunakan oleh **percetakan/tinta** (menyerap cahaya) — tambah tinta = tambah gelap.
            - **Rumus konversi:** C = 1 - R, M = 1 - G, Y = 1 - B (nilai 0–1).
            
            **Contoh:** Pixel yang sangat merah (R=255) akan menjadi Cyan = 0 (hitam di gambar Cyan),
            karena tinta Cyan menyerap cahaya merah. Pixel yang tidak memiliki merah (R=0) akan menjadi
            Cyan = 255 (putih), artinya banyak tinta cyan diperlukan.
            """)

        with col2:
            st.subheader("Citra Grayscale (Hitam-Putih)")
            st.image(gray_img, caption="Konversi RGB → Grayscale", use_column_width=True, channels="GRAY")

            st.info("""
            **📌 Penjelasan:** **Grayscale** adalah citra dengan **1 kanal** saja (bukan 3), dengan nilai 0 (hitam) sampai 255 (putih).
            Rumus konversi menggunakan **Luminance (ITU-R BT.601)**, meniru sensitivitas mata manusia:
            
            **Gray = 0,299 × R + 0,587 × G + 0,114 × B**
            
            Mata paling sensitif ke **hijau** (bobot 0,587), lalu **merah** (0,299), lalu **biru** (0,114).
            **Mengapa perlu grayscale?** Banyak operasi PCD seperti deteksi tepi dan thresholding
            bekerja pada 1 kanal — lebih sederhana, lebih cepat, dan cukup untuk analisis pola.
            """)

            with st.expander("📐 Rumus Konversi Grayscale"):
                st.latex(r"Gray = 0.299 \times R + 0.587 \times G + 0.114 \times B")
                st.markdown("Bobot hijau paling besar karena mata manusia paling sensitif terhadap spektrum hijau.")

            # Histogram
            st.subheader("Histogram Grayscale")
            fig_hist_g, ax_h = plt.subplots(figsize=(6, 3))
            ax_h.hist(gray_img.ravel(), bins=256, range=[0, 256], color="gray", alpha=0.8)
            ax_h.set_title("Histogram Grayscale")
            ax_h.set_xlabel("Intensitas (0 = Hitam, 255 = Putih)")
            ax_h.set_ylabel("Frekuensi (Jumlah Pixel)")
            st.pyplot(fig_hist_g)

            st.info("""
            **📌 Penjelasan Histogram:** Histogram menunjukkan **distribusi intensitas pixel**.
            - **Sumbu X:** Nilai intensitas dari 0 (hitam) sampai 255 (putih).
            - **Sumbu Y:** Berapa banyak pixel yang memiliki intensitas tersebut.
            
            **Cara membaca:**
            - Jika puncak histogram berada di **kiri** (nilai kecil) → citra **gelap**.
            - Jika puncak di **kanan** (nilai besar) → citra **terang**.
            - Jika menyebar merata → citra memiliki **kontras yang baik**.
            
            Untuk segmentasi sampah plastik, kita berharap ada **dua puncak (bimodal)**: 
            satu untuk air (gelap) dan satu untuk sampah (terang) — ini ideal untuk Otsu thresholding.
            """)

            # RGB Histogram
            st.subheader("Histogram RGB")
            st.pyplot(plot_rgb_histogram(cv2_img_rgb))
            st.info("""
            **📌 Penjelasan:** Histogram RGB menampilkan distribusi intensitas **tiap kanal warna** secara bersamaan.
            - **Garis Merah** = distribusi intensitas Red channel.
            - **Garis Hijau** = distribusi intensitas Green channel.
            - **Garis Biru** = distribusi intensitas Blue channel.
            
            Dengan histogram ini kita bisa melihat **dominasi warna** dalam citra. Misalnya pada gambar sungai,
            jika garis biru dan hijau dominan di kanan, berarti air memantulkan banyak warna biru/hijau.
            Jika ketiga garis memiliki puncak di tempat yang sama, berarti citra cenderung **netral** (gray).
            """)

            # Pixel Matrix Display
            st.subheader("Matriks Pixel (Area 16×16)")
            patch = gray_img[:16, :16]
            st.pyplot(make_pixel_matrix_display(patch))
            st.info("""
            **📌 Penjelasan:** Ini adalah **esensi dari citra digital** — setiap pixel hanyalah **angka**!
            Tampilan di atas adalah area 16×16 pixel dari pojok kiri atas citra grayscale.
            - Angka **0** = hitam pekat, **255** = putih bersih.
            - Angka di antaranya = berbagai tingkat keabuan.
            
            **Inti pembelajaran:** Citra digital TIDAK lebih dari sekumpulan angka dalam matriks.
            Semua operasi PCD (filter, deteksi tepi, segmentasi) hanyalah operasi matematika
            pada angka-angka ini — penjumlahan, pengurangan, perkalian matriks, dan konvolusi.
            """)

    # ========================================================
    # TAB 2: DIGITALISASI CITRA
    # ========================================================
    with tab2:
        st.header("🔢 Digitalisasi Citra")
        st.markdown("*Sampling, kuantisasi, resolusi, format, dan hubungan tetangga pixel*")

        st.info("""
        **🔍 APA YANG DIPELAJARI DI TAB INI?**
        **Digitalisasi** adalah proses mengubah gambar nyata (analog) menjadi representasi digital.
        Dua parameter utama: **Sampling** (seberapa banyak pixel) dan **Kuantisasi** (seberapa detail
        nilai warnanya). Kita juga akan mempelajari format file dan hubungan antar pixel tetangga.
        """)

        cols = st.columns(2)

        with cols[0]:
            st.subheader("📐 Sampling (Resolusi)")
            st.markdown("**Sampling = menentukan jumlah pixel** (seberapa sering kita mengambil sampel dari gambar asli)")

            scale_factor = st.slider("Faktor Sampling (resolusi)", 0.1, 1.0, 0.5, 0.1, key="sampling")
            new_w = int(cv2_img_rgb.shape[1] * scale_factor)
            new_h = int(cv2_img_rgb.shape[0] * scale_factor)
            sampled = cv2.resize(cv2_img_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            st.image(sampled, caption=f"Sampling {scale_factor:.1f}x → {new_w}×{new_h} px", use_column_width=True)
            st.caption(f"Resolusi asli: {cv2_img_rgb.shape[1]}×{cv2_img_rgb.shape[0]} → {new_w}×{new_h}")

            st.info("""
            **📌 Penjelasan Sampling:**
            - **Sampling tinggi** (1.0×) = menggunakan semua pixel → gambar **detail**.
            - **Sampling rendah** (0.1×) = hanya sebagian kecil pixel → gambar **blur/kotak-kotak (pixelated)**.
            - Efek blok terjadi karena 1 pixel mewakili area yang lebih luas.
            
            **Rumus:** Jumlah pixel baru = faktor × jumlah pixel asli.
            - 1.0× = 100% pixel = kualitas asli
            - 0.5× = 50% pixel = 75% lebih sedikit data!
            - 0.1× = 10% pixel = 99% data hilang!
            
            **Aplikasi:** Thumbnail gambar, preview cepat, penghematan bandwidth.
            """)

        with cols[1]:
            st.subheader("🎨 Kuantisasi (Level Warna)")
            st.markdown("**Kuantisasi = menentukan jumlah level intensitas** (seberapa detail gradasi warna)")

            bit_depth = st.slider("Bit Depth (level warna)", 1, 8, 4, 1, key="quant")
            levels = 2 ** bit_depth
            scale_q = 255 / (levels - 1)
            quantized = (np.round(gray_img / scale_q) * scale_q).astype(np.uint8)
            st.image(quantized, caption=f"Kuantisasi {bit_depth} bit ({levels} level)", use_column_width=True, channels="GRAY")
            st.caption(f"Level warna: {levels} (bit depth: {bit_depth})")

            st.info("""
            **📌 Penjelasan Kuantisasi:**
            - **Bit depth** = jumlah bit yang digunakan untuk menyimpan 1 pixel.
            - **Rumus:** Jumlah level = 2^(bit_depth)
            - **8 bit** = 256 level → gradasi halus (kualitas foto).
            - **4 bit** = 16 level → mulai terlihat pita-pita warna (**posterization**).
            - **2 bit** = 4 level → hanya hitam, abu gelap, abu terang, putih.
            - **1 bit** = 2 level → hitam-putih total (biner).
            
            **Semakin rendah bit depth** → semakin sedikit detail → efek **banding** 
            (perubahan warna yang mendadak, tidak halus). Ukuran file juga lebih kecil.
            """)

        st.markdown("---")

        # Image format information
        with st.expander("📁 Format Citra Digital (JPG vs PNG vs BMP)"):
            st.markdown("""
            | Format | Tipe | Kompresi | Alpha | Kegunaan |
            |---|---|---|---|---|
            | **JPG/JPEG** | Lossy (ada info hilang) | Tinggi | ❌ Tidak | Foto, web (ukuran kecil) |
            | **PNG** | Lossless (tanpa hilang) | Rendah | ✅ Ya | Grafis, transparansi, segmentasi |
            | **BMP** | Uncompressed | Tidak ada | ❌ Tidak | Aplikasi Windows legacy |
            
            **Lossy** = sebagian data dibuang untuk memperkecil ukuran (kualitas turun).
            **Lossless** = semua data dipertahankan (kualitas sempurna, ukuran lebih besar).
            """)
            st.info(f"Format file yang diupload: **{uploaded_file.type}**")

        st.markdown("---")

        # Pixel Neighborhood
        st.subheader("🔍 Hubungan Tetangga Pixel (4-Neighbour & 8-Neighbour)")
        st.markdown("**Konsep penting untuk segmentasi, deteksi tepi, dan operasi morfologi.**")

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
            st.markdown("**4-Neighbourhood (N4)** — Hanya 4 tetangga langsung")
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
            ax4.set_title("4-Neighbour Grid (Merah = Pusat)")
            ax4.axis("off")
            st.pyplot(fig4)

            st.caption("**N4 = Atas (N) + Bawah (S) + Kiri (W) + Kanan (E)** — juga disebut **von Neumann neighborhood**")

        with col_n2:
            st.markdown("**8-Neighbourhood (N8)** — Semua tetangga termasuk diagonal")
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
            ax8.set_title("8-Neighbour Grid (Merah = Pusat)")
            ax8.axis("off")
            st.pyplot(fig8)

            st.caption("**N8 = N4 + NW + NE + SW + SE** — juga disebut **Moore neighborhood**")

        st.info("""
        **📌 Penjelasan Hubungan Tetangga:**
        - **4-Neighbour (N4):** Hanya pixel yang berbagi **sisi** dengan pixel pusat (atas, bawah, kiri, kanan).
        - **8-Neighbour (N8):** Semua pixel yang berbagi **sisi ATAU sudut** (termasuk diagonal).
        
        **Mengapa penting?** Konsep ini digunakan dalam:
        1. **Segmentasi** — menentukan apakah pixel termasuk dalam region yang sama (connected component).
        2. **Deteksi Tepi** — kernel Sobel, Prewitt menggunakan informasi tetangga untuk menghitung gradien.
        3. **Operasi Morfologi** — kernel 3×3 bekerja pada N8 dari setiap pixel.
        4. **Connected Component Labeling** — memberi label pada objek-objek terpisah dalam citra.
        
        **Coba ubah nilai baris/kolom** untuk melihat nilai pixel di berbagai posisi!
        """)

    # ========================================================
    # TAB 3: OPERASI GEOMETRI
    # ========================================================
    with tab3:
        st.header("🔄 Operasi Aritmatika & Geometri Citra")

        st.info("""
        **🔍 APA YANG DIPELAJARI DI TAB INI?**
        **Operasi geometri** mengubah posisi, ukuran, atau orientasi pixel dalam citra.
        Termasuk rotasi, flipping (pencerminan), cropping (pemotongan), scaling (perbesaran/
        pengecilan), dan negasi (pembalikan warna). Operasi ini adalah transformasi dasar
        yang sering digunakan dalam preprocessing sebelum analisis lebih lanjut.
        """)

        ops = ["Grayscale", "Rotasi", "Flipping", "Cropping", "Scaling (Resize)", "Negasi"]
        selected_ops = st.multiselect("Pilih operasi yang ingin ditampilkan:", ops, default=["Grayscale", "Rotasi", "Flipping"])

        result_img = cv2_img_rgb.copy()
        show_result = False

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if "Grayscale" in selected_ops:
                st.subheader("⚫ Grayscale (Konversi ke Abu-Abu)")
                st.image(gray_img, caption="Hasil Konversi Grayscale", use_column_width=True, channels="GRAY")
                st.info("""
                **📌 Penjelasan:** Grayscale menggabungkan 3 kanal RGB menjadi 1 kanal menggunakan rumus:
                **Gray = 0,299R + 0,587G + 0,114B**
                - Bobot hijau terbesar (0,587) karena mata paling sensitif ke hijau.
                - **Kegunaan:** Operasi lanjutan seperti deteksi tepi dan thresholding lebih sederhana
                  pada 1 kanal dibanding 3 kanal.
                - Ini adalah **operasi aritmatika** pada pixel: kombinasi linear dari nilai R, G, B.
                """)

            if "Rotasi" in selected_ops:
                st.subheader("🔄 Rotasi")
                angle = st.slider("Sudut Rotasi (°)", -180, 180, 45, 5, key="rot_angle")
                h, w = cv2_img_rgb.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(cv2_img_rgb, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                st.image(rotated, caption=f"Rotasi {angle}°", use_column_width=True)
                show_result = True
                result_img = rotated
                st.info("""
                **📌 Penjelasan Rotasi:** Rotasi memutar citra terhadap **titik pusat** (tengah gambar).
                - **Matriks rotasi 2×3** dihasilkan oleh `cv2.getRotationMatrix2D(center, angle, scale)`.
                - **Angle positif** = berlawanan arah jarum jam (CCW).
                - **Angle negatif** = searah jarum jam (CW).
                - **WarpAffine** menerapkan transformasi affine ke setiap pixel.
                - Pixel yang "keluar" dari frame diisi dengan warna tepi (`BORDER_REPLICATE`).
                """)

            if "Flipping" in selected_ops:
                st.subheader("🔁 Flipping (Pencerminan)")
                flip_dir = st.radio("Arah Flip:", ["Horizontal", "Vertikal", "Horizontal & Vertikal"], horizontal=True, key="flip_dir")
                flip_code = {"Horizontal": 1, "Vertikal": 0, "Horizontal & Vertikal": -1}[flip_dir]
                flipped = cv2.flip(cv2_img_rgb, flip_code)
                st.image(flipped, caption=f"Flip {flip_dir}", use_column_width=True)
                show_result = True
                result_img = flipped
                st.info("""
                **📌 Penjelasan Flipping:**
                - **Horizontal** (kode=1): baris tetap, kolom dibalik (seperti bercermin kiri-kanan).
                - **Vertikal** (kode=0): kolom tetap, baris dibalik (seperti tengkurap).
                - **Horizontal + Vertikal** (kode=-1): keduanya = rotasi 180°.
                - Implementasi: cukup dengan `cv2.flip(img, flipCode)`.
                """)

        with col_g2:
            if "Cropping" in selected_ops:
                st.subheader("✂️ Cropping (Pemotongan)")
                h, w = cv2_img_rgb.shape[:2]
                x1 = st.number_input("X awal", 0, w - 1, 0, key="crop_x1")
                y1 = st.number_input("Y awal", 0, h - 1, 0, key="crop_y1")
                x2 = st.number_input("X akhir", 0, w - 1, w // 2, key="crop_x2")
                y2 = st.number_input("Y akhir", 0, h - 1, h // 2, key="crop_y2")
                if x1 < x2 and y1 < y2:
                    cropped = cv2_img_rgb[y1:y2, x1:x2].copy()
                    st.image(cropped, caption=f"Crop ({x1},{y1}) - ({x2},{y2})", use_column_width=True)
                    show_result = True
                    result_img = cropped
                else:
                    st.warning("⚠️ X awal harus < X akhir dan Y awal < Y akhir")
                st.info("""
                **📌 Penjelasan Cropping:** Cropping mengambil **sub-matriks** dari matriks citra.
                - Tidak perlu fungsi khusus — cukup **array slicing** NumPy: `img[y1:y2, x1:x2]`.
                - Seperti memotong foto secara fisik — hanya menyisakan area yang diinginkan.
                - **Aplikasi:** Menghapus area yang tidak perlu, fokus pada objek tertentu (ROI = Region of Interest).
                """)

            if "Scaling (Resize)" in selected_ops:
                st.subheader("📏 Scaling / Resize (Perbesaran/Pengecilan)")
                scale_pct = st.slider("Skala (%)", 10, 200, 80, 10, key="scale_pct")
                new_w = int(cv2_img_rgb.shape[1] * scale_pct / 100)
                new_h = int(cv2_img_rgb.shape[0] * scale_pct / 100)
                interp = st.selectbox("Metode Interpolasi", ["INTER_LINEAR", "INTER_NEAREST", "INTER_CUBIC", "INTER_LANCZOS4"], key="interp")
                interp_map = {
                    "INTER_LINEAR": cv2.INTER_LINEAR,
                    "INTER_NEAREST": cv2.INTER_NEAREST,
                    "INTER_CUBIC": cv2.INTER_CUBIC,
                    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
                }
                scaled = cv2.resize(cv2_img_rgb, (new_w, new_h), interpolation=interp_map[interp])
                st.image(scaled, caption=f"Resize {scale_pct}% → {new_w}×{new_h}", use_column_width=True)
                st.caption(f"Dimensi asli: {w}×{h} → Dimensi baru: {new_w}×{new_h}")
                show_result = True
                result_img = scaled
                st.info("""
                **📌 Penjelasan Scaling:** Scaling mengubah ukuran/resolusi citra.
                **Interpolasi** = cara menentukan nilai pixel baru saat ukuran berubah:
                
                | Metode | Kecepatan | Kualitas | Efek Visual |
                |--------|-----------|----------|-------------|
                | **NEAREST** | ⚡ Tercepat | Rendah | Kotak-kotak (blok) |
                | **LINEAR** | ✅ Sedang | Sedang | Halus (default) |
                | **CUBIC** | 🐢 Lambat | Tinggi | Sangat halus |
                | **LANCZOS4** | 🐌 Terlambat | Tertinggi | Paling tajam |
                
                **Rekomendasi:** LINEAR untuk umum, CUBIC untuk foto, NEAREST untuk pixel art.
                """)

            if "Negasi" in selected_ops:
                st.subheader("🎞️ Negasi (Invers Warna / Efek Film Negatif)")
                negated = 255 - cv2_img_rgb
                st.image(negated, caption="Negasi Citra", use_column_width=True)
                show_result = True
                result_img = negated
                st.info("""
                **📌 Penjelasan Negasi:** Negasi membalikkan semua nilai pixel.
                - **Rumus:** `output = 255 - input` untuk setiap pixel, setiap kanal.
                - Pixel hitam (0) → putih (255). Pixel putih (255) → hitam (0).
                - Pixel merah (255,0,0) → cyan (0,255,255).
                - Efeknya seperti **film negatif** foto analog.
                - **Aplikasi:** Dalam segmentasi, kadang perlu menginversi mask biner.
                - Operasi ini adalah **operasi aritmatika sederhana**: pengurangan dari nilai maksimum.
                """)

        if show_result:
            st.markdown("---")
            st.subheader("📸 Preview Hasil Operasi Terakhir")
            col_preview = st.columns([1, 2, 1])
            with col_preview[1]:
                st.image(result_img, use_column_width=True)

    # ========================================================
    # TAB 4: DETEKSI TEPI
    # ========================================================
    with tab4:
        st.header("✏️ Deteksi Tepi")
        st.markdown("*Implementasi metode Sobel, Prewitt, Robert Cross, dan Laplacian*")

        st.info("""
        **🔍 APA YANG DIPELAJARI DI TAB INI?**
        **Tepi (edge)** adalah batas antara dua region dengan intensitas berbeda signifikan.
        Deteksi tepi mencari **perubahan intensitas yang besar** (gradien tinggi) menggunakan
        **konvolusi** dengan kernel khusus. Empat metode diimplementasikan: Sobel, Prewitt,
        Robert Cross, dan Laplacian.
        """)

        edge_methods = st.multiselect(
            "Pilih metode deteksi tepi:",
            ["Sobel", "Prewitt", "Robert Cross", "Laplacian"],
            default=["Sobel", "Laplacian"]
        )
        thresh_edge = st.slider("🎚️ Threshold tepi (semakin tinggi = semakin sedikit tepi yang terdeteksi)", 0, 255, 50, 5, key="edge_thresh")

        cols_edge = st.columns(4)

        edge_results = {}

        if "Sobel" in edge_methods:
            with cols_edge[0]:
                st.subheader("Sobel")
                sobel_res = apply_sobel(gray_img)
                _, sobel_th = cv2.threshold(sobel_res, thresh_edge, 255, cv2.THRESH_BINARY)
                edge_results["Sobel"] = (sobel_res, sobel_th)
                st.image(sobel_res, caption="Sobel Gradient", use_column_width=True, channels="GRAY")
                st.image(sobel_th, caption=f"Sobel Threshold {thresh_edge}", use_column_width=True, channels="GRAY")
                st.info("""
                **📌 Cara Kerja Sobel:**
                - Kernel **3×3** dengan bobot tengah lebih besar (2) untuk mengurangi noise.
                - **Sobel X:** `[-1,0,1; -2,0,2; -1,0,1]` → mendeteksi tepi **vertikal**.
                - **Sobel Y:** `[-1,-2,-1; 0,0,0; 1,2,1]` → mendeteksi tepi **horizontal**.
                - Gradien total = √(Gx² + Gy²).
                - **Keunggulan:** Paling tahan noise dibanding Prewitt/Robert.
                - **Kekurangan:** Tepi bisa lebih tebal dari yang diinginkan.
                """)

        if "Prewitt" in edge_methods:
            with cols_edge[1]:
                st.subheader("Prewitt")
                prewitt_res = apply_prewitt(gray_img)
                _, prewitt_th = cv2.threshold(prewitt_res, thresh_edge, 255, cv2.THRESH_BINARY)
                edge_results["Prewitt"] = (prewitt_res, prewitt_th)
                st.image(prewitt_res, caption="Prewitt Gradient", use_column_width=True, channels="GRAY")
                st.image(prewitt_th, caption=f"Prewitt Threshold {thresh_edge}", use_column_width=True, channels="GRAY")
                st.info("""
                **📌 Cara Kerja Prewitt:**
                - Kernel **3×3** dengan bobot **sama rata** (semua 1), tanpa penekanan di tengah.
                - **Prewitt X:** `[1,0,-1; 1,0,-1; 1,0,-1]` → tepi vertikal.
                - **Prewitt Y:** `[1,1,1; 0,0,0; -1,-1,-1]` → tepi horizontal.
                - **Perbedaan dengan Sobel:** Lebih sederhana tapi **lebih sensitif terhadap noise**
                  karena tidak ada bobot tambahan di tengah.
                - **Cocok untuk:** Citra dengan noise rendah dan tepi yang jelas.
                """)

        if "Robert Cross" in edge_methods:
            with cols_edge[2]:
                st.subheader("Robert Cross")
                robert_res = apply_robert(gray_img)
                _, robert_th = cv2.threshold(robert_res, thresh_edge, 255, cv2.THRESH_BINARY)
                edge_results["Robert Cross"] = (robert_res, robert_th)
                st.image(robert_res, caption="Robert Gradient", use_column_width=True, channels="GRAY")
                st.image(robert_th, caption=f"Robert Threshold {thresh_edge}", use_column_width=True, channels="GRAY")
                st.info("""
                **📌 Cara Kerja Robert Cross:**
                - Kernel **2×2** (paling kecil dari semua metode).
                - **Robert X:** `[1,0; 0,-1]` — deteksi tepi diagonal 45°.
                - **Robert Y:** `[0,1; -1,0]` — deteksi tepi diagonal 135°.
                - **Karakteristik:**
                  - ⚡ **Paling cepat** secara komputasi.
                  - 🔊 **Paling sensitif terhadap noise** (kernel terlalu kecil untuk smoothing).
                  - 📏 Tepi yang dihasilkan **tipis** dan bisa **terputus-putus**.
                  - 📐 Baik untuk mendeteksi tepi **diagonal**.
                """)

        if "Laplacian" in edge_methods:
            with cols_edge[3]:
                st.subheader("Laplacian")
                lap_res = apply_laplacian(gray_img)
                _, lap_th = cv2.threshold(lap_res, thresh_edge, 255, cv2.THRESH_BINARY)
                edge_results["Laplacian"] = (lap_res, lap_th)
                st.image(lap_res, caption="Laplacian", use_column_width=True, channels="GRAY")
                st.image(lap_th, caption=f"Laplacian Threshold {thresh_edge}", use_column_width=True, channels="GRAY")
                st.info("""
                **📌 Cara Kerja Laplacian:**
                - Menggunakan **turunan kedua** (second derivative), berbeda dari 3 metode lain
                  yang menggunakan turunan pertama.
                - **Kernel:** `[0,-1,0; -1,4,-1; 0,-1,0]`
                - **Karakteristik:**
                  - Mendeteksi tepi dari **zero-crossing** (perubahan tanda turunan).
                  - **Omnidirectional** — tidak membedakan arah tepi.
                  - 🔊 **Sangat sensitif terhadap noise** — sering dikombinasikan dengan
                    Gaussian blur (LoG = Laplacian of Gaussian).
                  - 👻 Sering menghasilkan **double edge** (garis ganda) di sekitar tepi.
                """)

        if edge_results:
            st.markdown("---")
            st.subheader("📊 Perbandingan Metode Deteksi Tepi")
            st.info("""
            **📌 Perbandingan Keempat Metode:**
            
            | Metode | Ukuran Kernel | Derivatif | Tahan Noise | Ketebalan Tepi | Keunggulan |
            |--------|:------------:|:---------:|:-----------:|:--------------:|------------|
            | **Sobel** | 3×3 | Pertama | ✅ Sedang | Sedang | Paling seimbang |
            | **Prewitt** | 3×3 | Pertama | ⚠️ Rendah | Sedang | Sederhana & cepat |
            | **Robert** | 2×2 | Pertama | ❌ Sangat Rendah | Tipis | Deteksi diagonal |
            | **Laplacian** | 3×3 | Kedua | ❌ Sangat Rendah | Ganda | Omnidirectional |
            
            **Aturan Threshold:** Nilai threshold yang lebih rendah → lebih banyak tepi terdeteksi
            (termasuk noise). Nilai lebih tinggi → tepi lebih bersih tapi mungkin kehilangan tepi yang lemah.
            """)
            n_methods = len(edge_results)
            comp_cols = st.columns(n_methods)
            for idx, (name, _) in enumerate(edge_results.items()):
                with comp_cols[idx]:
                    st.caption(f"**{name}**")
                    res_img, th_img = edge_results[name]
                    combined = np.hstack([res_img, th_img])
                    st.image(combined, caption=f"{name}: Gradient | Threshold", use_column_width=True, channels="GRAY")

    # ========================================================
    # TAB 5: SEGMENTASI & VOLUME — INTI APLIKASI
    # ========================================================
    with tab5:
        st.header("♻️ Segmentasi & Perhitungan Volume Sampah Plastik")
        st.markdown("*Otsu Thresholding, Operasi Morfologi, Overlay, dan Estimasi Volume*")

        st.info("""
        **🎯 TAB INI ADALAH INTI DARI APLIKASI INI!**
        
        **Alur Segmentasi Lengkap:**
        1. **Konversi RGB → Grayscale** — menyederhanakan citra ke 1 kanal.
        2. **Otsu Thresholding** — memisahkan sampah (terang) dari air (gelap) secara otomatis.
        3. **Operasi Morfologi** — membersihkan hasil segmentasi (noise, lubang, tepi kasar).
        4. **Overlay** — menampilkan area sampah dengan warna pada citra asli.
        5. **Perhitungan Volume** — estimasi luas area dan volume sampah plastik.
        """)

        col_seg1, col_seg2 = st.columns(2)

        with col_seg1:
            st.subheader("1️⃣ Otsu Thresholding — Segmentasi Otomatis")
            st.image(gray_img, caption="Langkah 1: Citra Grayscale Input", use_column_width=True, channels="GRAY")

            # Show Otsu threshold value
            _, otsu_test = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_val = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
            st.success(f"✅ Nilai threshold Otsu otomatis: **{otsu_val:.1f}**")

            st.image(otsu_test, caption=f"Langkah 2: Otsu Threshold (T={otsu_val:.0f}) → Putih = Sampah, Hitam = Air", use_column_width=True, channels="GRAY")

            st.info("""
            **📌 Penjelasan Otsu Thresholding:**
            
            **Thresholding** mengubah citra grayscale menjadi **biner** (hanya 0 dan 255) 
            berdasarkan nilai ambang (threshold).
            
            **Otsu's method** menemukan threshold **optimal secara otomatis** dengan:
            1. Menghitung histogram intensitas citra.
            2. Untuk setiap threshold t (0–255), pixel dipisah ke 2 kelas.
            3. Menghitung **within-class variance** (varians di dalam kelas).
            4. Memilih t yang **meminimalkan** varians tersebut.
            
            **Rumus Otsu:**
            $\\sigma^2_w(t) = w_0(t)\\sigma^2_0(t) + w_1(t)\\sigma^2_1(t)$
            
            Dimana:
            - $w_0, w_1$ = probabilitas masing-masing kelas (foreground/background)
            - $\\sigma^2_0, \\sigma^2_1$ = varians intensitas masing-masing kelas
            
            **Mengapa Otsu cocok untuk sampah plastik?**
            Sampah plastik floating (botol, kantong, styrofoam) biasanya **lebih terang**
            dari air sungai yang cenderung gelap/hijau kecoklatan. Otsu secara efektif
            memisahkan dua kelompok intensitas yang kontras ini.
            
            **Output Otsu:** Citra biner dengan pixel **putih = sampah**, **hitam = bukan sampah (air)**.
            """)

            with st.expander("📐 Detail Rumus Otsu Thresholding"):
                st.markdown("""
                Otsu thresholding secara otomatis menentukan nilai threshold optimal
                dengan meminimalkan varians intra-class (weighted variance within classes).
                
                **Langkah-langkah Otsu:**
                1. Hitung histogram citra (256 bins).
                2. Untuk setiap threshold $t$ dari 0 sampai 255:
                   - $w_0(t) = \\sum_{i=0}^{t} p(i)$ = probabilitas kelas 1 (foreground)
                   - $w_1(t) = \\sum_{i=t+1}^{255} p(i)$ = probabilitas kelas 2 (background)
                   - $\\mu_0(t) = \\frac{\\sum_{i=0}^{t} i \\cdot p(i)}{w_0(t)}$ = mean kelas 1
                   - $\\mu_1(t) = \\frac{\\sum_{i=t+1}^{255} i \\cdot p(i)}{w_1(t)}$ = mean kelas 2
                   - $\\sigma^2_w(t) = w_0(t)\\sigma^2_0(t) + w_1(t)\\sigma^2_1(t)$
                3. Pilih $t$ yang **meminimalkan** $\\sigma^2_w(t)$.
                
                Otsu bekerja optimal ketika histogram memiliki **2 puncak (bimodal)**,
                yang umum terjadi pada citra sungai dengan sampah plastik (terang vs gelap).
                """)

        with col_seg2:
            st.subheader("2️⃣ Operasi Morfologi — Membersihkan Hasil Segmentasi")
            st.markdown("**Memperbaiki kualitas segmentasi dengan operasi pada bentuk (morfologi) objek**")

            morph_op = st.selectbox(
                "Pilih operasi morfologi:",
                ["closing", "opening", "dilation", "erosion", "gradient"],
                format_func=lambda x: {
                    "closing": "Closing (Tutup Lubang) ✅",
                    "opening": "Opening (Hilangkan Noise) ✅",
                    "dilation": "Dilation (Perbesar Objek)",
                    "erosion": "Erosion (Kecilkan Objek)",
                    "gradient": "Gradient (Tepi Objek)"
                }.get(x, x.capitalize()),
                key="morph_op"
            )
            kernel_size = st.slider("Ukuran Kernel (struktur elemen)", 3, 21, 5, 2, key="kernel_size")
            morph_iter = st.slider("Jumlah Iterasi (pengulangan)", 1, 10, 2, 1, key="morph_iter")

            binary_otsu, morph_result = segment_otsu_morphology(
                gray_img, morph_op=morph_op,
                kernel_size=kernel_size, iterations=morph_iter
            )

            st.image(binary_otsu, caption="Sebelum: Otsu Binary (masih ada noise & lubang)", use_column_width=True, channels="GRAY")
            st.image(morph_result, caption=f"Sesudah: Otsu + {morph_op.capitalize()}", use_column_width=True, channels="GRAY")

            morph_explanations = {
                "erosion": """
                **📌 EROSION (Erosi):** 
                - **Cara kerja:** Pixel pusat = 1 **HANYA jika SEMUA** pixel dalam kernel = 1.
                - **Efek:** ✅ Menghilangkan noise putih kecil. ❌ Objek mengecil.
                - **Rumus:** $A \\ominus B = \\{z | (B)_z \\subseteq A\\}$
                - **Aplikasi:** Memisahkan objek yang nyambung tipis, menghilangkan titik noise.
                """,
                "dilation": """
                **📌 DILATION (Dilasi):**
                - **Cara kerja:** Pixel pusat = 1 **jika MINIMAL SATU** pixel dalam kernel = 1.
                - **Efek:** ✅ Mengisi lubang kecil, menyambung objek terputus. ❌ Objek membesar.
                - **Rumus:** $A \\oplus B = \\{z | (\\hat{B})_z \\cap A \\neq \\emptyset\\}$
                - **Aplikasi:** Menggabungkan bagian sampah yang terpotong oleh threshold.
                """,
                "opening": """
                **📌 OPENING (Erosi → Dilasi):**
                - **Cara kerja:** Erosi dulu, baru dilasi dengan kernel yang sama.
                - **Efek:** ✅ Menghilangkan noise kecil TANPA mengubah ukuran objek utama.
                - **Rumus:** $A \\circ B = (A \\ominus B) \\oplus B$
                - **Aplikasi:** Membersihkan titik-titik putih di area air (false positive).
                """,
                "closing": """
                **📌 CLOSING (Dilasi → Erosi):**
                - **Cara kerja:** Dilasi dulu, baru erosi dengan kernel yang sama.
                - **Efek:** ✅ Mengisi lubang kecil dalam objek TANPA mengubah ukuran.
                - **Rumus:** $A \\bullet B = (A \\oplus B) \\ominus B$
                - **Aplikasi:** Mengisi rongga dalam area sampah (false negative di dalam objek).
                """,
                "gradient": """
                **📌 GRADIENT MORFOLOGI (Dilasi − Erosi):**
                - **Cara kerja:** Kurangkan hasil erosi dari hasil dilasi.
                - **Efek:** ✅ Menampilkan **tepi/kontur** dari objek biner.
                - **Rumus:** $G(A) = (A \\oplus B) - (A \\ominus B)$
                - **Aplikasi:** Mendapatkan outline/contour objek sampah untuk visualisasi batas.
                """
            }
            st.info(morph_explanations.get(morph_op, ""))

        st.markdown("---")

        # Overlay and Volume Calculation
        st.subheader("3️⃣ Overlay Segmentasi pada Citra Asli & Perhitungan Volume")

        st.info("""
        **📌 Penjelasan Overlay:** Overlay menggabungkan **mask biner** (hasil segmentasi) 
        dengan **citra asli** sehingga area yang terdeteksi sebagai sampah diberi warna 
        tertentu (default merah). Ini memudahkan verifikasi visual — apakah yang terdeteksi
        benar-benar sampah atau bukan.
        """)

        overlay_color = st.color_picker("🎨 Pilih warna overlay untuk area sampah", "#FF0000")

        hex_color = overlay_color.lstrip("#")
        overlay_bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

        alpha_overlay = st.slider("🔮 Tingkat transparansi overlay", 0.1, 1.0, 0.5, 0.1, key="alpha_overlay")

        overlay_img = colormap_overlay(cv2_img_bgr, morph_result, color=overlay_bgr, alpha=alpha_overlay)
        overlay_pil = cv2_to_pil(overlay_img)

        col_over1, col_over2 = st.columns(2)

        with col_over1:
            st.image(overlay_pil, caption="🖼️ Overlay: Area berwarna = sampah terdeteksi", use_column_width=True)
            st.caption("Alpha blending: `output = (1-α) × original + α × colored_mask`")

        with col_over2:
            st.subheader("📊 Hasil Perhitungan Volume Sampah")
            pixel_to_cm = st.number_input(
                "📐 Konversi 1 pixel ke cm (resolution)",
                min_value=0.001, max_value=1.0, value=0.026458, format="%.6f",
                help="1 pixel = ? cm. Default 0.026458 cm untuk ~96 DPI. Jika tahu resolusi kamera, sesuaikan nilainya."
            )
            analysis = analyze_segmentation(morph_result, pixel_to_cm)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("📦 Total Pixel Citra", f"{analysis['total_pixels']:,}")
                st.metric("🥤 Pixel Sampah", f"{analysis['plastic_pixels']:,}")
                st.metric("💧 Pixel Non-Sampah (Air)", f"{analysis['non_plastic_pixels']:,}")
            with col_m2:
                st.metric("📊 Coverage Area", f"{analysis['coverage_pct']:.2f}%")
                st.metric("📏 Luas Area Sampah", f"{analysis['area_cm2']:.2f} cm²")
                st.metric("🧴 Estimasi Volume", f"{analysis['est_volume_ml']:.2f} ml")

            st.info("""
            **📌 Penjelasan Metrik Perhitungan:**
            
            | Metrik | Rumus | Makna |
            |--------|-------|-------|
            | **Total Pixel** | `tinggi × lebar` | Ukuran citra (resolusi) |
            | **Pixel Sampah** | `sum(mask > 0)` | Jumlah pixel yang terdeteksi sebagai sampah |
            | **Coverage** | `(pixel_sampah / total) × 100%` | Persentase area sungai yang tertutup sampah |
            | **Luas Area** | `pixel_sampah × (pixel_to_cm)²` | Estimasi luas dalam cm² |
            | **Estimasi Volume** | `luas_area × 0.5 cm` | Volume dengan asumsi ketebalan sampah ~5mm |
            
            **Catatan:** Nilai `pixel_to_cm` dapat disesuaikan. Default 0.026458 cm/pixel
            (berdasarkan 96 DPI: 1 inch = 2.54 cm, 1 pixel = 2.54/96 cm).
            
            **Estimasi volume** bersifat aproksimasi dengan asumsi ketebalan rata-rata
            sampah plastik floating sekitar 0.5 cm (5 mm). Untuk presisi lebih tinggi,
            diperlukan kalibrasi dengan objek referensi di lapangan.
            """)

            # Bar chart
            fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
            categories = ["🥤 Sampah Plastik", "💧 Non-Sampah (Air)"]
            values = [analysis['plastic_pixels'], analysis['non_plastic_pixels']]
            colors_bar = ["#e74c3c", "#2ecc71"]
            bars = ax_bar.bar(categories, values, color=colors_bar, edgecolor="black", linewidth=1.5)
            for bar, val in zip(bars, values):
                ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                           f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
            ax_bar.set_ylabel("Jumlah Pixel")
            ax_bar.set_title("📊 Distribusi Pixel: Sampah Plastik vs Non-Sampah")
            ax_bar.set_ylim(0, max(values) * 1.15)
            st.pyplot(fig_bar)

            st.caption("**🔴 Merah** = Area sampah terdeteksi | **🟢 Hijau** = Area non-sampah (air sungai)")

        st.markdown("---")

        # Side-by-side comparison
        st.subheader("4️⃣ Perbandingan: Perjalanan dari Citra Asli → Segmentasi Akhir")

        result_rgb_display = cv2_to_pil(cv2_img_bgr)

        # Combine into single comparison
        fig_comp, axes_comp = plt.subplots(1, 3, figsize=(15, 5))
        axes_comp[0].imshow(cv2_img_rgb)
        axes_comp[0].set_title("📷 Citra Asli (RGB)\nLangkah 0: Input", fontsize=11, fontweight="bold")
        axes_comp[0].axis("off")

        axes_comp[1].imshow(binary_otsu, cmap="gray")
        axes_comp[1].set_title(f"⚫ Otsu Threshold (T={otsu_val:.0f})\nLangkah 1: Segmentasi", fontsize=11, fontweight="bold")
        axes_comp[1].axis("off")

        axes_comp[2].imshow(morph_result, cmap="gray")
        axes_comp[2].set_title(f"✅ Otsu + {morph_op.capitalize()}\nLangkah 2: Morfologi", fontsize=11, fontweight="bold")
        axes_comp[2].axis("off")

        st.pyplot(fig_comp)

        st.info("""
        **📌 Kesimpulan Segmentasi:**
        - **Gambar Kiri:** Citra asli sungai dengan sampah plastik.
        - **Gambar Tengah:** Hasil Otsu thresholding — sampah terpisah dari air, tapi masih ada noise/lubang.
        - **Gambar Kanan:** Setelah operasi morfologi **{}** — noise berkurang, objek lebih rapi.
        
        **Dari sini kita bisa melihat bahwa** kombinasi Otsu + Morfologi efektif untuk
        segmentasi sampah plastik pada citra sungai, terutama jika sampah memiliki warna
        yang kontras dengan air.
        """.format(morph_op.capitalize()))

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
        st.caption("Format PNG dipilih karena lossless — kualitas gambar biner terjaga sempurna.")

    # ========================================================
    # FOOTER — PENJELASAN AKHIR
    # ========================================================
    st.markdown("---")
    st.markdown("## 📚 PENJELASAN AKHIR — RINGKASAN MATERI PCD")

    st.markdown("""
    <div style="background-color:#f0f8ff; padding:20px; border-radius:10px; border:1px solid #b0d4e8;">

    ### 🎯 TUJUAN PROYEK
    Aplikasi ini bertujuan untuk **mendeteksi, mensegmentasi, dan memperkirakan volume sampah plastik makro**
    pada citra sungai menggunakan metode **Otsu Thresholding** dan **Operasi Morfologi**.
    Seluruh konsep Pengolahan Citra Digital (PCD) telah diintegrasikan dalam satu aplikasi utuh.

    ---

    ### 📋 LIMA PILAR PCD YANG TELAH DIIMPLEMENTASIKAN

    **1️⃣ REPRESENTASI CITRA** *(Tab 1)*
    - Citra digital = **matriks pixel** dengan nilai 0–255.
    - **Model RGB:** Setiap pixel adalah kombinasi Red (R), Green (G), Blue (B).
    - **Model CMY:** Model subtraktif kebalikan RGB, digunakan dalam percetakan.
    - **Konversi Grayscale:** Gray = 0,299R + 0,587G + 0,114B.
    - **Histogram:** Distribusi intensitas pixel — alat diagnostik kualitas citra.

    **2️⃣ DIGITALISASI CITRA** *(Tab 2)*
    - **Sampling:** Menentukan jumlah pixel (resolusi). Makin tinggi sampling → makin detail.
    - **Kuantisasi:** Menentukan jumlah level warna (bit depth). 8 bit = 256 level.
    - **Format file:** JPG (lossy, kecil), PNG (lossless, transparansi), BMP (tanpa kompresi).
    - **Hubungan pixel:** 4-Neighbour (atas/bawah/kiri/kanan) dan 8-Neighbour (termasuk diagonal).
      Konsep penting untuk segmentasi, deteksi tepi, dan morfologi.

    **3️⃣ OPERASI GEOMETRI CITRA** *(Tab 3)*
    - **Grayscale:** Kombinasi linear 3 kanal → 1 kanal.
    - **Rotasi:** Transformasi affine, memutar citra terhadap titik pusat.
    - **Flipping:** Pencerminan horizontal/vertikal — operasi array sederhana.
    - **Cropping:** Sub-matriks — memotong area tertentu dari citra.
    - **Scaling:** Mengubah ukuran dengan interpolasi (NEAREST, LINEAR, CUBIC, LANCZOS).
    - **Negasi:** Invers warna (255 − nilai pixel) — operasi aritmatika paling sederhana.

    **4️⃣ DETEKSI TEPI** *(Tab 4)*
    - **Sobel:** Kernel 3×3 dengan bobot tengah diperkuat — paling tahan noise.
    - **Prewitt:** Kernel 3×3 bobot sama — sederhana tapi sensitif noise.
    - **Robert Cross:** Kernel 2×2 — tercepat, terbaik untuk tepi diagonal.
    - **Laplacian:** Turunan kedua — omnidirectional, sering hasilkan double edge.
    - Semua metode bekerja dengan **konvolusi** antara citra dan kernel.

    **5️⃣ SEGMENTASI CITRA** *(Tab 5) — ⭐ INTI APLIKASI*
    - **Otsu Thresholding:** Menemukan threshold optimal secara otomatis dengan
      meminimalkan within-class variance. Ideal untuk citra dengan distribusi bimodal.
    - **Operasi Morfologi:**
      - *Erosi:* Menghilangkan noise, mengecilkan objek.
      - *Dilasi:* Mengisi lubang, membesarkan objek.
      - *Opening:* Erosi → Dilasi (hilangkan noise tanpa ubah ukuran).
      - *Closing:* Dilasi → Erosi (isi lubang tanpa ubah ukuran).
      - *Gradient:* Dilasi − Erosi (dapatkan kontur/tepi objek).
    - **Perhitungan Volume:** Estimasi luas area (cm²) dan volume (ml) berdasarkan
      jumlah pixel terdeteksi dan konversi pixel-to-cm.

    ---

    ### 💡 KESIMPULAN AKHIR

    Aplikasi ini berhasil mengintegrasikan **seluruh materi Pengolahan Citra Digital**
    dalam satu kesatuan yang aplikatif dan interaktif:

    - **Dari konsep dasar** (pixel, matriks, warna) → **hingga aplikasi nyata**
      (deteksi & estimasi volume sampah plastik di sungai).
    - **Dari teori** (rumus konversi, kernel konvolusi) → **implementasi kode**
      (OpenCV, NumPy, Streamlit).
    - **Dari input** (upload gambar) → **output bermakna** (metrik volume, visualisasi overlay).

    Dengan pendekatan **Otsu Thresholding** + **Operasi Morfologi**, sampah plastik makro
    pada citra sungai dapat disegmentasi secara otomatis. Metode ini bekerja efektif ketika
    terdapat kontras yang cukup antara sampah (terang) dan air (gelap).

    **Pengembangan lebih lanjut:** Untuk akurasi yang lebih tinggi, dapat dikombinasikan dengan
    *deep learning* (CNN, U-Net) atau *color space transformation* (HSV, LAB) untuk menangani
    variasi kondisi pencahayaan dan jenis sampah.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<center>"
        "🌊 **Segmentasi & Perhitungan Volume Sampah Plastik Makro pada Citra Sungai**<br>"
        "Menggunakan Metode Otsu Thresholding dan Operasi Morfologi<br>"
        "<small>© UAS Project Pengolahan Citra Digital</small>"
        "</center>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
