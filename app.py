import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os

st.set_page_config(
    page_title="Segmentasi & Perhitungan Volume Sampah Plastik Makro",
    page_icon="🌊",
    layout="wide"
)

st.markdown("""
<style>
    .main > div { padding: 1rem 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; border-radius: 4px 4px 0 0; }
    .info-box { background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0; }
    .metric-card { background-color: #e8f4fd; padding: 1rem; border-radius: 0.5rem; text-align: center; border-left: 4px solid #4CAF50; }
</style>
""", unsafe_allow_html=True)

if 'img_rgb' not in st.session_state:
    st.session_state.img_rgb = None
if 'img_gray' not in st.session_state:
    st.session_state.img_gray = None
if 'img_bgr' not in st.session_state:
    st.session_state.img_bgr = None

st.sidebar.title("🌊 Aplikasi Pengolahan Citra")
st.sidebar.markdown("**Segmentasi & Perhitungan Volume Sampah Plastik Makro**")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Citra Sungai",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Format: JPG, PNG, BMP"
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.sidebar.error("Gagal membaca file. Coba upload file lain.")
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        st.session_state.img_rgb = img_rgb
        st.session_state.img_gray = img_gray
        st.session_state.img_bgr = img_bgr
        st.sidebar.success(f"✅ Berhasil: {uploaded_file.name}")
        st.sidebar.markdown(f"**Dimensi:** {img_rgb.shape[1]} x {img_rgb.shape[0]} px")
        st.sidebar.markdown(f"**Format:** {uploaded_file.type or 'image'}")

@st.cache_data
def convert_rgb_to_cmy(rgb):
    norm = rgb.astype(np.float32) / 255.0
    cmy = 1.0 - norm
    return cmy[:,:,0], cmy[:,:,1], cmy[:,:,2]

def quantize_image(gray, bits):
    levels = 2 ** bits
    factor = 256 // levels
    return (gray // factor) * factor

def downsample_image(img, factor):
    h, w = img.shape[:2]
    nh, nw = max(1, h // factor), max(1, w // factor)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

def detect_edges(gray, method):
    if method == "Sobel":
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobel_x**2 + sobel_y**2)
        return np.uint8(np.clip(mag, 0, 255))
    elif method == "Prewitt":
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        px = cv2.filter2D(gray.astype(np.float32), cv2.CV_64F, kx)
        py = cv2.filter2D(gray.astype(np.float32), cv2.CV_64F, ky)
        mag = np.sqrt(px**2 + py**2)
        return np.uint8(np.clip(mag, 0, 255))
    elif method == "Robert Cross":
        kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        rx = cv2.filter2D(gray.astype(np.float32), cv2.CV_64F, kx)
        ry = cv2.filter2D(gray.astype(np.float32), cv2.CV_64F, ky)
        mag = np.sqrt(rx**2 + ry**2)
        return np.uint8(np.clip(mag, 0, 255))
    elif method == "Laplacian":
        lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_64F)
        return np.uint8(np.clip(np.abs(lap), 0, 255))
    return None

def apply_morphology(binary, operation, kernel_size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    ops = {
        "Erosi": cv2.erode,
        "Dilasi": cv2.dilate,
        "Opening": cv2.morphologyEx,
        "Closing": cv2.morphologyEx
    }
    fn = ops.get(operation)
    if fn is None:
        return binary
    if operation in ("Opening", "Closing"):
        morph_type = cv2.MORPH_OPEN if operation == "Opening" else cv2.MORPH_CLOSE
        return fn(binary, morph_type, kernel, iterations=iterations)
    return fn(binary, kernel, iterations=iterations)

def calculate_waste_metrics(binary, pixel_to_cm=None, thickness_mm=None):
    waste_pixels = np.count_nonzero(binary)
    total_pixels = binary.shape[0] * binary.shape[1]
    percentage = (waste_pixels / total_pixels) * 100
    metrics = {
        "waste_pixels": int(waste_pixels),
        "total_pixels": int(total_pixels),
        "percentage": round(percentage, 2)
    }
    if pixel_to_cm and pixel_to_cm > 0:
        area_cm2 = waste_pixels / (pixel_to_cm ** 2)
        metrics["area_cm2"] = round(area_cm2, 2)
        if thickness_mm and thickness_mm > 0:
            volume_cm3 = area_cm2 * (thickness_mm / 10)
            metrics["volume_cm3"] = round(volume_cm3, 2)
    return metrics

def show_neighbourhood(gray, cx, cy, mode):
    h, w = gray.shape
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    half = 3
    x1, x2 = max(0, cx - half), min(w, cx + half + 1)
    y1, y2 = max(0, cy - half), min(h, cy + half + 1)
    region = gray[y1:y2, x1:x2].copy()
    overlay = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR).astype(np.uint16)
    lx, ly = cx - x1, cy - y1
    overlay[ly, lx] = [0, 255, 0]
    if mode == "4-Neighbour":
        offsets = [(-1,0),(1,0),(0,-1),(0,1)]
    else:
        offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for dy, dx in offsets:
        ny, nx = ly + dy, lx + dx
        if 0 <= ny < region.shape[0] and 0 <= nx < region.shape[1]:
            overlay[ny, nx] = [0, 0, 255]
    return np.uint8(np.clip(overlay, 0, 255))

# ===== SIDEBAR CONTROLS =====
with st.sidebar:
    st.markdown("---")
    st.markdown("**Informasi Project**")
    st.markdown("""
    - **Mata Kuliah:** Pengolahan Citra Digital
    - **Metode:** Otsu Thresholding & Morfologi
    - **Fitur:** Representasi, Digitalisasi, Geometri, Deteksi Tepi, Segmentasi
    """)

# ===== MAIN CONTENT =====
if st.session_state.img_rgb is None:
    st.title("🌊 Segmentasi dan Perhitungan Volume Sampah Plastik Makro")
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Aplikasi Pengolahan Citra Digital
        
        Aplikasi ini mengimplementasikan konsep **Pengolahan Citra Digital** untuk 
        **segmentasi** dan **perhitungan volume** sampah plastik makro pada citra sungai.
        
        **Fitur Utama:**
        1. **Representasi Citra** – Matriks pixel, model RGB/CMY, histogram
        2. **Digitalisasi Citra** – Sampling, kuantisasi, hubungan antar pixel
        3. **Operasi Geometri** – Rotasi, flipping, cropping, scaling, negasi
        4. **Deteksi Tepi** – Sobel, Prewitt, Robert Cross, Laplacian
        5. **Segmentasi & Volume** – Otsu thresholding, morfologi, estimasi volume
        """)
    with col2:
        st.info("👈 Upload citra sungai melalui sidebar untuk memulai")
        st.markdown("**Format didukung:** JPG, PNG, BMP")
    st.markdown("---")
    st.stop()

if st.session_state.img_rgb is None:
    st.error("Silakan upload citra terlebih dahulu melalui sidebar.")
    st.stop()

img_rgb = st.session_state.img_rgb
img_gray = st.session_state.img_gray
img_bgr = st.session_state.img_bgr
h_orig, w_orig = img_rgb.shape[:2]

tab_repr, tab_digit, tab_geo, tab_edge, tab_seg = st.tabs([
    "1. Representasi Citra",
    "2. Digitalisasi Citra",
    "3. Operasi Geometri",
    "4. Deteksi Tepi",
    "5. Segmentasi & Volume"
])

# ====================================================================
# TAB 1: REPRESENTASI CITRA
# ====================================================================
with tab_repr:
    st.header("1. Representasi Citra")
    st.markdown("Memahami citra sebagai matriks pixel dan model warna.")
    
    st.subheader("1.1 Informasi Citra")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dimensi (px)", f"{w_orig} x {h_orig}")
    col2.metric("Jumlah Pixel", f"{w_orig * h_orig:,}")
    col3.metric("Channel", "3 (RGB)")
    
    st.subheader("1.2 Channel RGB")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Citra Original (RGB)", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    for idx, (ch, title) in enumerate([(0, 'Red Channel'), (1, 'Green Channel'), (2, 'Blue Channel')]):
        ax = axes[(idx+1)//2, (idx+1)%2]
        channel_img = np.zeros_like(img_rgb)
        channel_img[:, :, ch] = img_rgb[:, :, ch]
        ax.imshow(channel_img)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    with st.expander("📖 Penjelasan Model Warna RGB"):
        st.markdown("""
        **RGB (Red, Green, Blue)** adalah model warna aditif di mana setiap pixel terdiri dari 3 channel.
        Setiap channel memiliki nilai 0–255 (8 bit), sehingga total kombinasi warna = 256³ = 16.777.216 warna.
        Citra digital disimpan sebagai matriks 3D dengan ukuran (tinggi × lebar × 3).
        """)

    st.subheader("1.3 Konversi RGB ke Grayscale")
    st.markdown("Rumus: **Y = 0.299·R + 0.587·G + 0.114·B**")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Original RGB", use_column_width=True)
    with col2:
        st.image(img_gray, caption="Grayscale", use_column_width=True, clamp=True)
    
    st.subheader("1.4 Model Warna CMY (Cyan, Magenta, Yellow)")
    st.markdown("Rumus: **C = 1 − R, M = 1 − G, Y = 1 − B** (dengan nilai 0–1)")
    c_ch, m_ch, y_ch = convert_rgb_to_cmy(img_rgb)
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    axes2[0, 0].imshow(img_rgb)
    axes2[0, 0].set_title("Original RGB", fontsize=11, fontweight='bold')
    axes2[0, 0].axis('off')
    titles = ["Cyan Channel", "Magenta Channel", "Yellow Channel"]
    for idx, (ch_data, title) in enumerate(zip([c_ch, m_ch, y_ch], titles)):
        ax = axes2[(idx+1)//2, (idx+1)%2]
        im = ax.imshow(ch_data, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    
    st.subheader("1.5 Matriks Pixel")
    show_full_res = st.checkbox("Tampilkan dengan resolusi penuh (mungkin lambat untuk citra besar)", value=False)
    region_h, region_w = img_gray.shape
    if not show_full_res and (region_h > 20 or region_w > 20):
        step = max(1, region_h // 20, region_w // 20)
        display_matrix = img_gray[::step, ::step]
        st.info(f"Menampilkan sampling matriks (setiap {step} pixel) untuk performa. Centang checkbox di atas untuk resolusi penuh.")
    else:
        display_matrix = img_gray
    st.dataframe(display_matrix, use_container_width=True)

# ====================================================================
# TAB 2: DIGITALISASI CITRA
# ====================================================================
with tab_digit:
    st.header("2. Digitalisasi Citra")
    st.markdown("Proses pembentukan citra digital: sampling (resolusi) dan kuantisasi (level warna).")
    
    st.subheader("2.1 Sampling – Resolusi Gambar")
    sampling_factor = st.slider("Faktor Sampling (downsampling)", 1, 10, 1, 
                                 help="1 = resolusi asli, semakin besar semakin rendah resolusinya")
    if sampling_factor > 1:
        sampled_rgb = downsample_image(img_rgb, sampling_factor)
        sampled_gray = downsample_image(img_gray, sampling_factor)
        st.info(f"Resolusi diturunkan dari {w_orig}×{h_orig} → {sampled_rgb.shape[1]}×{sampled_rgb.shape[0]} pixel")
    else:
        sampled_rgb, sampled_gray = img_rgb, img_gray
    st.image(sampled_rgb, caption=f"Citra hasil sampling (faktor = {sampling_factor})", use_column_width=True)
    
    st.subheader("2.2 Kuantisasi – Level Warna")
    q_bits = st.slider("Jumlah bit per pixel (kuantisasi)", 1, 8, 8,
                        help="8 bit = 256 level (penuh), 1 bit = 2 level (biner)")
    q_img = quantize_image(img_gray, q_bits)
    q_levels = 2 ** q_bits
    st.info(f"Kuantisasi: {q_bits} bit → {q_levels} level warna")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_gray, caption="Grayscale asli (8 bit, 256 level)", use_column_width=True, clamp=True)
    with col2:
        st.image(q_img, caption=f"Hasil kuantisasi ({q_bits} bit, {q_levels} level)", use_column_width=True, clamp=True)
    
    st.subheader("2.3 Format Citra")
    col1, col2, col3 = st.columns(3)
    col1.info("**JPG/JPEG**\n\nKompresi lossy\nUkuran kecil\nCocok untuk foto")
    col2.info("**PNG**\n\nKompresi lossless\nTransparansi\nKualitas tinggi")
    col3.info("**BMP**\n\nTanpa kompresi\nUkuran besar\nKualitas sempurna")

    st.subheader("2.4 Hubungan Antar Pixel (Neighbourhood)")
    st.markdown("Klik pada citra di bawah untuk melihat hubungan 4-neighbour dan 8-neighbour dari pixel yang dipilih.")
    neigh_mode = st.radio("Mode Neighbourhood", ["4-Neighbour", "8-Neighbour"], horizontal=True)
    if neigh_mode == "4-Neighbour":
        st.markdown("**4-Neighbour:** $N_4(p)$ = pixel di atas, bawah, kiri, kanan dari pixel pusat.")
    else:
        st.markdown("**8-Neighbour:** $N_8(p)$ = semua pixel yang mengelilingi pixel pusat (termasuk diagonal).")
    neigh_info = st.empty()
    neigh_display = st.empty()
    try:
        import matplotlib.image as mpimg
        fig_n, ax_n = plt.subplots(figsize=(6, 6))
        ax_n.imshow(img_gray, cmap='gray')
        ax_n.set_title("Klik pada pixel untuk melihat neighbourhood", fontsize=10)
        ax_n.set_xlabel("X (kolom)")
        ax_n.set_ylabel("Y (baris)")
        ax_n.grid(True, alpha=0.3)
        neigh_display.pyplot(fig_n)
        plt.close()
    except:
        pass

# ====================================================================
# TAB 3: OPERASI GEOMETRI
# ====================================================================
with tab_geo:
    st.header("3. Operasi Aritmatika & Geometri Citra")
    
    st.subheader("3.1 Konversi Grayscale")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Original RGB", use_column_width=True)
    with col2:
        st.image(img_gray, caption="Hasil Grayscale", use_column_width=True, clamp=True)
    
    st.subheader("3.2 Rotasi")
    rot_angle = st.slider("Sudut Rotasi (derajat)", -180, 180, 0)
    if rot_angle != 0:
        h, w = img_rgb.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
        rotated = cv2.warpAffine(img_rgb, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        st.image(rotated, caption=f"Rotasi {rot_angle}°", use_column_width=True)
    
    st.subheader("3.3 Flipping")
    flip_choice = st.selectbox("Arah Flip", ["Tidak Ada", "Horizontal", "Vertikal", "Horizontal & Vertikal"])
    flip_map = {"Tidak Ada": None, "Horizontal": 1, "Vertikal": 0, "Horizontal & Vertikal": -1}
    flip_code = flip_map[flip_choice]
    if flip_code is not None:
        flipped = cv2.flip(img_rgb, flip_code)
        st.image(flipped, caption=f"Flip {flip_choice}", use_column_width=True)
    
    st.subheader("3.4 Cropping")
    crop_enabled = st.checkbox("Aktifkan Cropping", value=False)
    if crop_enabled:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            crop_x = st.number_input("X awal", 0, w_orig - 1, 0)
        with col2:
            crop_y = st.number_input("Y awal", 0, h_orig - 1, 0)
        with col3:
            crop_w = st.number_input("Lebar", 1, w_orig, min(300, w_orig))
        with col4:
            crop_h = st.number_input("Tinggi", 1, h_orig, min(300, h_orig))
        x1, y1 = min(crop_x, w_orig - 1), min(crop_y, h_orig - 1)
        x2, y2 = min(x1 + crop_w, w_orig), min(y1 + crop_h, h_orig)
        if x2 > x1 and y2 > y1:
            cropped = img_rgb[y1:y2, x1:x2]
            st.image(cropped, caption=f"Crop region ({x1},{y1}) → ({x2},{y2})", use_column_width=False, width=400)
    
    st.subheader("3.5 Scaling (Resize)")
    scale_method = st.selectbox("Metode Scaling", ["Persentase", "Ukuran Tetap"])
    if scale_method == "Persentase":
        scale_pct = st.slider("Skala (%)", 10, 200, 100)
        if scale_pct != 100:
            new_w = int(w_orig * scale_pct / 100)
            new_h = int(h_orig * scale_pct / 100)
            scaled = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if scale_pct < 100 else cv2.INTER_LINEAR)
            st.image(scaled, caption=f"Resize {scale_pct}% → {new_w}×{new_h}", use_column_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            fixed_w = st.number_input("Lebar (px)", 1, 2000, w_orig)
        with col2:
            fixed_h = st.number_input("Tinggi (px)", 1, 2000, h_orig)
        if fixed_w != w_orig or fixed_h != h_orig:
            scaled = cv2.resize(img_rgb, (fixed_w, fixed_h), interpolation=cv2.INTER_AREA)
            st.image(scaled, caption=f"Resize ke {fixed_w}×{fixed_h}", use_column_width=True)
    
    st.subheader("3.6 Negasi (Invers Warna)")
    negasi = cv2.bitwise_not(img_rgb)
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Original", use_column_width=True)
    with col2:
        st.image(negasi, caption="Negasi (Invers Warna)", use_column_width=True)

# ====================================================================
# TAB 4: DETEKSI TEPI
# ====================================================================
with tab_edge:
    st.header("4. Deteksi Tepi")
    st.markdown("Deteksi tepi digunakan untuk menemukan batas-batas objek dalam citra.")
    
    edge_method = st.selectbox("Pilih Metode Deteksi Tepi", ["Sobel", "Prewitt", "Robert Cross", "Laplacian"])
    edge_params = {}
    if edge_method in ("Sobel", "Prewitt", "Robert Cross"):
        edge_params["threshold"] = st.slider("Threshold (nilai minimum tepi)", 0, 255, 30,
                                              help="Semakin rendah, semakin banyak tepi yang terdeteksi")
    else:
        edge_params["threshold"] = st.slider("Threshold (nilai minimum tepi)", 0, 255, 10)
    
    blur_kernel = st.slider("Blur (perhalus citra sebelum deteksi tepi)", 1, 15, 3, step=2,
                            help="Gaussian blur untuk mengurangi noise. Gunakan nilai ganjil.")
    
    pre_gray = cv2.GaussianBlur(img_gray, (blur_kernel, blur_kernel), 0) if blur_kernel > 1 else img_gray.copy()
    edges = detect_edges(pre_gray, edge_method)
    
    if edges is not None:
        thresh = edge_params.get("threshold", 30)
        _, edges_thresh = cv2.threshold(edges, thresh, 255, cv2.THRESH_BINARY)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_rgb, caption="Original", use_column_width=True)
        with col2:
            st.image(edges, caption=f"{edge_method} (full gradien)", use_column_width=True, clamp=True)
        with col3:
            st.image(edges_thresh, caption=f"{edge_method} (threshold={thresh})", use_column_width=True, clamp=True)
        
        with st.expander(f"📖 Tentang Metode {edge_method}"):
            descriptions = {
                "Sobel": """
                **Sobel** menggunakan konvolusi kernel 3×3 untuk menghitung gradien arah horizontal ($G_x$) dan vertikal ($G_y$).
                
                $$\\text{Magnitude} = \\sqrt{G_x^2 + G_y^2}$$
                
                Kernel Sobel:
                $$G_x = \\begin{bmatrix} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1 \\end{bmatrix} \\quad 
                G_y = \\begin{bmatrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 2 & 1 \\end{bmatrix}$$
                """,
                "Prewitt": """
                **Prewitt** mirip Sobel tetapi menggunakan bobot yang sama (1) untuk semua pixel tetangga.
                
                $$G_x = \\begin{bmatrix} -1 & 0 & 1 \\\\ -1 & 0 & 1 \\\\ -1 & 0 & 1 \\end{bmatrix} \\quad 
                G_y = \\begin{bmatrix} -1 & -1 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 1 & 1 \\end{bmatrix}$$
                """,
                "Robert Cross": """
                **Robert Cross** menggunakan kernel 2×2 yang sederhana dan cepat.
                
                $$G_x = \\begin{bmatrix} 1 & 0 \\\\ 0 & -1 \\end{bmatrix} \\quad 
                G_y = \\begin{bmatrix} 0 & 1 \\\\ -1 & 0 \\end{bmatrix}$$
                
                Kernel ini sensitif terhadap tepi diagonal.
                """,
                "Laplacian": """
                **Laplacian** adalah detektor tepi orde-2 yang menggunakan turunan kedua.
                
                $$\\nabla^2 f = \\frac{\\partial^2 f}{\\partial x^2} + \\frac{\\partial^2 f}{\\partial y^2}$$
                
                Kernel Laplacian 3×3:
                $$\\begin{bmatrix} 0 & 1 & 0 \\\\ 1 & -4 & 1 \\\\ 0 & 1 & 0 \\end{bmatrix}$$
                
                Laplacian sensitif terhadap noise, sehingga perlu dilakukan blur terlebih dahulu.
                """
            }
            st.markdown(descriptions.get(edge_method, ""))

# ====================================================================
# TAB 5: SEGMENTASI & VOLUME
# ====================================================================
with tab_seg:
    st.header("5. Segmentasi & Perhitungan Volume Sampah Plastik")
    st.markdown("Menggunakan **Otsu Thresholding** dan **Operasi Morfologi** untuk segmentasi objek.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("5.1 Preprocessing")
        seg_blur = st.slider("Gaussian Blur Kernel", 1, 21, 5, step=2, 
                             help="Perhalus citra untuk mengurangi noise")
        blur_gray = cv2.GaussianBlur(img_gray, (seg_blur, seg_blur), 0)
        st.image(blur_gray, caption="Citra setelah Gaussian Blur", use_column_width=True, clamp=True)
    
    with col2:
        st.subheader("5.2 Otsu Thresholding")
        st.markdown("Otsu secara otomatis menentukan nilai threshold optimal berdasarkan histogram.")
        _, otsu_binary = cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        st.image(otsu_binary, caption="Hasil Otsu Thresholding", use_column_width=True, clamp=True)
        
        if st.checkbox("Adjust Manual Threshold", value=False):
            manual_thresh = st.slider("Threshold Manual", 0, 255, 127)
            _, manual_binary = cv2.threshold(blur_gray, manual_thresh, 255, cv2.THRESH_BINARY)
            st.image(manual_binary, caption=f"Threshold Manual ({manual_thresh})", use_column_width=True, clamp=True)
    
    st.markdown("---")
    st.subheader("5.3 Operasi Morfologi")
    st.markdown("Membersihkan hasil segmentasi dengan operasi morfologi.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        morph_op = st.selectbox("Operasi Morfologi", ["Erosi", "Dilasi", "Opening", "Closing"])
    with col2:
        kernel_size = st.slider("Ukuran Kernel", 3, 21, 5, step=2)
    with col3:
        morph_iter = st.slider("Iterasi", 1, 10, 1)
    
    morph_result = apply_morphology(otsu_binary, morph_op, kernel_size, morph_iter)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(otsu_binary, caption="Sebelum Morfologi (Otsu)", use_column_width=True, clamp=True)
    with col2:
        st.image(morph_result, caption=f"Sesudah Morfologi ({morph_op}, kernel={kernel_size}, iter={morph_iter})", use_column_width=True, clamp=True)
    
    st.markdown("---")
    st.subheader("5.4 Deteksi Kontur Sampah")
    contours, hierarchy = cv2.findContours(morph_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img_rgb.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    
    min_area = st.slider("Min Area Kontur (filter noise)", 10, 1000, 100,
                         help="Hanya menampilkan kontur dengan area di atas nilai ini")
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contour_img_filtered = img_rgb.copy()
    cv2.drawContours(contour_img_filtered, filtered_contours, -1, (0, 255, 0), 2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(contour_img, caption=f"Semua kontur ({len(contours)})", use_column_width=True)
    with col2:
        st.image(contour_img_filtered, caption=f"Kontur terfilter (area ≥ {min_area} px) = {len(filtered_contours)}", use_column_width=True)
    
    st.markdown("---")
    st.subheader("5.5 Perhitungan Volume / Estimasi Sampah")
    st.markdown("Menghitung luas area sampah yang terdeteksi.")
    
    if len(filtered_contours) > 0:
        hull_img = img_rgb.copy()
        total_area_px = 0
        for contour in filtered_contours:
            area = cv2.contourArea(contour)
            total_area_px += area
            hull = cv2.convexHull(contour)
            cv2.drawContours(hull_img, [hull], -1, (255, 0, 0), 2)
        
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        area_pct = (total_area_px / total_pixels) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Objek Terdeteksi", len(filtered_contours))
        col2.metric("Total Area Sampah (pixel²)", f"{int(total_area_px):,}")
        col3.metric("Persentase Coverage", f"{area_pct:.2f}%")
        
        with st.expander("Estimasi Area & Volume (Real World)"):
            st.markdown("Masukkan parameter kalibrasi untuk estimasi area dan volume di dunia nyata.")
            col_a, col_b = st.columns(2)
            with col_a:
                img_width_cm = st.number_input("Lebar citra di dunia nyata (cm)", 1.0, 1000.0, 100.0,
                                               help="Contoh: jika foto sungai mencakup area selebar 100 cm")
            with col_b:
                thickness = st.number_input("Estimasi ketebalan sampah (mm)", 0.1, 50.0, 1.0,
                                            help="Rata-rata ketebalan sampah plastik (default: 1 mm)")
            
            pixel_to_cm = w_orig / img_width_cm if img_width_cm > 0 else 0
            area_cm2 = total_area_px / (pixel_to_cm ** 2) if pixel_to_cm > 0 else 0
            volume_cm3 = area_cm2 * (thickness / 10)
            
            col_c, col_d, col_e = st.columns(3)
            col_c.metric("Kalibrasi (px/cm)", f"{pixel_to_cm:.2f}")
            col_d.metric("Estimasi Area", f"{area_cm2:.2f} cm²")
            col_e.metric("Estimasi Volume", f"{volume_cm3:.3f} cm³")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(contour_img_filtered, caption=f"Kontur sampah ({len(filtered_contours)} objek)", use_column_width=True)
        with col2:
            st.image(hull_img, caption="Convex Hull setiap objek", use_column_width=True)
    else:
        st.warning("Tidak ada kontur sampah terdeteksi. Coba sesuaikan parameter threshold, morfologi, atau min area kontur.")
    
    st.markdown("---")
    st.subheader("5.6 Histogram & Analisis Threshold")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
    ax_hist.hist(blur_gray.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
    ax_hist.axvline(x=cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0], 
                    color='red', linestyle='--', linewidth=2, label=f'Otsu Threshold = {cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]:.0f}')
    ax_hist.set_title("Histogram Intensitas Pixel dengan Threshold Otsu", fontsize=12, fontweight='bold')
    ax_hist.set_xlabel("Intensitas Pixel")
    ax_hist.set_ylabel("Frekuensi")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    st.pyplot(fig_hist)
    plt.close()
