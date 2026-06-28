import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Segmentasi & Perhitungan Volume Sampah Plastik Makro",
    page_icon="🌊",
    layout="wide"
)

st.markdown("""
<style>
    * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    
    .gradient-header {
        background: linear-gradient(135deg, #0d7377 0%, #14a3a8 50%, #0ea5e9 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .gradient-header h1 {
        color: white;
        font-weight: 700;
        font-size: 1.8rem;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .gradient-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
        border: 1px solid #f0f0f0;
    }
    .card-title {
        font-weight: 600;
        font-size: 1rem;
        color: #0d7377;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-header {
        font-weight: 600;
        font-size: 1.1rem;
        color: #1e293b;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .img-compare {
        display: flex;
        gap: 1rem;
        align-items: flex-start;
    }
    .img-compare-item {
        flex: 1;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .img-compare-item img {
        width: 100%;
        height: auto;
        display: block;
    }
    .img-compare-label {
        text-align: center;
        font-size: 0.8rem;
        font-weight: 500;
        color: #475569;
        padding: 0.3rem;
        background: #f8fafc;
        border-top: 1px solid #e2e8f0;
    }
    
    .pipeline {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 1.5rem 0;
    }
    .pipeline-step {
        background: white;
        border: 2px solid #0d7377;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        text-align: center;
        font-size: 0.85rem;
        font-weight: 500;
        color: #0d7377;
        min-width: 90px;
        flex: 1;
    }
    .pipeline-arrow {
        font-size: 1.3rem;
        color: #0d7377;
        font-weight: 300;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 0.75rem;
        margin: 1rem 0;
    }
    .metric-item {
        background: #f0fdf4;
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
        border-left: 3px solid #22c55e;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #166534;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #f8fafc;
        padding: 0.3rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748b;
        border: none;
        background: transparent;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0d7377;
        background: rgba(13, 115, 119, 0.08);
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #0d7377 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }
    
    div[data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e2e8f0;
    }
    div[data-testid="stSidebar"] .st-emotion-cache-1cypcdb {
        padding: 1.5rem 1rem;
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #94a3b8;
        font-size: 0.8rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }
    
    .badge {
        display: inline-block;
        background: #dbeafe;
        color: #1d4ed8;
        font-size: 0.7rem;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .control-group {
        background: #f8fafc;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

if 'img_rgb' not in st.session_state:
    st.session_state.img_rgb = None
if 'img_gray' not in st.session_state:
    st.session_state.img_gray = None
if 'img_bgr' not in st.session_state:
    st.session_state.img_bgr = None
if 'uploaded_name' not in st.session_state:
    st.session_state.uploaded_name = ''

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
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)
        return np.uint8(np.clip(mag, 0, 255))
    elif method == "Prewitt":
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        px = cv2.filter2D(gray.astype(np.float32), cv2.CV_64F, kx)
        py = cv2.filter2D(gray.astype(np.float32), cv2.CV_64F, ky)
        return np.uint8(np.clip(np.sqrt(px**2 + py**2), 0, 255))
    elif method == "Robert Cross":
        kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        rx = cv2.filter2D(gray.astype(np.float32), cv2.CV_64F, kx)
        ry = cv2.filter2D(gray.astype(np.float32), cv2.CV_64F, ky)
        return np.uint8(np.clip(np.sqrt(rx**2 + ry**2), 0, 255))
    elif method == "Laplacian":
        lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_64F)
        return np.uint8(np.clip(np.abs(lap), 0, 255))
    return None

def apply_morphology(binary, operation, kernel_size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if operation == "Erosi":
        return cv2.erode(binary, kernel, iterations=iterations)
    elif operation == "Dilasi":
        return cv2.dilate(binary, kernel, iterations=iterations)
    elif operation == "Opening":
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "Closing":
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return binary

def show_rgb_channels(img):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.patch.set_facecolor('white')
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original RGB", fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    for idx, (ch, color, title) in enumerate([
        (0, 'Reds', 'Red Channel'),
        (1, 'Greens', 'Green Channel'),
        (2, 'Blues', 'Blue Channel')
    ]):
        ax = axes[(idx+1)//2, (idx+1)%2]
        channel_img = np.zeros_like(img)
        channel_img[:, :, ch] = img[:, :, ch]
        ax.imshow(channel_img)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    return fig

def show_cmy_channels(img):
    c, m, y = convert_rgb_to_cmy(img)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.patch.set_facecolor('white')
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original RGB", fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    for idx, (ch_data, title) in enumerate(zip([c, m, y], ["Cyan", "Magenta", "Yellow"])):
        ax = axes[(idx+1)//2, (idx+1)%2]
        im = ax.imshow(ch_data, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"{title} Channel", fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1rem 0;border-bottom:1px solid #e2e8f0;margin-bottom:1rem;">
        <div style="font-size:2rem;margin-bottom:0.3rem;">🌊</div>
        <div style="font-weight:700;font-size:1rem;color:#0d7377;">EcoWatch River</div>
        <div style="font-size:0.7rem;color:#94a3b8;">Segmentasi Sampah Plastik</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Citra Sungai",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Format: JPG, PNG, BMP"
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Gagal membaca file. Coba upload file lain.")
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            st.session_state.img_rgb = img_rgb
            st.session_state.img_gray = img_gray
            st.session_state.img_bgr = img_bgr
            st.session_state.uploaded_name = uploaded_file.name

            st.markdown(f"""
        <div style="background:#f0fdf4;padding:0.8rem;border-radius:10px;border-left:3px solid #22c55e;margin-bottom:1rem;">
            <div style="font-size:0.8rem;font-weight:600;color:#166534;">✓ {uploaded_file.name}</div>
            <div style="font-size:0.7rem;color:#64748b;">{img_rgb.shape[1]}×{img_rgb.shape[0]} px</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:1rem 0;border-color:#e2e8f0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem;color:#64748b;">
        <b style="color:#1e293b;">Info Project</b><br>
        UAS Pengolahan Citra Digital<br>
        Metode: Otsu + Morfologi<br>
        <span style="color:#94a3b8;">© 2026</span>
    </div>
    """, unsafe_allow_html=True)

# ===== MAIN =====
if st.session_state.img_rgb is None:
    st.markdown("""
    <div class="gradient-header">
        <h1>🌊 Segmentasi & Perhitungan Volume Sampah Plastik Makro</h1>
        <p>Pada Citra Sungai Menggunakan Metode Otsu Thresholding dan Operasi Morfologi</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="card-title">📋 Tentang Aplikasi</div>
        <p style="color:#475569;line-height:1.6;font-size:0.9rem;">
        Aplikasi ini menerapkan konsep <b>Pengolahan Citra Digital</b> untuk 
        melakukan <b>segmentasi</b> dan <b>estimasi volume</b> sampah plastik makro 
        pada citra sungai secara otomatis.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("<div class='pipeline'>", unsafe_allow_html=True)
        steps = [
            ("📷", "Input\nCitra"), ("🎯", "Pre-\nprocessing"), ("⚡", "Otsu\nThreshold"),
            ("🔧", "Operasi\nMorfologi"), ("📊", "Deteksi\nKontur"), ("📐", "Estimasi\nVolume")
        ]
        for i, (icon, label) in enumerate(steps):
            st.markdown(f"<div class='pipeline-step'>{icon}<br>{label}</div>", unsafe_allow_html=True)
            if i < len(steps) - 1:
                st.markdown("<div class='pipeline-arrow'>→</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-top:0.8rem;">
            <span class='badge'>Representasi Citra</span>
            <span class='badge'>Digitalisasi Citra</span>
            <span class='badge'>Operasi Geometri</span>
            <span class='badge'>Deteksi Tepi</span>
            <span class='badge'>Segmentasi</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#f8fafc;border-radius:10px;padding:1.2rem;text-align:center;">
            <div style="font-size:3rem;margin-bottom:0.5rem;">🌊</div>
            <div style="font-weight:600;color:#1e293b;font-size:0.9rem;">Upload Citra Sungai</div>
            <div style="color:#94a3b8;font-size:0.8rem;margin:0.3rem 0 0.8rem 0;">Melalui sidebar untuk memulai</div>
            <div style="display:flex;justify-content:center;gap:0.5rem;">
                <span class='badge'>JPG</span>
                <span class='badge'>PNG</span>
                <span class='badge'>BMP</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='footer'>Universitas — UAS Pengolahan Citra Digital 2026</div>
    """, unsafe_allow_html=True)
    st.stop()

img_rgb = st.session_state.img_rgb
img_gray = st.session_state.img_gray
img_bgr = st.session_state.img_bgr
h_orig, w_orig = img_rgb.shape[:2]

st.markdown(f"""
<div class="gradient-header">
    <h1>🌊 Segmentasi & Perhitungan Volume Sampah Plastik Makro</h1>
    <p>{w_orig}×{h_orig} px | File: {st.session_state.uploaded_name or '—'}</p>
</div>
""", unsafe_allow_html=True)

tab1_label, tab2_label, tab3_label, tab4_label, tab5_label = st.tabs([
    "🖼️ Representasi", "💻 Digitalisasi", "🔧 Geometri", "✏️ Deteksi Tepi", "🎯 Segmentasi & Volume"
])

err = st.container()

# ===== TAB 1: REPRESENTASI =====
with tab1_label:
    try:
        st.markdown("<div class='card'><div class='card-title'>📊 Informasi Citra</div>", unsafe_allow_html=True)
        cols = st.columns(4)
        cols[0].metric("Dimensi", f"{w_orig} × {h_orig}")
        cols[1].metric("Total Pixel", f"{w_orig * h_orig:,}")
        cols[2].metric("Channel", "3 (RGB)")
        cols[3].metric("Tipe Data", "uint8 (0–255)")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='card-title'>🎨 Channel RGB</div>", unsafe_allow_html=True)
        st.pyplot(show_rgb_channels(img_rgb))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='card-title'>⬜ Konversi RGB → Grayscale</div>", unsafe_allow_html=True)
        st.markdown("Rumus: **Y = 0.299 R + 0.587 G + 0.114 B**", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.image(img_rgb, caption="Original RGB", use_column_width=True)
        col2.image(img_gray, caption="Hasil Grayscale", use_column_width=True, clamp=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='card-title'>🟦 Model Warna CMY</div>", unsafe_allow_html=True)
        st.markdown("Rumus: **C = 1 − R, M = 1 − G, Y = 1 − B** (0–1)", unsafe_allow_html=True)
        st.pyplot(show_cmy_channels(img_rgb))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='card-title'>🔢 Matriks Pixel</div>", unsafe_allow_html=True)
        rh, rw = img_gray.shape
        step = max(1, rh // 20, rw // 20)
        display_matrix = img_gray[::step, ::step]
        st.info(f"Sampling setiap {step} pixel ({display_matrix.shape[0]}×{display_matrix.shape[1]})")
        st.dataframe(display_matrix, use_container_width=True, height=300)
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        import traceback
        err.error(f"Tab 1 Error: {e}")
        err.code(traceback.format_exc())

# ===== TAB 2: DIGITALISASI =====
with tab2_label:
    try:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📐 Sampling – Resolusi Gambar</div>", unsafe_allow_html=True)
        samp_factor = st.slider("Faktor Downsampling", 1, 10, 1, help="1 = resolusi asli, makin besar makin rendah")
        if samp_factor > 1:
            sampled_rgb = downsample_image(img_rgb, samp_factor)
            st.info(f"Resolusi: {w_orig}×{h_orig} → {sampled_rgb.shape[1]}×{sampled_rgb.shape[0]} px")
            st.image(sampled_rgb, caption=f"Sampling faktor {samp_factor}", use_column_width=True)
        else:
            st.image(img_rgb, caption="Resolusi asli", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🎨 Kuantisasi – Level Warna</div>", unsafe_allow_html=True)
        q_bits = st.slider("Bit per pixel", 1, 8, 8, help="8 bit = 256 level, 1 bit = 2 level (biner)")
        q_img = quantize_image(img_gray, q_bits)
        st.info(f"{q_bits} bit → {2**q_bits} level warna")
        col1, col2 = st.columns(2)
        col1.image(img_gray, caption=f"Asli (8 bit, 256 level)", use_column_width=True, clamp=True)
        col2.image(q_img, caption=f"Kuantisasi ({q_bits} bit, {2**q_bits} level)", use_column_width=True, clamp=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>💾 Format Citra</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        cols[0].info("**JPG/JPEG**\n\nKompresi lossy\nUkuran kecil\nCocok untuk foto")
        cols[1].info("**PNG**\n\nKompresi lossless\nTransparansi\nKualitas tinggi")
        cols[2].info("**BMP**\n\nTanpa kompresi\nUkuran besar\nKualitas sempurna")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🔗 Hubungan Antar Pixel (Neighbourhood)</div>", unsafe_allow_html=True)
        neigh_mode = st.radio("Mode", ["4-Neighbour", "8-Neighbour"], horizontal=True)
        st.markdown(f"""
        **{neigh_mode}:** {('N₄(p) = pixel di atas, bawah, kiri, kanan' if neigh_mode == '4-Neighbour' else 'N₈(p) = seluruh pixel tetangga termasuk diagonal')}
        """)
        fig_n, ax_n = plt.subplots(figsize=(6, 5))
        fig_n.patch.set_facecolor('white')
        ax_n.imshow(img_gray, cmap='gray')
        ax_n.set_title("Klik pada pixel: perhatikan koordinat X (kolom), Y (baris)", fontsize=9)
        ax_n.set_xlabel("X")
        ax_n.set_ylabel("Y")
        ax_n.grid(True, alpha=0.3)
        st.pyplot(fig_n)
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        import traceback
        err.error(f"Tab 2 Error: {e}")
        err.code(traceback.format_exc())

# ===== TAB 3: GEOMETRI =====
with tab3_label:
    try:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>⬜ 3.1 Konversi Grayscale</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.image(img_rgb, caption="Original RGB", use_column_width=True)
        col2.image(img_gray, caption="Grayscale", use_column_width=True, clamp=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🔄 3.2 Rotasi</div>", unsafe_allow_html=True)
        rot_angle = st.slider("Sudut rotasi (°)", -180, 180, 0)
        if rot_angle != 0:
            center = (w_orig // 2, h_orig // 2)
            M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
            rotated = cv2.warpAffine(img_rgb, M, (w_orig, h_orig), borderValue=(255, 255, 255))
            st.image(rotated, caption=f"Rotasi {rot_angle}°", use_column_width=True)
        else:
            st.image(img_rgb, caption="Rotasi 0° (asli)", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🪞 3.3 Flipping</div>", unsafe_allow_html=True)
        flip_choice = st.selectbox("Arah Flip", ["Tidak Ada", "Horizontal", "Vertikal", "Horizontal & Vertikal"])
        flip_map = {"Tidak Ada": None, "Horizontal": 1, "Vertikal": 0, "Horizontal & Vertikal": -1}
        flip_code = flip_map[flip_choice]
        if flip_code is not None:
            flipped = cv2.flip(img_rgb, flip_code)
            st.image(flipped, caption=f"Flip {flip_choice}", use_column_width=True)
        else:
            st.image(img_rgb, caption="Asli (tanpa flip)", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>✂️ 3.4 Cropping</div>", unsafe_allow_html=True)
        crop_enabled = st.checkbox("Aktifkan cropping", value=False)
        if crop_enabled:
            c1, c2, c3, c4 = st.columns(4)
            cx = c1.number_input("X", 0, w_orig - 1, 0)
            cy = c2.number_input("Y", 0, h_orig - 1, 0)
            cw = c3.number_input("Lebar", 1, w_orig, min(300, w_orig))
            ch = c4.number_input("Tinggi", 1, h_orig, min(300, h_orig))
            x1, y1 = min(cx, w_orig - 1), min(cy, h_orig - 1)
            x2, y2 = min(x1 + cw, w_orig), min(y1 + ch, h_orig)
            if x2 > x1 and y2 > y1:
                st.image(img_rgb[y1:y2, x1:x2], caption=f"Crop ({x1},{y1}) → ({x2},{y2})", use_column_width=True)
        else:
            st.info("Centang checkbox untuk mengaktifkan cropping")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📏 3.5 Scaling (Resize)</div>", unsafe_allow_html=True)
        scale_method = st.selectbox("Metode", ["Persentase", "Ukuran Tetap"], key="scale_method")
        if scale_method == "Persentase":
            pct = st.slider("Skala (%)", 10, 200, 100)
            if pct != 100:
                nw = int(w_orig * pct / 100)
                nh = int(h_orig * pct / 100)
                scaled = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA if pct < 100 else cv2.INTER_LINEAR)
                st.image(scaled, caption=f"{pct}% → {nw}×{nh}", use_column_width=True)
            else:
                st.image(img_rgb, caption="100% (asli)", use_column_width=True)
        else:
            c1, c2 = st.columns(2)
            fw = c1.number_input("Lebar (px)", 1, 2000, w_orig)
            fh = c2.number_input("Tinggi (px)", 1, 2000, h_orig)
            if fw != w_orig or fh != h_orig:
                st.image(cv2.resize(img_rgb, (fw, fh), interpolation=cv2.INTER_AREA), caption=f"{fw}×{fh}", use_column_width=True)
            else:
                st.image(img_rgb, caption="Ukuran asli", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🌓 3.6 Negasi (Invers Warna)</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.image(img_rgb, caption="Original", use_column_width=True)
        col2.image(cv2.bitwise_not(img_rgb), caption="Negasi", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        import traceback
        err.error(f"Tab 3 Error: {e}")
        err.code(traceback.format_exc())

# ===== TAB 4: DETEKSI TEPI =====
with tab4_label:
    try:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>✏️ Deteksi Tepi</div>", unsafe_allow_html=True)
        st.markdown("Menemukan batas objek dalam citra menggunakan operator gradien.")

        edge_method = st.selectbox("Metode", ["Sobel", "Prewitt", "Robert Cross", "Laplacian"])
        thresh_edge = st.slider("Threshold", 0, 255, 30, help="Makin rendah, makin banyak tepi terdeteksi")
        blur_k = st.slider("Gaussian Blur", 1, 15, 3, step=2, help="Perhalus citra sebelum deteksi")

        pre_gray = cv2.GaussianBlur(img_gray, (blur_k, blur_k), 0) if blur_k > 1 else img_gray.copy()
        edges = detect_edges(pre_gray, edge_method)

        if edges is not None:
            _, edges_bin = cv2.threshold(edges, thresh_edge, 255, cv2.THRESH_BINARY)
            col1, col2, col3 = st.columns(3)
            col1.image(img_rgb, caption="Original", use_column_width=True)
            col2.image(edges, caption=f"{edge_method} (gradien)", use_column_width=True, clamp=True)
            col3.image(edges_bin, caption=f"{edge_method} (threshold={thresh_edge})", use_column_width=True, clamp=True)
        st.markdown("</div>", unsafe_allow_html=True)

        edge_info = {
            "Sobel": """
            **Sobel** — Konvolusi kernel 3×3 untuk gradien horizontal ($G_x$) dan vertikal ($G_y$).
            $$\\text{Magnitude} = \\sqrt{G_x^2 + G_y^2}$$
            $$G_x = \\begin{bmatrix} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1 \\end{bmatrix} \\quad 
            G_y = \\begin{bmatrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 2 & 1 \\end{bmatrix}$$
            """,
            "Prewitt": """
            **Prewitt** — Mirip Sobel dengan bobot seragam (1) untuk semua tetangga.
            $$G_x = \\begin{bmatrix} -1 & 0 & 1 \\\\ -1 & 0 & 1 \\\\ -1 & 0 & 1 \\end{bmatrix} \\quad 
            G_y = \\begin{bmatrix} -1 & -1 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 1 & 1 \\end{bmatrix}$$
            """,
            "Robert Cross": """
            **Robert Cross** — Kernel 2×2 sederhana, sensitif terhadap tepi diagonal.
            $$G_x = \\begin{bmatrix} 1 & 0 \\\\ 0 & -1 \\end{bmatrix} \\quad 
            G_y = \\begin{bmatrix} 0 & 1 \\\\ -1 & 0 \\end{bmatrix}$$
            """,
            "Laplacian": """
            **Laplacian** — Detektor orde-2 menggunakan turunan kedua, sensitif terhadap noise.
            $$\\nabla^2 f = \\frac{\\partial^2 f}{\\partial x^2} + \\frac{\\partial^2 f}{\\partial y^2}$$
            Kernel: $$\\begin{bmatrix} 0 & 1 & 0 \\\\ 1 & -4 & 1 \\\\ 0 & 1 & 0 \\end{bmatrix}$$
            """
        }
        with st.expander(f"📖 Teori Metode {edge_method}"):
            st.markdown(edge_info.get(edge_method, ""))
    except Exception as e:
        import traceback
        err.error(f"Tab 4 Error: {e}")
        err.code(traceback.format_exc())

# ===== TAB 5: SEGMENTASI & VOLUME =====
with tab5_label:
    try:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🎯 Pipeline Segmentasi</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='pipeline'>
            <div class='pipeline-step'>📷 Input</div><div class='pipeline-arrow'>→</div>
            <div class='pipeline-step'>🌀 Gaussian<br>Blur</div><div class='pipeline-arrow'>→</div>
            <div class='pipeline-step'>⚡ Otsu<br>Threshold</div><div class='pipeline-arrow'>→</div>
            <div class='pipeline-step'>🔧 Morfologi</div><div class='pipeline-arrow'>→</div>
            <div class='pipeline-step'>📊 Deteksi<br>Kontur</div><div class='pipeline-arrow'>→</div>
            <div class='pipeline-step'>📐 Estimasi<br>Volume</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>🌀 Preprocessing</div>", unsafe_allow_html=True)
            seg_blur = st.slider("Kernel Gaussian Blur", 1, 21, 5, step=2)
            blur_gray = cv2.GaussianBlur(img_gray, (seg_blur, seg_blur), 0)
            st.image(blur_gray, caption=f"Gaussian Blur (kernel={seg_blur})", use_column_width=True, clamp=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>⚡ Otsu Thresholding</div>", unsafe_allow_html=True)
            _, otsu_binary = cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            st.image(otsu_binary, caption="Otsu Threshold (otomatis)", use_column_width=True, clamp=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🔧 Operasi Morfologi</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        morph_op = c1.selectbox("Operasi", ["Erosi", "Dilasi", "Opening", "Closing"])
        morph_k = c2.slider("Kernel", 3, 21, 5, step=2)
        morph_iter = c3.slider("Iterasi", 1, 10, 1)
        morph_result = apply_morphology(otsu_binary, morph_op, morph_k, morph_iter)
        col1, col2 = st.columns(2)
        col1.image(otsu_binary, caption="Sebelum morfologi", use_column_width=True, clamp=True)
        col2.image(morph_result, caption=f"Sesudah {morph_op} (kernel={morph_k}, iter={morph_iter})", use_column_width=True, clamp=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📊 Deteksi Kontur & Estimasi Volume</div>", unsafe_allow_html=True)
        contours, _ = cv2.findContours(morph_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = st.slider("Min area kontur (filter noise)", 10, 1000, 100)
        filtered = [c for c in contours if cv2.contourArea(c) >= min_area]

        contour_all = img_rgb.copy()
        cv2.drawContours(contour_all, contours, -1, (0, 255, 0), 2)
        contour_filt = img_rgb.copy()
        cv2.drawContours(contour_filt, filtered, -1, (0, 255, 0), 2)

        col1, col2 = st.columns(2)
        col1.image(contour_all, caption=f"Semua kontur ({len(contours)})", use_column_width=True)
        col2.image(contour_filt, caption=f"Filtered (≥{min_area} px) = {len(filtered)}", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if len(filtered) > 0:
            hull_img = img_rgb.copy()
            total_area = 0
            for c in filtered:
                total_area += cv2.contourArea(c)
                hull = cv2.convexHull(c)
                cv2.drawContours(hull_img, [hull], -1, (255, 0, 0), 2)

            pct = (total_area / (img_rgb.shape[0] * img_rgb.shape[1])) * 100

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>📐 Hasil Perhitungan</div>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div class='metric-item'><div class='metric-value'>{len(filtered)}</div><div class='metric-label'>Objek Terdeteksi</div></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-item'><div class='metric-value'>{int(total_area):,}</div><div class='metric-label'>Area (pixel²)</div></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-item'><div class='metric-value'>{pct:.2f}%</div><div class='metric-label'>Coverage</div></div>", unsafe_allow_html=True)

            with st.expander("📏 Estimasi Area & Volume (Real World)"):
                c_a, c_b = st.columns(2)
                img_w_cm = c_a.number_input("Lebar citra nyata (cm)", 1.0, 1000.0, 100.0)
                thickness = c_b.number_input("Ketebalan sampah (mm)", 0.1, 50.0, 1.0)
                px_per_cm = w_orig / img_w_cm if img_w_cm > 0 else 0
                area_cm2 = total_area / (px_per_cm ** 2) if px_per_cm > 0 else 0
                vol_cm3 = area_cm2 * (thickness / 10)

                x1, x2, x3 = st.columns(3)
                x1.markdown(f"<div class='metric-item'><div class='metric-value'>{px_per_cm:.1f}</div><div class='metric-label'>Kalibrasi (px/cm)</div></div>", unsafe_allow_html=True)
                x2.markdown(f"<div class='metric-item'><div class='metric-value'>{area_cm2:.1f}</div><div class='metric-label'>Area (cm²)</div></div>", unsafe_allow_html=True)
                x3.markdown(f"<div class='metric-item'><div class='metric-value'>{vol_cm3:.2f}</div><div class='metric-label'>Volume (cm³)</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>📌 Visualisasi Kontur & Convex Hull</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.image(contour_filt, caption=f"Kontur ({len(filtered)} objek)", use_column_width=True)
            col2.image(hull_img, caption="Convex Hull", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Tidak ada kontur terdeteksi. Sesuaikan parameter threshold, morfologi, atau min area.")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📈 Histogram & Threshold Otsu</div>", unsafe_allow_html=True)
        otsu_val = cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        fig_h, ax_h = plt.subplots(figsize=(10, 3.5))
        fig_h.patch.set_facecolor('white')
        ax_h.hist(blur_gray.ravel(), bins=256, range=[0, 256], color='#94a3b8', alpha=0.7)
        ax_h.axvline(otsu_val, color='#ef4444', linestyle='--', linewidth=2, label=f'Otsu = {otsu_val:.0f}')
        ax_h.set_title("Histogram Intensitas", fontsize=11, fontweight='bold')
        ax_h.set_xlabel("Intensitas Pixel")
        ax_h.set_ylabel("Frekuensi")
        ax_h.legend()
        ax_h.grid(True, alpha=0.2)
        st.pyplot(fig_h)
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        import traceback
        err.error(f"Tab 5 Error: {e}")
        err.code(traceback.format_exc())

st.markdown("""
<div class='footer'>
    UAS Pengolahan Citra Digital — Segmentasi & Perhitungan Volume Sampah Plastik Makro<br>
    Metode Otsu Thresholding dan Operasi Morfologi
</div>
""", unsafe_allow_html=True)
