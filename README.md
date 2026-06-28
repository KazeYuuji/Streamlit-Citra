# 🌊 Segmentasi & Perhitungan Volume Sampah Plastik Makro pada Citra Sungai

**Aplikasi Streamlit untuk Pengolahan Citra Digital** — UAS Project

Menggunakan Metode **Otsu Thresholding** dan **Operasi Morfologi** untuk mendeteksi,
mensegmentasi, dan memperkirakan volume sampah plastik makro pada citra sungai.

---

## Fitur Aplikasi

| Modul | Deskripsi | Konsep PCD |
|---|---|---|
| **📊 Representasi Citra** | Matriks pixel, kanal RGB, model CMY, histogram, konversi grayscale | Representasi Citra |
| **🔢 Digitalisasi Citra** | Sampling (resolusi), kuantisasi (bit depth), format citra, tetangga pixel 4/8-N | Digitalisasi Citra |
| **🔄 Operasi Geometri** | Grayscale, Rotasi, Flipping, Cropping, Scaling, Negasi | Operasi Geometri |
| **✏️ Deteksi Tepi** | Sobel, Prewitt, Robert Cross, Laplacian + threshold | Deteksi Tepi |
| **♻️ Segmentasi & Volume** | Otsu thresholding + morfologi (erosion, dilation, opening, closing, gradient) + overlay + estimasi volume | Segmentasi Citra |

## Ruang Lingkup Materi yang Dicakup

### 1. Representasi Citra
- ✅ Citra sebagai matriks pixel (tampilan area 16×16)
- ✅ Model warna RGB (kanal R, G, B terpisah + histogram)
- ✅ Konversi RGB ke Grayscale (rumus luminance)
- ✅ Model warna CMY (Cyan, Magenta, Yellow)
- ✅ Hubungan warna dan pixel (histogram)

### 2. Digitalisasi Citra
- ✅ Sampling (resolusi gambar dengan slider faktor skala)
- ✅ Kuantisasi (bit depth 1–8 bit, level warna)
- ✅ Format citra (info format JPG/PNG/BMP)
- ✅ Hubungan antar pixel (4-neighbour dan 8-neighbour dengan visualisasi grid)

### 3. Operasi Aritmatika & Geometri Citra (7 operasi)
- ✅ Grayscale
- ✅ Rotasi (derajat bebas)
- ✅ Flipping (horizontal, vertikal, keduanya)
- ✅ Cropping (koordinat bebas)
- ✅ Scaling / Resize (dengan pilihan interpolasi)
- ✅ Negasi (invers warna)

### 4. Deteksi Tepi (4 metode)
- ✅ Sobel
- ✅ Prewitt
- ✅ Robert Cross
- ✅ Laplacian
- ✅ Threshold adjustable + perbandingan metode

### 5. Segmentasi Citra
- ✅ **Otsu Thresholding** (threshold otomatis)
- ✅ **Operasi Morfologi**: Erosion, Dilation, Opening, Closing, Gradient
- ✅ Perhitungan luas area sampah (cm²)
- ✅ Estimasi volume sampah (ml)
- ✅ Overlay segmentasi pada citra asli
- ✅ Download hasil segmentasi (PNG)

## Tech Stack

- **Python 3.9+**
- **Streamlit** — UI Framework
- **OpenCV** — Image Processing
- **NumPy** — Matrix Operations
- **Pillow** — Image I/O
- **Matplotlib** — Visualization
- **scikit-image** — Utilities

## Cara Menjalankan

### Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy ke Streamlit Cloud

1. Push repository ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Pilih repository ini
4. Set `app.py` sebagai entry point
5. Deploy!

## Struktur File

```
streamlit-citra/
├── app.py                 # Aplikasi utama Streamlit
├── requirements.txt       # Dependensi Python
├── README.md              # Dokumentasi
└── .streamlit/
    └── config.toml        # Konfigurasi tema Streamlit
```

## Video Demo

> *(Tambahkan link video demo di sini)*

## Screenshot Output

> *(Tambahkan screenshot hasil output di sini)*

---

**Dibuat untuk UAS Project Pengolahan Citra Digital**
