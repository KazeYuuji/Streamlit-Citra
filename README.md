# PENGOLAHAN CITRA DIGITAL
## Segmentasi & Perhitungan Volume Sampah Plastik Makro pada Citra Sungai
### Metode: Otsu Thresholding dan Operasi Morfologi

---

## Deskripsi Proyek

Proyek UAS ini mengimplementasikan **satu aplikasi pengolahan citra digital** yang mencakup seluruh materi yang telah dipelajari:

1. **Representasi Citra** — Matriks piksel, model warna RGB, konversi grayscale, model CMY
2. **Digitalisasi Citra** — Sampling (resolusi), kuantisasi (level warna), format citra, hubungan antar piksel
3. **Operasi Aritmatika & Geometri Citra** — Rotasi, flipping, cropping, scaling, negasi
4. **Deteksi Tepi** — Sobel, Prewitt, Robert Cross, Laplacian
5. **Segmentasi Citra** — Otsu Thresholding + Operasi Morfologi untuk deteksi plastik

Fokus aplikasi: mendeteksi, mengkuantifikasi, dan memvisualisasikan sampah plastik makro pada citra sungai.

---

## Fitur Aplikasi (Minimal)

| Fitur | Status |
|-------|--------|
| Load image | Ada |
| Konversi grayscale | Ada |
| Operasi geometri (rotasi, flipping, cropping, scaling, negasi) | Ada (>1) |
| Deteksi tepi (Sobel, Prewitt, Robert Cross, Laplacian) | Ada (>1) |
| Segmentasi sederhana (Otsu + Morfologi) | Ada |
| Menampilkan hasil output | Ada (gambar + laporan) |

---

## Struktur Project

```
PNS/
├── README.md
├── requirements.txt
├── data/                       # Folder citra input
│   └── river_plastic_01.jpg    # Contoh citra sungai
├── output/                     # Semua hasil output
│   ├── 01_representasi/        # RGB, grayscale, CMY, pixel matrix
│   ├── 02_digitalisasi/        # Sampling, kuantisasi, neighbourhood
│   ├── 03_geometri/            # Rotasi, flip, crop, scale, negasi
│   ├── 04_edge_detection/      # Sobel, Prewitt, Robert, Laplacian
│   ├── 05_segmentasi/          # Mask, overlay, grid, volume
│   └── laporan.txt             # Laporan lengkap
└── src/
    ├── __init__.py
    ├── main.py                 # Entry point pipeline lengkap
    ├── representasi.py         # Representasi citra (RGB, CMY, grayscale)
    ├── digitalisasi.py         # Digitalisasi (sampling, kuantisasi, neighbours)
    ├── geometri.py             # Operasi geometri (rotate, flip, crop, scale, negate)
    ├── deteksi_tepi.py         # Deteksi tepi (Sobel, Prewitt, Robert, Laplacian)
    ├── segmentasi.py           # Segmentasi Otsu + morfologi
    ├── volume_calc.py          # Estimasi volume & massa
    └── visualisasi.py          # Visualisasi & laporan
```

---

## Alur Kerja Pipeline

```
Citra Input (RGB)
       │
       ▼
┌─────────────────────────────────────────────────┐
│ 1. REPRESENTASI CITRA                            │
│    • Matriks piksel (print sebagian)             │
│    • Pemisahan channel R, G, B                   │
│    • Konversi Grayscale                          │
│    • Konversi CMY (Cyan, Magenta, Yellow)        │
│    • Informasi piksel pada koordinat tertentu    │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│ 2. DIGITALISASI CITRA                            │
│    • Sampling: 100%, 50%, 25%, 12.5%             │
│    • Kuantisasi: 8-bit, 6-bit, 4-bit, 2-bit, 1-bit│
│    • Analisis 4-neighbour & 8-neighbour           │
│    • Informasi format file                        │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│ 3. OPERASI GEOMETRI                              │
│    • Rotasi 45°                                  │
│    • Flipping horizontal & vertikal              │
│    • Cropping 1/4 tengah                         │
│    • Scaling 1.5x                                │
│    • Negasi (invers warna)                       │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│ 4. DETEKSI TEPI                                  │
│    • Sobel (grad_x, grad_y, magnitude)           │
│    • Prewitt                                     │
│    • Robert Cross                                │
│    • Laplacian of Gaussian                       │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│ 5. SEGMENTASI (Otsu + Morfologi)                 │
│    • Grayscale → Gaussian Blur → Otsu Threshold  │
│    • Morphological Closing & Opening             │
│    • Remove Small Objects                        │
│    • Fill Holes                                  │
│    • Estimasi Volume & Massa                     │
└─────────────────────────────────────────────────┘
       │
       ▼
    Output: Gambar + Laporan
```

---

## Penjelasan Konsep per Modul

### 1. Representasi Citra (`src/representasi.py`)

- **Citra sebagai matriks piksel**: Dicetak sebagian kecil (10x10 piksel) ke file teks untuk menunjukkan bahwa citra adalah array 2D nilai intensitas.
- **Model warna RGB**: Channel Red, Green, Blue dipisahkan dan divisualisasikan secara terpisah.
- **Konversi Grayscale**: Menggunakan `cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)`.
- **Model warna CMY**: Dihitung sebagai `1.0 - RGB` (dalam float ternormalisasi).
- **Hubungan warna dan piksel**: Informasi nilai RGB pada koordinat pusat citra ditampilkan.

### 2. Digitalisasi Citra (`src/digitalisasi.py`)

- **Sampling**: Citra di-resize ke berbagai skala (100%, 50%, 25%, 12.5%) untuk menunjukkan efek resolusi.
- **Kuantisasi**: Level abu-abu dikurangi dari 8-bit (256 level) hingga 1-bit (2 level) untuk menunjukkan efek jumlah warna.
- **Format citra**: Informasi format file input (JPG, dimensi, ukuran) ditampilkan.
- **Hubungan antar piksel**: Analisis 4-neighbourhood (atas, bawah, kiri, kanan) dan 8-neighbourhood (termasuk diagonal) divisualisasikan untuk satu piksel pusat.

### 3. Operasi Aritmatika & Geometri Citra (`src/geometri.py`)

| Operasi | Implementasi |
|---------|-------------|
| Rotasi | `cv2.warpAffine` dengan rotasi 45° terhadap pusat |
| Flipping horizontal | `cv2.flip(image, 1)` — membalik kiri-kanan |
| Flipping vertikal | `cv2.flip(image, 0)` — membalik atas-bawah |
| Cropping | Memotong 1/4 bagian tengah citra |
| Scaling | `cv2.resize` dengan faktor 1.5x (INTER_CUBIC) |
| Negasi | `255 - image` — invers warna |

### 4. Deteksi Tepi (`src/deteksi_tepi.py`)

Empat metode deteksi tepi diimplementasikan:

| Metode | Kernel / Operator |
|--------|------------------|
| Sobel | `cv2.Sobel` dengan ksize=3 (Gx, Gy, magnitude) |
| Prewitt | Kernel 3x3 buatan ([−1,0,1], [−1,0,1], [−1,0,1]) |
| Robert Cross | Kernel 2x2 buatan ([1,0;0,−1] dan [0,1;−1,0]) |
| Laplacian | `cv2.Laplacian` dengan ksize=3 |

Semua metode menghitung magnitude gradien: `sqrt(Gx² + Gy²)`.

### 5. Segmentasi Citra (`src/segmentasi.py`)

Pipeline segmentasi lengkap:
1. **Grayscale** → konversi ke 1 channel
2. **Gaussian Blur** (5x5) → menghaluskan noise
3. **Otsu Thresholding** (THRESH_BINARY) → threshold otomatis
4. **Morphological Closing** → menutup lubang kecil dalam objek
5. **Morphological Opening** → menghilangkan noise kecil
6. **Remove Small Objects** → buang objek < area minimum
7. **Fill Holes** → isi rongga dalam objek plastik

Hasil: mask biner, overlay, estimasi luas, volume, dan massa.

---

## Cara Penggunaan

### 1. Instalasi Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Pipeline Lengkap
```bash
python src/main.py --image data/river_plastic_01.jpg --output output
```

### Parameter Opsional:
| Argumen | Default | Deskripsi |
|---------|---------|-----------|
| `--kernel` | 5 | Ukuran kernel morfologi |
| `--min-area` | 500 | Area minimum objek (piksel) |
| `--px-ratio` | 0.05 | Rasio cm per piksel |
| `--thickness` | 0.3 | Tebal rata-rata plastik (cm) |
| `--no-display` | False | Sembunyikan plot interaktif |

### 3. Contoh Lengkap
```bash
python src/main.py -i data/river_plastic_01.jpg -o hasil_analisis --px-ratio 0.08 --thickness 0.5
```

---

## Output

### Struktur Folder Output
```
output/
├── 01_representasi/
│   ├── original.png
│   ├── red_channel.png
│   ├── green_channel.png
│   ├── blue_channel.png
│   ├── grayscale.png
│   ├── cmy_cyan.png
│   ├── cmy_magenta.png
│   ├── cmy_yellow.png
│   ├── rgb_channels.png         (grid figure)
│   ├── grayscale_cmy.png        (grid figure)
│   └── pixel_matrix.txt         (nilai piksel 10x10)
├── 02_digitalisasi/
│   ├── sampling_1.0.png
│   ├── sampling_0.5.png
│   ├── sampling_0.25.png
│   ├── sampling_0.125.png
│   ├── sampling_grid.png        (grid figure)
│   ├── quantization_8bit.png
│   ├── quantization_6bit.png
│   ├── quantization_4bit.png
│   ├── quantization_2bit.png
│   ├── quantization_1bit.png
│   ├── quantization_grid.png    (grid figure)
│   └── neighbourhood.png        (visualisasi neighbours)
├── 03_geometri/
│   ├── rotate.png
│   ├── flip_h.png
│   ├── flip_v.png
│   ├── crop.png
│   ├── scale.png
│   ├── negate.png
│   └── geometric_grid.png       (grid figure)
├── 04_edge_detection/
│   ├── sobel.png
│   ├── sobel_grad_x.png
│   ├── sobel_grad_y.png
│   ├── prewitt.png
│   ├── robert.png
│   ├── laplacian.png
│   ├── laplacian_detail.png
│   └── edge_detection_grid.png  (grid figure)
├── 05_segmentasi/
│   ├── 01_segmentation_grid.png
│   ├── 02_overlay.png
│   ├── 03_cleaned_mask.png
│   └── 04_overlay_image.png
└── laporan.txt                  (laporan teknis lengkap)
```

### Laporan Konsol
Pipeline mencetak laporan ke konsol yang mencakup:
- Informasi representasi citra (dimensi, channel, statistik)
- Detail digitalisasi (sampling, kuantisasi)
- Operasi geometri yang diterapkan
- Metode deteksi tepi yang digunakan
- Hasil segmentasi (threshold, coverage, volume, massa)

---

## Referensi

- Otsu, N. (1979). *A threshold selection method from gray-level histograms*. IEEE Trans. Sys., Man., Cyber.
- Gonzalez, R. C. & Woods, R. E. *Digital Image Processing*. 4th Ed., Pearson.
- OpenCV Documentation: https://docs.opencv.org/
- scikit-image: https://scikit-image.org/

---

**Dibuat untuk tugas UAS Pengolahan Citra Digital**
Program Studi Ilmu Komputer / Informatika
