# 49_IF-E_titut-titut

<div align="center">

# ANALISIS DAN ESTIMASI INDEKS STANDAR PENCEMAR UDARA (ISPU) DKI JAKARTA
### Menggunakan Support Vector Regression (SVR) dengan Kernel Radial Basis Function (RBF)

**Proyek Data Science — Kelompok 49**

</div>

---

## Identitas Kelompok

| Keterangan | Informasi |
|---|---|
| Nomor Kelompok | **49** |
| Nama Kelompok | **titut-titut** |
| Kode Repository | **49_IF-E_titut-titut** |

### Anggota Kelompok

| No. | Nama | NIM |
|---:|---|---|
| 1 | Nayla Saskia Zallianti | 123240016 |
| 2 | Cantika Zahra Putri Maharani | 123240091 |
| 3 | Dewi Rahmawati | 123240108 |
| 4 | Friska Rima Indriani | 123240111 |

---

## 1. Deskripsi Proyek

Kualitas udara di DKI Jakarta menjadi perhatian karena tingginya aktivitas kendaraan bermotor, industri, dan mobilitas penduduk yang dapat menyebabkan pencemaran udara. Salah satu indikator yang digunakan untuk menggambarkan kualitas udara adalah **Indeks Standar Pencemar Udara (ISPU)**.

Proyek ini bertujuan untuk menganalisis hubungan antara parameter pencemar udara dan nilai ISPU maksimum, sekaligus membangun model machine learning untuk mengestimasi nilai ISPU maksimum berdasarkan konsentrasi parameter pencemar.

Algoritma yang digunakan adalah **Support Vector Regression (SVR)** dengan kernel **Radial Basis Function (RBF)**. Metode ini dipilih karena dapat mempelajari hubungan nonlinier antara variabel pencemar udara dan nilai ISPU maksimum.

Proyek dikerjakan menggunakan Python dalam Jupyter Notebook.

## 2. Tujuan Penelitian

Tujuan dari proyek ini adalah:

1. Menganalisis pola dan hubungan antara parameter pencemar udara dengan nilai ISPU maksimum.
2. Melakukan Exploratory Data Analysis (EDA) untuk memahami karakteristik dataset.
3. Melakukan data cleaning dan preprocessing agar data siap digunakan dalam pemodelan.
4. Membangun model estimasi ISPU maksimum menggunakan algoritma SVR dengan kernel RBF.
5. Mengoptimalkan hyperparameter model, yaitu `C`, `gamma`, dan `epsilon`.
6. Mengevaluasi dan membandingkan performa model menggunakan MAE, RMSE, dan R².

## 3. Dataset

Dataset yang digunakan merupakan **data Indeks Standar Pencemar Udara Provinsi DKI Jakarta** yang berasal dari portal **Satu Data Jakarta**.

Dataset awal terdiri dari **3.501 baris dan 13 kolom**. Setelah proses cleaning, jumlah observasi menjadi **3.480 baris**.

### Variabel Penelitian

**Variabel input (X):**

| Parameter | Keterangan |
|---|---|
| `pm_sepuluh` | PM10 |
| `pm_duakomalima` | PM2.5 |
| `sulfur_dioksida` | SO2 |
| `karbon_monoksida` | CO |
| `ozon` | O3 |
| `nitrogen_dioksida` | NO2 |

**Variabel target (y):**

| Kolom | Keterangan |
|---|---|
| `max` | Nilai ISPU maksimum yang akan diestimasi |

Dataset juga memuat informasi pendukung seperti periode pengamatan, stasiun pemantauan, dan kategori kualitas udara. Namun, tidak seluruh kolom digunakan sebagai fitur model.

> **Catatan:** File dataset yang dibutuhkan notebook bernama `data-ispu-jakarta.csv`. Pastikan file tersebut tersedia di direktori yang sama dengan notebook sebelum menjalankan proyek.

## 4. Metodologi

Tahapan pengerjaan proyek meliputi Business Understanding, Data Understanding, Exploratory Data Analysis, Data Preparation, dan Modeling.

### 4.1 Business Understanding

Tahap ini menjelaskan permasalahan kualitas udara di DKI Jakarta, tujuan penelitian, dataset, variabel yang digunakan, metode pemodelan, stakeholder, kriteria keberhasilan, asumsi, batasan, risiko, serta manfaat penelitian.

### 4.2 Data Understanding dan Exploratory Data Analysis (EDA)

EDA dilakukan untuk memahami karakteristik dan pola pada data ISPU. Analisis mencakup:

- Pemeriksaan struktur dataset, jumlah baris dan kolom, serta tipe data.
- Pemeriksaan statistik deskriptif.
- Pemeriksaan distribusi nilai ISPU dan parameter pencemar.
- Analisis kategori kualitas udara.
- Analisis nilai ISPU berdasarkan stasiun pemantauan.
- Analisis tren rata-rata ISPU bulanan.
- Analisis hubungan parameter pencemar terhadap ISPU maksimum.
- Pemeriksaan korelasi antarvariabel.

Beberapa temuan EDA dalam notebook:

- Kategori **SEDANG** mendominasi data dengan persentase **76,97%**.
- PM2.5 memiliki korelasi positif paling kuat terhadap ISPU maksimum, yaitu **0,94**.
- PM10 memiliki korelasi sebesar **0,71** terhadap ISPU maksimum.
- Rata-rata ISPU bulanan tertinggi terjadi pada bulan Juni, yaitu **90,2**.
- DKI5 Kebon Jeruk memiliki rata-rata ISPU tertinggi sebesar **76,6**, sedangkan DKI4 Lubang Buaya memiliki rata-rata terendah sebesar **70,8**.

### 4.3 Data Preparation

Data preparation dilakukan untuk memastikan data siap digunakan dalam pemodelan SVR.

#### A. Audit Kualitas Data

Pemeriksaan awal meliputi missing value, duplikasi, kandidat outlier, dan nilai target yang perlu ditinjau.

Temuan audit awal:

- Jumlah data awal: 3.501 baris.
- Jumlah atribut: 13 kolom.
- Missing value terbanyak terdapat pada PM10 (`pm_sepuluh`), yaitu 264 data.
- Kolom target `max` memiliki 8 missing value.
- Tidak ditemukan baris duplikat.
- Kandidat outlier terbanyak terdapat pada O3, yaitu 169 data.

Kandidat outlier tidak langsung dihapus karena nilai ekstrem dapat merepresentasikan kondisi pencemaran udara yang sebenarnya.

#### B. Data Cleaning

Strategi cleaning yang dilakukan:

1. Menghapus baris duplikat apabila ditemukan.
2. Mengubah kolom parameter pencemar dan target `max` menjadi tipe numerik.
3. Menghapus baris yang tidak memiliki nilai target `max`.
4. Menghapus observasi dengan kategori `TIDAK ADA DATA`.
5. Mempertahankan missing value pada fitur pencemar untuk ditangani pada tahap imputasi.

Setelah cleaning, jumlah data berkurang dari **3.501 menjadi 3.480 baris**.

#### C. Pemilihan Fitur (Feature Selection)

Enam parameter pencemar digunakan sebagai fitur input:

- PM10
- PM2.5
- SO2
- CO
- O3
- NO2

Target yang digunakan adalah nilai ISPU maksimum (`max`).

Kolom waktu, stasiun, kategori, dan informasi turunan lainnya tidak digunakan sebagai fitur model. Pemilihan ini dilakukan agar model berfokus pada estimasi ISPU maksimum berdasarkan parameter pencemar dan menghindari potensi kebocoran informasi target.

#### D. Pembagian Data

Dataset dibagi menggunakan `train_test_split` dengan konfigurasi:

- **80% data latih (training data)**
- **20% data uji (testing data)**
- `random_state=42`

Pembagian dilakukan sebelum imputasi dan standardisasi untuk menghindari data leakage.

#### E. Imputasi dan Standardisasi

Preprocessing yang digunakan:

- **SimpleImputer (median):** mengisi missing value pada fitur menggunakan nilai median yang dipelajari dari data latih.
- **StandardScaler:** melakukan standardisasi agar skala antarfitur lebih sebanding.

Encoding tidak dilakukan karena seluruh enam fitur yang digunakan merupakan data numerik. SMOTE juga tidak diterapkan karena penelitian ini merupakan permasalahan regresi dengan target numerik kontinu, bukan klasifikasi.

### 4.4 Modeling

Model yang digunakan adalah **Support Vector Regression (SVR)** dengan kernel **RBF**.

Tahapan pemodelan:

1. Membuat baseline model menggunakan `DummyRegressor`.
2. Membangun model SVR awal dengan kernel RBF dan parameter default.
3. Melakukan hyperparameter tuning menggunakan `GridSearchCV`.
4. Membentuk model SVR terbaik berdasarkan kombinasi hyperparameter terpilih.
5. Mengevaluasi model pada data uji.

#### Hyperparameter Tuning

GridSearchCV menggunakan 5-Fold Cross Validation dengan kombinasi parameter berikut:

| Hyperparameter | Nilai yang diuji |
|---|---|
| `C` | 1, 10, 100 |
| `gamma` | `scale`, 0.01, 0.1 |
| `epsilon` | 0.1, 0.2, 0.5 |

Total terdapat **27 kombinasi hyperparameter** dan **135 proses fitting** selama 5-Fold Cross Validation.

Kombinasi parameter terbaik:

| Hyperparameter | Nilai terbaik |
|---|---:|
| `C` | 100 |
| `gamma` | 0.1 |
| `epsilon` | 0.5 |

RMSE terbaik pada cross-validation adalah **3,445**.

## 5. Hasil dan Evaluasi Model

Evaluasi dilakukan menggunakan tiga metrik:

- **MAE (Mean Absolute Error):** mengukur rata-rata nilai absolut selisih antara prediksi dan nilai aktual. Semakin kecil nilainya, semakin baik.
- **RMSE (Root Mean Squared Error):** mengukur akar dari rata-rata kuadrat kesalahan prediksi. Semakin kecil nilainya, semakin baik.
- **R² (Coefficient of Determination):** mengukur proporsi variasi target yang dapat dijelaskan model. Nilai yang semakin mendekati 1 menunjukkan kemampuan penjelasan model yang semakin baik.

### Perbandingan Performa Model

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline (`DummyRegressor`) | 17.6679 | 21.9796 | -0.0012 |
| SVR Awal (RBF) | 2.8423 | 6.8881 | 0.9017 |
| **SVR Terbaik (hasil tuning)** | **1.1435** | **4.1035** | **0.9651** |

### Interpretasi Hasil

Model SVR terbaik menghasilkan performa paling baik dibandingkan baseline dan SVR awal.

- MAE sebesar **1,1435** menunjukkan rata-rata kesalahan absolut prediksi sekitar 1,14 poin ISPU.
- RMSE sebesar **4,1035** menunjukkan besarnya kesalahan prediksi dengan penalti lebih besar pada error yang ekstrem.
- R² sebesar **0,9651** menunjukkan bahwa model mampu menjelaskan sekitar **96,51% variasi nilai ISPU maksimum** pada data pengujian.

Dibandingkan SVR awal, model hasil tuning mengalami penurunan MAE sekitar **59,8%** dan penurunan RMSE sekitar **40,4%**.

### Analisis Residual dan Visualisasi Prediksi

Notebook juga menyajikan beberapa visualisasi untuk mengevaluasi hasil model, yaitu:

- Residual plot untuk melihat selisih antara nilai aktual dan prediksi.
- Histogram residual untuk melihat distribusi kesalahan prediksi.
- Grafik Actual vs Predicted untuk membandingkan nilai aktual dengan hasil prediksi.
- Grafik perbandingan 50 observasi pertama pada data pengujian.

Secara umum, sebagian besar prediksi berada cukup dekat dengan nilai aktual. Namun, masih ditemukan beberapa observasi dengan kesalahan prediksi yang cukup besar, terutama kecenderungan overprediction pada sebagian nilai ISPU yang rendah.

## 6. Kesimpulan

Berdasarkan hasil penelitian, algoritma Support Vector Regression (SVR) dengan kernel Radial Basis Function (RBF) mampu mengestimasi nilai ISPU maksimum berdasarkan enam parameter pencemar udara, yaitu PM10, PM2.5, SO2, CO, O3, dan NO2.

Model terbaik diperoleh setelah proses hyperparameter tuning menggunakan GridSearchCV, dengan parameter `C = 100`, `gamma = 0.1`, dan `epsilon = 0.5`. Model tersebut menghasilkan MAE sebesar **1,1435**, RMSE sebesar **4,1035**, dan R² sebesar **0,9651** pada data pengujian.

Hasil tersebut menunjukkan bahwa proses tuning berhasil meningkatkan performa model dibandingkan SVR awal dan baseline. Meskipun demikian, masih terdapat beberapa observasi dengan kesalahan prediksi yang cukup besar.

Penelitian ini berfokus pada estimasi nilai ISPU maksimum berdasarkan parameter pencemar, bukan peramalan kualitas udara pada periode mendatang. Pembagian data secara acak juga belum menggambarkan kemampuan model dalam memprediksi periode waktu yang belum diamati. Penelitian selanjutnya dapat mempertimbangkan pembagian data berdasarkan waktu untuk mengevaluasi kemampuan prediksi temporal.

## 7. Teknologi dan Library

Proyek ini menggunakan:

- **Python** — bahasa pemrograman.
- **Jupyter Notebook** — lingkungan pengerjaan dan dokumentasi analisis.
- **Pandas** — pengolahan dan manipulasi data.
- **NumPy** — komputasi numerik.
- **Matplotlib** — visualisasi data.
- **Seaborn** — visualisasi dan analisis eksploratif.
- **Scikit-learn** — preprocessing, pembagian data, pemodelan SVR, tuning, dan evaluasi.

Library scikit-learn yang digunakan antara lain `train_test_split`, `GridSearchCV`, `KFold`, `StandardScaler`, `SimpleImputer`, `SVR`, `DummyRegressor`, `Pipeline`, `mean_absolute_error`, `mean_squared_error`, dan `r2_score`.

## 8. Struktur File Proyek

Struktur file yang disarankan:

```text
49_IF-E_titut-titut/
│
├── README.md
├── Projek_ISPU.ipynb
└── data-ispu-jakarta.csv
```

Keterangan:

| File | Fungsi |
|---|---|
| `README.md` | Dokumentasi proyek, tujuan, metodologi, dan hasil penelitian. |
| `Projek_ISPU.ipynb` | Notebook berisi proses analisis data, preprocessing, pemodelan, evaluasi, dan visualisasi. |
| `data-ispu-jakarta.csv` | Dataset ISPU DKI Jakarta yang dibaca oleh notebook. |

## 9. Cara Menjalankan Proyek

### Prasyarat

Pastikan Python dan Jupyter Notebook/JupyterLab sudah terpasang.

### Instalasi Library

Jalankan perintah berikut pada terminal atau cell notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Menjalankan Notebook

1. Clone atau unduh repository proyek.
2. Pastikan file `Projek_ISPU.ipynb` dan `data-ispu-jakarta.csv` berada pada direktori yang sesuai.
3. Buka terminal pada direktori proyek.
4. Jalankan Jupyter Notebook:

```bash
jupyter notebook
```

5. Buka file `Projek_ISPU.ipynb`.
6. Jalankan seluruh cell secara berurutan dari awal hingga akhir.

> **Penting:** Notebook membaca dataset menggunakan nama file relatif `data-ispu-jakarta.csv`. Jika lokasi atau nama file dataset berbeda, sesuaikan path pada bagian pembacaan dataset di notebook.

## 10. Batasan Penelitian

Beberapa batasan dalam proyek ini adalah:

1. Data yang digunakan hanya mencakup wilayah DKI Jakarta sesuai dataset penelitian.
2. Fitur model dibatasi pada enam parameter pencemar: PM10, PM2.5, SO2, CO, O3, dan NO2.
3. Target penelitian berupa estimasi nilai numerik ISPU maksimum, bukan klasifikasi kategori kualitas udara.
4. Hasil prediksi bukan merupakan pengukuran resmi kualitas udara.
5. Model tidak secara otomatis dapat digunakan untuk meramalkan kualitas udara masa depan karena penelitian berfokus pada estimasi berdasarkan parameter pencemar yang tersedia.
6. Pembagian data secara acak belum mengevaluasi performa model pada periode waktu mendatang.
7. Beberapa observasi masih menghasilkan kesalahan prediksi yang cukup besar.

---

<div align="center">

**49_IF-E_titut-titut**  
*Proyek Data Science — Analisis dan Estimasi ISPU DKI Jakarta*

</div>
