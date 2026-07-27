# Prediksi Performa Siswa

## Pernyataan Masalah

Institusi pendidikan sering kali tidak punya cara berbasis data yang cepat untuk mengidentifikasi siswa yang berisiko mendapat performa akademik buruk sebelum evaluasi akhir berlangsung. Pengajar biasanya mengandalkan observasi informal, yang tidak bisa diterapkan dalam skala besar dan seringkali baru terdeteksi setelah terlambat untuk intervensi yang berarti. Project ini mengeksplorasi apakah indikator keterlibatan dan latar belakang yang terukur — waktu belajar, kehadiran, jam tidur, performa akademik sebelumnya, dan faktor pendukung lain — bisa digunakan untuk memprediksi apakah seorang siswa berpotensi lulus atau tidak.

## Tujuan

- Membangun pipeline klasifikasi end-to-end yang reproducible untuk memprediksi hasil lulus/tidak lulus siswa
- Mengidentifikasi faktor perilaku dan latar belakang mana yang paling berkorelasi dengan performa akademik
- Mendemonstrasikan struktur project ML berbasis config (tahap cleaning, EDA, feature engineering, dan modeling yang terpisah) yang sesuai untuk alur kerja bergaya produksi
- Membangun model baseline serta memvalidasi secara ketat apakah performanya mencerminkan keterbatasan asli dari fitur yang tersedia, bukan sekadar masalah modeling atau implementasi

## Dataset

Dataset berisi data tingkat siswa dengan atribut berikut:

| Kolom | Keterangan |
|---|---|
| `student_id` | ID unik siswa |
| `hours_studied` | Jam belajar per minggu |
| `attendance_rate` | Persentase kehadiran kelas |
| `sleep_hours` | Rata-rata jam tidur per hari |
| `previous_score` | Nilai ujian sebelumnya |
| `parental_education` | Tingkat pendidikan orang tua tertinggi |
| `extracurricular` | Partisipasi dalam kegiatan ekstrakurikuler (Yes/No) |
| `gender` | Jenis kelamin siswa |
| `passed` | Variabel target — apakah siswa lulus (1) atau tidak (0) |

Data mentah memerlukan proses cleaning: label kategorikal yang tidak konsisten, tipe data campuran (persentase tersimpan sebagai string), nilai di luar rentang wajar, entri yang hilang, dan data duplikat — semuanya ditangani di tahap data cleaning sebelum analisis.

## Pendekatan

Project ini mengikuti pipeline modular:

1. **Data Cleaning** — menstandardkan nilai kategorikal, memperbaiki tipe data, menangani missing value dan entri invalid, menghapus duplikat
2. **Exploratory Data Analysis** — analisis distribusi, cek keseimbangan target, analisis korelasi, dan perbandingan fitur terhadap target
3. **Feature Engineering** — encoding kategorikal (binary dan one-hot); fitur turunan `study_efficiency` sempat diuji dan akhirnya dihapus setelah validasi menunjukkan fitur ini tidak menambah nilai prediktif (lihat bagian Results)
4. **Modeling** — model klasifikasi baseline dengan hyperparameter yang dapat dikonfigurasi, feature scaling, dan cross-validation

Semua path, threshold, dan hyperparameter dieksternalisasi ke `config/config.yaml`, menjaga codebase tetap environment-agnostic dan mudah dikonfigurasi ulang.

## Model

**Logistic Regression** dipilih sebagai model baseline. Dipilih karena interpretabilitasnya — koefisien model langsung menunjukkan arah dan kekuatan relatif pengaruh tiap fitur terhadap hasil lulus/tidak lulus — menjadikannya cocok sebagai model pertama untuk memahami faktor apa yang mempengaruhi performa siswa, sebelum mempertimbangkan algoritma yang lebih kompleks. Fitur distandardisasi (`StandardScaler`) sebelum training, baik untuk memastikan model konvergen dengan baik maupun supaya besaran koefisien bisa dibandingkan secara adil antar fitur yang skalanya berbeda-beda.

**Random Forest** juga diuji sebagai pembanding non-linear, untuk membantu membedakan apakah performa yang terbatas berasal dari asumsi linear model atau memang dari fitur itu sendiri.

## Metrik Evaluasi

Model dievaluasi menggunakan:
- **Accuracy** — ketepatan prediksi secara keseluruhan
- **Precision** — proporsi prediksi "lulus" yang benar-benar tepat
- **Recall** — proporsi kasus "lulus" aktual yang berhasil teridentifikasi
- **F1-score** — rata-rata harmonik dari precision dan recall

5-fold cross-validation digunakan (bukan single train/test split) untuk mendapatkan estimasi yang lebih bisa dipercaya, mengingat ukuran test set yang kecil kalau hanya pakai satu kali split.

## Hasil

| Model | CV Accuracy (rata-rata 5-fold) |
|---|---|
| Logistic Regression (baseline) | ~0.542 |
| Logistic Regression (tanpa `study_efficiency`) | ~0.544 |
| Random Forest | ~0.522 |

Metrik single-split untuk Logistic Regression baseline: accuracy 0.53, precision 0.53, recall 0.48, F1 0.51.

**Temuan utama:**
- Convergence warning yang muncul di awal training berhasil diatasi dengan standardisasi fitur — ini juga membuat koefisien fitur bisa dibandingkan secara langsung, mengungkap `previous_score`, `attendance_rate`, dan `gender` sebagai prediktor linear terkuat terhadap `passed`.
- Fitur turunan `study_efficiency` (nilai sebelumnya relatif terhadap jam belajar) tidak menunjukkan peningkatan akurasi yang terukur pada cross-validation, dan akhirnya dihapus untuk menjaga fitur tetap bersih dan menghindari redundansi dengan fitur sumbernya.
- Random Forest — yang mampu menangkap hubungan non-linear — tidak menunjukkan performa lebih baik dibanding baseline linear. Karena dua pendekatan modeling yang secara struktural berbeda sama-sama konvergen ke performa yang serupa dan moderat, ini mengindikasikan keterbatasan yang memang ada pada fitur yang tersedia, bukan keterbatasan dari model linear.

**Kesimpulan:** fitur yang ada saat ini (jam belajar, kehadiran, jam tidur, nilai sebelumnya, dan faktor latar belakang) memiliki sinyal yang terbatas untuk memprediksi hasil lulus/tidak lulus secara mandiri. Peningkatan performa yang berarti kemungkinan besar membutuhkan fitur yang lebih kaya — seperti tingkat penyelesaian tugas, metrik keterlibatan, atau penilaian dari pengajar — dibandingkan sekadar melakukan tuning lebih lanjut pada fitur yang ada sekarang.

## Struktur Project

```
student-performance/
├── data/               # dataset raw, interim, processed, external
├── notebooks/          # data cleaning, EDA, feature engineering, modeling
├── src/                # kode pipeline reusable (data, features, models, visualization)
├── models/             # artifact model terlatih
├── reports/            # metrik dan figure yang dihasilkan
├── tests/              # unit test untuk fungsi pipeline
└── config/             # konfigurasi terpusat (path, threshold, hyperparameter)
```