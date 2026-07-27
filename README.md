
# Prediksi Performa Siswa

## Pernyataan Masalah

Institusi pendidikan sering kali tidak punya cara berbasis data yang cepat untuk mengidentifikasi siswa yang berisiko mendapat performa akademik buruk sebelum evaluasi akhir berlangsung. Pengajar biasanya mengandalkan observasi informal, yang tidak bisa diterapkan dalam skala besar dan seringkali baru terdeteksi setelah terlambat untuk intervensi yang berarti. Project ini mengeksplorasi apakah indikator keterlibatan dan latar belakang yang terukur — waktu belajar, kehadiran, jam tidur, performa akademik sebelumnya, dan faktor pendukung lain — bisa digunakan untuk memprediksi apakah seorang siswa berpotensi lulus atau tidak.

## Tujuan

- Membangun pipeline klasifikasi end-to-end yang reproducible untuk memprediksi hasil lulus/tidak lulus siswa
- Mengidentifikasi faktor perilaku dan latar belakang mana yang paling berkorelasi dengan performa akademik
- Mendemonstrasikan struktur project ML berbasis config (tahap cleaning, EDA, feature engineering, dan modeling yang terpisah) yang sesuai untuk alur kerja bergaya produksi
- Membangun model baseline yang nantinya bisa dikembangkan dengan fitur lebih kaya atau algoritma yang lebih kompleks

## Dataset

Dataset berisi data tingkat siswa dengan atribut berikut:

| Kolom                  | Keterangan                                               |
| ---------------------- | -------------------------------------------------------- |
| `student_id`         | ID unik siswa                                            |
| `hours_studied`      | Jam belajar per minggu                                   |
| `attendance_rate`    | Persentase kehadiran kelas                               |
| `sleep_hours`        | Rata-rata jam tidur per hari                             |
| `previous_score`     | Nilai ujian sebelumnya                                   |
| `parental_education` | Tingkat pendidikan orang tua tertinggi                   |
| `extracurricular`    | Partisipasi dalam kegiatan ekstrakurikuler (Yes/No)      |
| `gender`             | Jenis kelamin siswa                                      |
| `passed`             | Variabel target — apakah siswa lulus (1) atau tidak (0) |

Data mentah memerlukan proses cleaning: label kategorikal yang tidak konsisten, tipe data campuran (persentase tersimpan sebagai string), nilai di luar rentang wajar, entri yang hilang, dan data duplikat — semuanya ditangani di tahap data cleaning sebelum analisis.

## Pendekatan

Project ini mengikuti pipeline modular:

1. **Data Cleaning** — menstandardkan nilai kategorikal, memperbaiki tipe data, menangani missing value dan entri invalid, menghapus duplikat
2. **Exploratory Data Analysis** — analisis distribusi, cek keseimbangan target, analisis korelasi, dan perbandingan fitur terhadap target
3. **Feature Engineering** — encoding kategorikal (binary dan one-hot) serta fitur turunan `study_efficiency`
4. **Modeling** — model klasifikasi baseline dengan hyperparameter yang dapat dikonfigurasi

Semua path, threshold, dan hyperparameter dieksternalisasi ke `config/config.yaml`, menjaga codebase tetap environment-agnostic dan mudah dikonfigurasi ulang.

## Model

**Logistic Regression** dipilih sebagai model baseline. Dipilih karena interpretabilitasnya — koefisien model langsung menunjukkan arah dan kekuatan relatif pengaruh tiap fitur terhadap hasil lulus/tidak lulus — menjadikannya cocok sebagai model pertama untuk memahami faktor apa yang mempengaruhi performa siswa, sebelum mempertimbangkan algoritma yang lebih kompleks.

## Metrik Evaluasi

Model dievaluasi menggunakan:

- **Accuracy** — ketepatan prediksi secara keseluruhan
- **Precision** — proporsi prediksi "lulus" yang benar-benar tepat
- **Recall** — proporsi kasus "lulus" aktual yang berhasil teridentifikasi
- **F1-score** — rata-rata harmonik dari precision dan recall

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
