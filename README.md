# Modul Deteksi Webshell dalam Metadata Gambar

Repositori ini berisi kode sumber dan material pendukung untuk penelitian skripsi berjudul "Pengembangan Modul Deteksi Gambar yang Mengandung Webshell Menggunakan CNN Berbasis Python".

## 📖 Deskripsi
Proyek ini mengembangkan sebuah modul deteksi berbasis *Convolutional Neural Network* (CNN) untuk mengidentifikasi skrip PHP *webshell* yang disembunyikan di dalam metadata EXIF pada gambar berformat JPG/JPEG. Metode yang digunakan adalah mengubah metadata menjadi representasi citra *grayscale*, yang kemudian dianalisis oleh model CNN yang telah dioptimalkan untuk mengenali pola-pola anomali.

## ✨ Fitur Utama
- **Deteksi Berbasis Konten:** Menganalisis pola visual dari metadata, tangguh terhadap teknik obfuskasi teks.
- **Antarmuka Adaptif:** Modul deteksi dapat berjalan sebagai antarmuka baris perintah (CLI) di lingkungan lokal dan sebagai menu interaktif di Google Colaboratory.
- **Alur Kerja Terstruktur:** Menyediakan skrip terpisah untuk persiapan data, pelatihan model, dan deteksi, memungkinkan reprodusibilitas penelitian yang tinggi.
- **Fungsionalitas Pelatihan Ulang:** Memungkinkan pengguna untuk melatih ulang model dengan dataset baru untuk beradaptasi dengan ancaman yang terus berkembang.

## 📁 Struktur Repositori
- `/model/`: Direktori untuk menyimpan file model `.keras` yang telah dilatih.
- `/notebooks/`: Berisi *notebook* Jupyter/Colab yang mendokumentasikan alur kerja.
    - `01_data_preparation.ipynb`: Kode untuk proses persiapan data.
    - `02_model_training.ipynb`: Kode untuk proses *hyperparameter tuning* dan pelatihan.
- `/src/`: Berisi skrip-skrip Python utama.
    - `data_preparation.py`: Skrip untuk membuat dan memproses dataset.
    - `model_training.py`: Skrip untuk melatih model.
    - `detector.py`: Skrip utama untuk menjalankan deteksi.
- `requirements.txt`: Daftar pustaka Python yang dibutuhkan.
- `Panduan Instalasi dan Penggunaan.pdf`: Dokumen panduan instalasi dan penggunaan modul deteksi.
- `README.md`: Dokumen ini.

---

## 🚀 Memulai (Quick Start)
Untuk panduan instalasi dan penggunaan yang lengkap dan terperinci, silakan merujuk ke dokumen **`Panduan Instalasi dan Penggunaan.pdf`** yang ada di repositori ini.

---

## 🛠️ Alur Kerja Penelitian
Proyek ini dibagi menjadi tiga skrip utama yang mencerminkan alur kerja penelitian:

### 1. Persiapan Data (`data_preparation.py`)
Skrip ini bertanggung jawab untuk mengubah data mentah (gambar bersih dan skrip *webshell*) menjadi dataset yang siap dilatih dalam format `.npy`.
- **Fitur:**
    - Membuat dataset *malicious* dengan menyisipkan *webshell* ke metadata EXIF.
    - Mengonversi metadata dari seluruh dataset (jinak & berbahaya) menjadi citra *grayscale*.
    - Menyimpan dataset akhir sebagai file `.npy`.
- **Cara Menjalankan (Lokal):**
  ```bash
  # Membuat dataset malicious
  python src/data_preparation.py create_malicious --input_dir "path/ke/gambar/bersih" --webshell_dir "path/ke/webshell" --output_dir "path/output/malicious"

  # Mengonversi dataset menjadi format .npy
  python src/data_preparation.py create_numpy --base_path "path/ke/dataset/utama"


### 2. Pelatihan Model (model_training.py)
Skrip ini menangani proses hyperparameter tuning untuk menemukan arsitektur CNN terbaik dan melatih model final.

- **Fitur:**
    - Pencarian arsitektur optimal menggunakan Bayesian Optimization.
    - Pelatihan model final dengan konfigurasi terbaik.
    - Evaluasi model dan pembuatan laporan kinerja (Confusion Matrix, Kurva ROC).

- **Cara Menjalankan (Lokal):**
  ```bash
  python src/model_training.py --dataset_dir "path/ke/dataset_npy" --output_dir "path/hasil/pelatihan"


### 3. Modul Deteksi (detector.py)
Ini adalah implementasi fungsional dari model yang telah dilatih, yang dapat digunakan untuk memindai file baru.

- **Fitur:**
    - Memindai file gambar tunggal atau direktori penuh.
    - Menghasilkan laporan deteksi dalam format .csv.
    - Antarmuka adaptif untuk lingkungan lokal (CLI) dan Google Colab.

- **Cara Menjalankan (Lokal):**
  ```bash
  # Memindai satu file
  python src/detector.py scan "path/ke/gambar.jpg"

  # Memindai seluruh folder
  python src/detector.py scan "path/ke/folder_gambar" --output "laporan.csv"