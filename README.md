<<<<<<< HEAD
# Modul Deteksi Webshell dalam Metadata Gambar

Repositori ini berisi kode sumber untuk penelitian skripsi berjudul "Pengembangan Modul Deteksi Gambar yang Mengandung Webshell Menggunakan CNN Berbasis Python".

## Deskripsi Singkat
Modul ini adalah sebuah alat prototipe yang dirancang untuk mendeteksi keberadaan skrip PHP webshell yang disembunyikan di dalam metadata EXIF pada gambar berformat JPG/JPEG. Deteksi dilakukan dengan mengubah metadata menjadi representasi citra dan menganalisisnya menggunakan model Convolutional Neural Network (CNN).

## Fitur Utama
- **Deteksi Berbasis Konten:** Menganalisis pola visual dari metadata.
- **Antarmuka Adaptif:** Berjalan sebagai antarmuka baris perintah (CLI) di lingkungan lokal dan sebagai menu interaktif di Google Colaboratory.
- **Pemindaian Massal:** Mampu memindai satu direktori penuh dan menghasilkan laporan dalam format `.csv`.

## Memulai (Quick Start)
Untuk panduan instalasi dan penggunaan yang lengkap, silakan merujuk ke dokumen **[panduan_pengguna.pdf](panduan_pengguna.pdf)** yang ada di repositori ini.

## Struktur Repositori
- `detector.py`: Skrip utama untuk menjalankan deteksi.
- `requirements.txt`: Daftar pustaka Python yang dibutuhkan.
- `/model/`: Direktori untuk menyimpan file model `.keras` yang telah dilatih.
- `/images/`: Berisi beberapa contoh gambar untuk pengujian cepat.
=======
# php-webshell-in-exif-metadata-detection-using-cnn
>>>>>>> 02a63a20c5baed53c1f9680239118020f99a85d8
