import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import tensorflow as tf
import cv2
import argparse
import time
import hashlib
import pandas as pd
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm as tqdm_cli
from datetime import datetime
from PIL import Image, ImageFile
from PIL.ExifTags import TAGS
import sys
import warnings

# ======================================================================
# --- KONFIGURASI GLOBAL ---
# ======================================================================
# Menentukan path model berdasarkan lingkungan
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    DEFAULT_MODEL_PATH = "/content/drive/My Drive/SKRIPSI/model/model3/best_model.keras" 
else:
    # Path relatif untuk penggunaan lokal
    DEFAULT_MODEL_PATH = "model/best_model.keras"

IMAGE_SIZE = (256, 256)
SUPPORTED_EXTS = ('.jpg', '.jpeg')
THRESHOLD = 0.5

# --- INISIALISASI ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ======================================================================
# --- FUNGSI PREPROCESSING & DETEKSI (Tidak ada perubahan di sini) ---
# ======================================================================
def extract_metadata(image_path):
    try:
        with Image.open(image_path) as img:
            exif_data = img.getexif()
            return {TAGS.get(k, str(k)): str(v) for k, v in exif_data.items()}
    except Exception:
        return {}

def metadata_to_image_array(metadata):
    try:
        byte_array = np.array([], dtype=np.uint8)
        if metadata:
            metadata_str = json.dumps(metadata, sort_keys=True)
            metadata_bytes = metadata_str.encode('utf-8', errors='ignore')
            byte_array = np.frombuffer(metadata_bytes, dtype=np.uint8)
        max_bytes = IMAGE_SIZE[0] * IMAGE_SIZE[1]
        if len(byte_array) > max_bytes: byte_array = byte_array[:max_bytes]
        else: byte_array = np.pad(byte_array, (0, max_bytes - len(byte_array)), 'constant')
        img_array = byte_array.reshape(IMAGE_SIZE).astype(np.float32) / 255.0
        return np.expand_dims(img_array, axis=-1)
    except Exception:
        return None

def load_detection_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan di path: {model_path}")
    return tf.keras.models.load_model(model_path)

def detect_single_image(image_path, model):
    start_time = time.time()
    try:
        with open(image_path, 'rb') as f:
            file_bytes = f.read()
            md5_hash = hashlib.md5(file_bytes).hexdigest()
            file_size = len(file_bytes)
        metadata = extract_metadata(image_path)
        image_array = metadata_to_image_array(metadata)
        details, score, result, status = "", -1.0, "ERROR", "ERROR"
        if image_array is not None:
            score = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0][0]
            result = "WEBSHELL" if score >= THRESHOLD else "BENIGN"
            status = "PROCESSED"
        else:
            details = "Gagal konversi metadata menjadi gambar"
        return {
            'timestamp': datetime.now().isoformat(), 'file': os.path.basename(image_path),
            'md5_hash': md5_hash, 'size_bytes': file_size, 'status': status,
            'details': details, 'score': f"{score:.4f}", 'result': result,
            'time_sec': f"{time.time() - start_time:.4f}", 'full_path': image_path
        }
    except Exception as e:
        return {'timestamp': datetime.now().isoformat(), 'file': os.path.basename(image_path), 'md5_hash': 'N/A', 'size_bytes': 0, 'status': 'CRITICAL_ERROR', 'details': str(e), 'score': -1.0, 'result': 'ERROR', 'time_sec': f"{time.time() - start_time:.4f}", 'full_path': image_path}


# ======================================================================
# --- FUNGSI INTERFACE (Tidak ada perubahan di sini) ---
# ======================================================================
def print_single_result(result_dict):
    """Mencetak hasil deteksi tunggal dengan format yang rapi."""
    print("\n" + "="*60)
    print("🔍 HASIL DETEKSI WEBSHELL")
    print("="*60)
    print(f"File         : {result_dict['file']}")
    print(f"Path         : {result_dict['full_path']}")
    print(f"Ukuran       : {result_dict['size_bytes']} bytes")
    print(f"MD5 Hash     : {result_dict['md5_hash']}")
    print("-" * 60)
    print(f"Hasil        : {result_dict['result']}")
    print(f"Skor         : {result_dict['score']}")
    print(f"Waktu Deteksi: {result_dict['time_sec']} detik")
    print("="*60)
    if result_dict['result'] == "WEBSHELL":
        print("🚨 PERINGATAN: Potensi webshell terdeteksi pada metadata gambar!")
        print("   Disarankan untuk segera mengkarantina file ini dan memeriksa server.")

def scan_folder_and_report(folder_path, model, report_path):
    """Memindai folder dan menyimpan laporan ke CSV (digunakan oleh Colab & Lokal)."""
    image_files = [os.path.join(r, f) for r, _, fs in os.walk(folder_path) for f in fs if f.lower().endswith(SUPPORTED_EXTS)]
    if not image_files:
        print(f"❌ Tidak ada gambar dengan ekstensi {SUPPORTED_EXTS} ditemukan di: {folder_path}")
        return

    tqdm_instance = tqdm_notebook if IN_COLAB else tqdm_cli
    print(f"🔍 Memindai {len(image_files)} file JPEG... Laporan akan disimpan di '{report_path}'")
    results_list = [detect_single_image(p, model) for p in tqdm_instance(image_files, desc="Memindai", unit="file")]

    report_df = pd.DataFrame(results_list)
    report_df['score'] = pd.to_numeric(report_df['score'])
    report_df = report_df.sort_values(by='score', ascending=False)
    report_df.to_csv(report_path, index=False, encoding='utf-8')

    webshell_count = (report_df['result'] == 'WEBSHELL').sum()
    error_count = (report_df['status'] != 'PROCESSED').sum()

    print("\n" + "="*60)
    print("📊 HASIL PEMINDAIAN FOLDER SELESAI")
    print("="*60)
    print(f"Total gambar dipindai : {len(results_list)}")
    print(f"Potensi Webshell      : {webshell_count}")
    print(f"Gagal Diproses        : {error_count}")
    if webshell_count > 0:
        print("\n🚨 DITEMUKAN POTENSI WEBSHELL (10 skor tertinggi):")
        print(report_df[report_df['result'] == 'WEBSHELL'][['file', 'score', 'size_bytes']].head(10).to_string(index=False))
    print(f"\n💾 Laporan lengkap telah disimpan di: {report_path}")
    print("="*60)

# ======================================================================
# --- FUNGSI UNTUK GOOGLE COLAB (Tidak ada perubahan di sini) ---
# ======================================================================
def colab_upload_and_detect(model):
    from google.colab import files
    print(f"Silakan upload file. Hanya file dengan ekstensi {SUPPORTED_EXTS} yang akan diproses.")
    uploaded = files.upload()

    valid_files_processed = 0
    for filename, content in uploaded.items():
        if not filename.lower().endswith(SUPPORTED_EXTS):
            print(f"⚠️  File '{filename}' DILEWATI karena bukan format yang didukung.")
            continue

        valid_files_processed += 1
        print(f"\n🔍 Memproses {filename}...")
        with open(filename, 'wb') as f: f.write(content)
        result = detect_single_image(filename, model)
        print_single_result(result)
        os.remove(filename)

    if valid_files_processed == 0:
        print("\nTidak ada file valid yang diupload untuk diproses.")

def colab_scan_drive_folder(model):
    from google.colab import drive
    try:
        drive.mount('/content/drive', force_remount=True)
        folder_path = input("Masukkan path folder lengkap di Google Drive Anda (contoh: /content/drive/My Drive/folder_gambar): ")
        if not os.path.isdir(folder_path):
            print(f"❌ Path tidak valid atau bukan folder: {folder_path}")
            return
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_name = f'laporan_colab_{timestamp}.csv'
        scan_folder_and_report(folder_path, model, report_name)
        print(f"Untuk mengunduh laporan, lihat panel file di sebelah kiri, klik kanan pada '{report_name}' dan pilih 'Download'.")
    except Exception as e:
        print(f"Terjadi error: {e}")

# ======================================================================
# --- FUNGSI UTAMA (Tidak ada perubahan di sini) ---
# ======================================================================
def main():
    """Fungsi utama yang menjalankan UI berbeda tergantung lingkungan."""
    try:
        print("Memuat model...")
        model = load_detection_model(DEFAULT_MODEL_PATH)
        print(f"✅ Model berhasil dimuat.")
    except Exception as e:
        print(f"❌ GAGAL MEMUAT MODEL: {e}\nPastikan path '{DEFAULT_MODEL_PATH}' sudah benar dan file model ada."); return

    if IN_COLAB:
        while True:
            print("\n" + "="*60)
            print("WEBSHELL DETECTOR ")
            print("="*60)
            print("Pilih opsi:")
            print("1. Upload file JPG/JPEG untuk dipindai")
            print("2. Pindai satu folder dari Google Drive (hanya JPG/JPEG)")
            print("3. Keluar")
            choice = input("Masukkan pilihan Anda (1-3): ")
            if choice == '1': colab_upload_and_detect(model)
            elif choice == '2': colab_scan_drive_folder(model)
            elif choice == '3': print("Keluar dari program."); break
            else: print("Pilihan tidak valid, silakan coba lagi.")
    else:
        # --- ANTARMUKA BARIS PERINTAH (CLI) UNTUK LOKAL ---
        parser = argparse.ArgumentParser(description='Deteksi Webshell dalam Metadata Gambar (Fokus pada EXIF JPG/JPEG).', formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('path', type=str, help='Path ke satu file gambar atau sebuah folder yang akan dipindai.')
        parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help=f'Path ke file model .keras.\nDefault: {DEFAULT_MODEL_PATH}')
        parser.add_argument('--output', '-o', type=str, help='Nama file untuk menyimpan laporan CSV.\nDefault: report_[nama_folder]_[timestamp].csv')
        args = parser.parse_args()

        if os.path.isfile(args.path):
            if not args.path.lower().endswith(SUPPORTED_EXTS):
                print(f"❌ Error: File '{os.path.basename(args.path)}' bukan format yang didukung ({SUPPORTED_EXTS}).")
                return
            result = detect_single_image(args.path, model)
            print_single_result(result)
        elif os.path.isdir(args.path):
            folder_name = os.path.basename(os.path.normpath(args.path))
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_name = args.output or f"report_{folder_name}_{timestamp}.csv"
            scan_folder_and_report(args.path, model, report_name)
        else:
            print(f"❌ Path tidak valid: {args.path}")

if __name__ == "__main__":
    main()