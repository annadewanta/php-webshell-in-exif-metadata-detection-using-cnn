import os
import json
import numpy as np
import cv2
import argparse
import random
import datetime
import sys
import warnings
from PIL import Image, ImageFile
from PIL.ExifTags import TAGS
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm as tqdm_cli

# ======================================================================
# --- KONFIGURASI DAN INISIALISASI ---
# ======================================================================
IMAGE_SIZE = (256, 256)
MAX_EXIF_PAYLOAD_SIZE = 60000
IN_COLAB = 'google.colab' in sys.modules

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

# ======================================================================
# --- FUNGSI-FUNGSI LOGIKA ---
# ======================================================================

def prepare_webshell_content_for_exif(content_bytes, max_size):
    if len(content_bytes) > max_size:
        content_bytes = content_bytes[:max_size]
    return {'UserComment': content_bytes}

def inject_jpeg_metadata(exif_data, webshell_metadata_dict):
    tag_map = {v: k for k, v in TAGS.items()}
    user_comment_id = tag_map.get('UserComment', 0x9286)
    webshell_payload = webshell_metadata_dict.get('UserComment')
    if webshell_payload is not None:
        exif_data[user_comment_id] = webshell_payload

def process_image_with_metadata_injection(input_path, output_path, webshell_content_bytes):
    webshell_metadata = prepare_webshell_content_for_exif(webshell_content_bytes, MAX_EXIF_PAYLOAD_SIZE)
    if not webshell_metadata.get('UserComment'):
        raise ValueError("Webshell content is empty after truncation.")
    img = Image.open(input_path)
    exif_data = img.getexif()
    inject_jpeg_metadata(exif_data, webshell_metadata)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, exif=exif_data, quality=95)
    return True

def get_webshell_content(ws_path):
    with open(ws_path, 'rb') as f:
        return f.read()

def extract_metadata(image_path):
    try:
        with Image.open(image_path) as img:
            exif_data = img.getexif()
            return {TAGS.get(tag_id, tag_id): str(value) for tag_id, value in exif_data.items()}
    except Exception:
        return {}

def metadata_to_image_array(metadata, size=(256, 256)):
    try:
        byte_array = np.array([], dtype=np.uint8)
        if metadata:
            metadata_str = json.dumps(metadata, sort_keys=True)
            metadata_bytes = metadata_str.encode('utf-8', errors='ignore')
            byte_array = np.frombuffer(metadata_bytes, dtype=np.uint8)
        max_bytes = size[0] * size[1]
        if len(byte_array) > max_bytes:
            byte_array = byte_array[:max_bytes]
        else:
            byte_array = np.pad(byte_array, (0, max_bytes - len(byte_array)), 'constant')
        return byte_array.reshape(size)
    except Exception:
        return None

# ======================================================================
# --- FUNGSI UTAMA ---
# ======================================================================

def create_malicious_dataset(input_dir, webshell_dir, output_dir):
    print("\n" + "="*50)
    print("TAHAP 1: PEMBUATAN DATASET MALICIOUS")
    print("="*50)

    print(">> Mengumpulkan file...")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    webshell_files = [f for f in os.listdir(webshell_dir) if f.endswith('.php')]
    if not image_files or not webshell_files:
        print("!! Error: Pastikan direktori input gambar dan webshell tidak kosong.")
        return

    print(f"-> Ditemukan {len(image_files)} gambar dan {len(webshell_files)} webshell.")
    webshell_sizes = sorted([(f, os.path.getsize(os.path.join(webshell_dir, f))) for f in webshell_files], key=lambda x: x[1])
    sorted_webshells = [ws[0] for ws in webshell_sizes]
    smallest_webshell_name = sorted_webshells[0]
    smallest_webshell_path = os.path.join(webshell_dir, smallest_webshell_name)
    
    random.shuffle(image_files)
    success_count = 0
    fail_count = 0
    
    tqdm_instance = tqdm_notebook if IN_COLAB else tqdm_cli
    print("\n>> Memulai proses injeksi webshell ke metadata...")
    for i, img_file in enumerate(tqdm_instance(image_files, desc="Injeksi Webshell", unit="file")):
        img_path = os.path.join(input_dir, img_file)
        output_file_name = f"{os.path.splitext(img_file)[0]}_ws_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
        output_path = os.path.join(output_dir, output_file_name)
        
        injection_successful = False
        ws_file_to_try = sorted_webshells[i % len(sorted_webshells)]
        ws_path_to_try = os.path.join(webshell_dir, ws_file_to_try)
        ws_content_to_try = get_webshell_content(ws_path_to_try)

        try:
            process_image_with_metadata_injection(img_path, output_path, ws_content_to_try)
            injection_successful = True
        except Exception as e:
            if "EXIF data too long" in str(e):
                try:
                    ws_content_smallest = get_webshell_content(smallest_webshell_path)
                    process_image_with_metadata_injection(img_path, output_path, ws_content_smallest)
                    injection_successful = True
                except:
                    fail_count += 1
            else:
                fail_count += 1
        
        if injection_successful:
            success_count += 1

    print(f"\n--- Ringkasan Proses Injeksi Selesai ---")
    print(f"Berhasil Diinjeksi: {success_count}/{len(image_files)}")

def create_numpy_dataset(base_path, image_size=(256, 256)):
    print("\n" + "="*50)
    print("TAHAP 2: PEMBUATAN DATASET NUMPY (.npy)")
    print("="*50)

    splits = ["train", "val", "test"]
    categories = ["malicious", "benign"]
    numpy_output_path = base_path
    
    tqdm_instance = tqdm_notebook if IN_COLAB else tqdm_cli

    for split in splits:
        images, labels = [], []
        print(f"\n>> Memproses set data: '{split}'")
        for category in categories:
            input_folder = os.path.join(base_path, split, category)
            if not os.path.exists(input_folder): continue
            
            image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg'))]
            for img_file in tqdm_instance(image_files, desc=f"  -> Kategori '{category}'", unit="file", leave=False):
                img_path = os.path.join(input_folder, img_file)
                metadata = extract_metadata(img_path)
                img_array = metadata_to_image_array(metadata, size=image_size)
                
                if img_array is not None:
                    img_normalized = img_array.astype(np.float32) / 255.0
                    img_tensor = np.expand_dims(img_normalized, axis=-1)
                    images.append(img_tensor)
                    labels.append(1 if category == "malicious" else 0)

        if images:
            np.save(os.path.join(numpy_output_path, f"X_{split}.npy"), np.array(images))
            np.save(os.path.join(numpy_output_path, f"y_{split}.npy"), np.array(labels))
            print(f"✅ Set '{split}': Berhasil disimpan {len(images)} sampel.")
    
    print("\n✅ SEMUA PROSES PERSIAPAN DATA SELESAI")

# ======================================================================
# --- FUNGSI MAIN (PENGENDALI UTAMA) ---
# ======================================================================
def main():
    if IN_COLAB:
        # Antarmuka interaktif untuk Google Colab
        print("Menjalankan dalam mode Google Colab.")
        
        # Tahap 1: Pembuatan Dataset Malicious
        input_dir = input("Masukkan path ke folder gambar bersih (di Drive): ")
        webshell_dir = input("Masukkan path ke folder skrip webshell (di Drive): ")
        output_dir = input("Masukkan path folder output untuk gambar malicious (di Drive): ")
        if input("Jalankan pembuatan dataset malicious? (y/n): ").lower() == 'y':
            create_malicious_dataset(input_dir, webshell_dir, output_dir)
            
        # Tahap 2: Pembuatan Dataset .npy
        base_path = input("\nMasukkan path ke folder dataset utama (yang berisi train, val, test): ")
        if input("Jalankan pembuatan dataset .npy? (y/n): ").lower() == 'y':
            create_numpy_dataset(base_path)

    else:
        # Antarmuka baris perintah (CLI) untuk lingkungan lokal
        parser = argparse.ArgumentParser(description="Skrip lengkap untuk persiapan dataset deteksi webshell.")
        subparsers = parser.add_subparsers(dest='command', required=True, help='Pilih perintah: "create_malicious" atau "create_numpy"')
        
        # Parser untuk perintah 'create_malicious'
        p_malicious = subparsers.add_parser('create_malicious', help='Menyisipkan webshell ke dalam metadata gambar bersih.')
        p_malicious.add_argument('--input_dir', required=True, help='Path ke folder gambar bersih.')
        p_malicious.add_argument('--webshell_dir', required=True, help='Path ke folder skrip webshell.')
        p_malicious.add_argument('--output_dir', required=True, help='Path folder output untuk menyimpan gambar malicious.')
        
        # Parser untuk perintah 'create_numpy'
        p_numpy = subparsers.add_parser('create_numpy', help='Mengonversi dataset gambar menjadi format .npy.')
        p_numpy.add_argument('--base_path', required=True, help='Path ke folder dataset utama (yang berisi train, val, test).')
        
        args = parser.parse_args()
        
        if args.command == 'create_malicious':
            create_malicious_dataset(args.input_dir, args.webshell_dir, args.output_dir)
        elif args.command == 'create_numpy':
            create_numpy_dataset(args.base_path)

if __name__ == "__main__":
    main()