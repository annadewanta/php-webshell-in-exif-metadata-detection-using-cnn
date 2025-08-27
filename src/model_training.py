import os
import json
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import layers, callbacks
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import argparse
import sys
import shutil

# ======================================================================
# --- KONFIGURASI DAN INISIALISASI ---
# ======================================================================
IN_COLAB = 'google.colab' in sys.modules
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ======================================================================
# --- FUNGSI-FUNGSI LOGIKA ---
# ======================================================================

def load_data(preprocess_dir):
    """Memuat dataset .npy dari direktori yang ditentukan."""
    print("Memuat dataset...")
    try:
        X_train = np.load(os.path.join(preprocess_dir, "X_train.npy"))
        y_train = np.load(os.path.join(preprocess_dir, "y_train.npy"))
        X_val = np.load(os.path.join(preprocess_dir, "X_val.npy"))
        y_val = np.load(os.path.join(preprocess_dir, "y_val.npy"))
        X_test = np.load(os.path.join(preprocess_dir, "X_test.npy"))
        y_test = np.load(os.path.join(preprocess_dir, "y_test.npy"))
        
        print("\n📊 Data Shapes:")
        print(f"Train: {X_train.shape}, {y_train.shape}")
        print(f"Val:   {X_val.shape}, {y_val.shape}")
        print(f"Test:  {X_test.shape}, {y_test.shape}\n")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    except FileNotFoundError as e:
        print(f"❌ Error: File dataset tidak ditemukan. Pastikan path '{preprocess_dir}' benar dan berisi file .npy.")
        print(f"Detail: {e}")
        sys.exit(1)

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(256, 256, 1))) 
    num_blocks = hp.Int('num_blocks', 2, 4)
    for i in range(num_blocks):
        kernel_size = hp.Choice(f'kernel_{i}', [3, 5, 7]) 
        filters = hp.Choice(f'filters_{i}', [32, 64, 128]) 
        model.add(layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        pool_size = hp.Choice(f'pool_{i}', [2, 3, 4]) 
        model.add(layers.MaxPooling2D(pool_size))
        model.add(layers.BatchNormalization())
        dropout_rate = hp.Float(f'dropout_{i}', 0.2, 0.5, step=0.1) 
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.GlobalAveragePooling2D())
    dense_units = hp.Int('dense_units', 64, 512, step=64) 
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(hp.Float('dense_dropout', 0.3, 0.7, step=0.1)))
    model.add(layers.Dense(1, activation='sigmoid'))
    optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop'])
    lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
    optimizer = tf.keras.optimizers.get(optimizer_name)
    optimizer.learning_rate = lr
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def run_tuning(data, args):
    """Menjalankan proses hyperparameter tuning."""
    (X_train, y_train), (X_val, y_val) = data
    
    print("\n" + "="*50)
    print("MEMULAI TAHAP 1: HYPERPARAMETER TUNING")
    print("="*50)
    
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_auc',
        max_trials=args.max_trials,
        executions_per_trial=args.exec_per_trial,
        directory=args.output_dir,
        project_name='webshell_tuning',
        overwrite=args.overwrite
    )

    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.tuning_epochs,
        batch_size=args.batch_size,
        callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)],
        verbose=1
    )
    
    print("\n💾 Menyimpan hasil tuning...")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_hp_path = os.path.join(args.output_dir, 'best_hyperparameters.json')
    with open(best_hp_path, 'w') as f:
        json.dump(best_hps.values, f, indent=2)
    print(f"✅ Hyperparameter terbaik disimpan di: {best_hp_path}")
    
    return tuner, best_hps

def train_final_model(tuner, best_hps, data, args):
    """Melatih model final dengan hyperparameter terbaik."""
    (X_train, y_train), (X_val, y_val), _ = data
    
    print("\n" + "="*50)
    print("MEMULAI TAHAP 2: PELATIHAN MODEL FINAL")
    print("="*50)
    
    final_model = tuner.hypermodel.build(best_hps)
    print("Arsitektur Model Final:")
    final_model.summary()

    # Menggabungkan data latih dan validasi untuk pelatihan akhir
    X_full = np.concatenate((X_train, X_val))
    y_full = np.concatenate((y_train, y_val))

    output_model_path = os.path.join(args.output_dir, 'best_model.keras')
    final_callbacks = [
        callbacks.EarlyStopping(monitor='loss', patience=args.patience, restore_best_weights=True),
        callbacks.ModelCheckpoint(output_model_path, monitor='loss', save_best_only=True, mode='min'),
        callbacks.CSVLogger(os.path.join(args.output_dir, 'final_training_log.csv')),
        callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-6)
    ]

    print(f"\n🚀 Memulai training... Model akan disimpan di '{output_model_path}'")
    history = final_model.fit(
        X_full, y_full,
        epochs=args.final_epochs,
        batch_size=args.batch_size,
        callbacks=final_callbacks,
        verbose=1
    )
    
    return final_model, history

def evaluate_model(model, test_data, output_dir):
    """Mengevaluasi model pada test set dan menyimpan hasilnya."""
    X_test, y_test = test_data
    
    print("\n" + "="*50)
    print("MEMULAI TAHAP 3: EVALUASI MODEL")
    print("="*50)
    
    y_pred_probs = model.predict(X_test, batch_size=32)
    y_pred_classes = (y_pred_probs > 0.5).astype("int32").flatten()

    print("\n📝 Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=['benign', 'malicious']))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['benign', 'malicious'], yticklabels=['benign', 'malicious'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

# ======================================================================
# --- FUNGSI MAIN (PENGENDALI UTAMA) ---
# ======================================================================
def main():
    if IN_COLAB:
        # Antarmuka interaktif untuk Google Colab
        print("Menjalankan dalam mode Google Colab.")
        # Menggunakan Namespace untuk meniru objek 'args' dari argparse
        args = argparse.Namespace()
        args.dataset_dir = input("Masukkan path ke direktori dataset (berisi file .npy): ")
        args.output_dir = input("Masukkan path ke direktori output (untuk menyimpan model dan hasil): ")
        args.max_trials = 50
        args.exec_per_trial = 1
        args.tuning_epochs = 50
        args.final_epochs = 100
        args.patience = 10
        args.batch_size = 64
        args.overwrite = input("Mulai tuning baru (hapus hasil lama)? (true/false): ").lower() == 'true'
        
    else:
        # Antarmuka baris perintah (CLI) untuk lingkungan lokal
        parser = argparse.ArgumentParser(description="Skrip untuk tuning dan pelatihan model deteksi webshell.")
        parser.add_argument('--dataset_dir', required=True, help='Path ke direktori dataset (berisi file .npy).')
        parser.add_argument('--output_dir', required=True, help='Path ke direktori output (untuk menyimpan model dan hasil).')
        parser.add_argument('--max_trials', type=int, default=50)
        parser.add_argument('--exec_per_trial', type=int, default=1)
        parser.add_argument('--tuning_epochs', type=int, default=50)
        parser.add_argument('--final_epochs', type=int, default=100)
        parser.add_argument('--patience', type=int, default=10)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--overwrite', action='store_true', help='Mulai tuning baru (menghapus hasil lama).')
        args = parser.parse_args()

    # --- Menjalankan Alur Kerja ---
    # 1. Muat data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(args.dataset_dir)
    
    # 2. Jalankan tuning untuk mendapatkan tuner dan hyperparameter terbaik
    tuner, best_hps = run_tuning(
        data=((X_train, y_train), (X_val, y_val)), 
        args=args
    )
    
    # 3. Latih model final
    final_model, history = train_final_model(
        tuner=tuner, 
        best_hps=best_hps,
        data=((X_train, y_train), (X_val, y_val), (X_test, y_test)),
        args=args
    )
    
    # 4. Evaluasi model final
    evaluate_model(
        model=final_model, 
        test_data=(X_test, y_test),
        output_dir=args.output_dir
    )
    
    print("\n✅ Semua proses selesai!")

if __name__ == "__main__":
    main()