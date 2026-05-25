# ============================================================
# SCRIPT EXPORT DATA DARI GOOGLE COLAB
# Jalankan cell ini di Colab SETELAH semua sel selesai
# Tujuan: download semua file yang dibutuhkan untuk lokal
# ============================================================
import os
import shutil
from google.colab import files

SAVE_DIR = '/content/drive/MyDrive/Colab Notebooks/Sistem temu kembali/Tugas 2 _new/'

# File yang perlu didownload
files_to_download = [
    '1_data_mentah.csv',
    'processed_paper.pkl',
    'thesaurus.pkl',
    'bert_embeddings.pkl',
]

print('📦 Mengecek file yang tersedia...')
for f in files_to_download:
    path = SAVE_DIR + f
    size = os.path.getsize(path) / 1024
    print(f'  ✅ {f:30s} ({size:.1f} KB)')

print('\n⬇️  Mendownload semua file...')
for f in files_to_download:
    path = SAVE_DIR + f
    files.download(path)
    print(f'  ✅ Downloaded: {f}')

print('\n✅ Semua file berhasil didownload!')
print('📁 Pindahkan semua file yang didownload ke folder: search_engine/data/')
