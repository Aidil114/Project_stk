# Sistem Temu Kembali Informasi
BERT + TF-IDF + Query Expansion | 50 Artikel tvonenews.com

---

## 📁 Struktur Folder

```
search_engine/
├── app.py                  ← Backend Flask (API)
├── requirements.txt        ← Daftar library
├── vercel.json             ← Konfigurasi Vercel
├── export_dari_colab.py    ← Script download data dari Colab
├── data/                   ← Folder data (isi dari Colab)
│   ├── 1_data_mentah.csv
│   ├── processed_paper.pkl
│   ├── thesaurus.pkl
│   └── bert_embeddings.pkl
└── templates/
    └── index.html          ← Tampilan web
```

---

## 🚀 LANGKAH 1 — Export Data dari Google Colab

1. Buka Google Colab kamu
2. Buat cell baru, copy-paste isi file `export_dari_colab.py`
3. Jalankan cell tersebut
4. 4 file akan otomatis terdownload ke komputer kamu:
   - `1_data_mentah.csv`
   - `processed_paper.pkl`
   - `thesaurus.pkl`
   - `bert_embeddings.pkl`
5. **Pindahkan semua file tersebut ke folder `search_engine/data/`**

---

## 💻 LANGKAH 2 — Jalankan di Komputer Lokal

### Install Python
Pastikan Python 3.10+ sudah terinstall:
```
python --version
```

### Buat Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Install Library
```bash
pip install -r requirements.txt
```

### Jalankan Server
```bash
python app.py
```

### Buka di Browser
```
http://localhost:5000
```

---

## 🌐 LANGKAH 3 — Deploy ke Vercel

### Install Vercel CLI
```bash
npm install -g vercel
```

### Login ke Vercel
```bash
vercel login
```
Pilih login dengan GitHub/Email, lalu verifikasi di browser.

### Deploy
```bash
# Masuk ke folder project
cd search_engine

# Deploy
vercel

# Ikuti instruksi:
# ? Set up and deploy? → Y
# ? Which scope? → pilih akun kamu
# ? Link to existing project? → N
# ? What's your project's name? → search-engine-stki
# ? In which directory is your code located? → ./
# → Tunggu sampai selesai
```

### Deploy ke Production
```bash
vercel --prod
```

Setelah selesai, kamu akan mendapat URL seperti:
```
https://search-engine-stki.vercel.app
```

---

## ⚠️ CATATAN PENTING untuk Vercel

Vercel memiliki **batas ukuran file 50MB**. File `bert_embeddings.pkl`
dan model BERT bisa melebihi batas ini.

### Solusi jika file terlalu besar:
Gunakan **Railway** atau **Render** sebagai alternatif Vercel yang
mendukung file lebih besar:

**Railway (Recommended):**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy
railway init
railway up
```

**Render:**
1. Buat akun di https://render.com
2. New → Web Service
3. Connect GitHub repo kamu
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `gunicorn app:app`

---

## 🧪 Test API Manual

```bash
# Test pencarian
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "bandara penerbangan", "method": "hybrid", "top_n": 5}'

# Test stats
curl http://localhost:5000/api/stats
```
