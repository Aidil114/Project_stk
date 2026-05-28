
import os
import pickle
import pandas as pd

from sentence_transformers import SentenceTransformer

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(__file__)

DATA_DIR = os.path.join(BASE_DIR, "data")

# =========================
# LOAD DATA
# =========================
print("⏳ Loading data...")

with open(
    os.path.join(DATA_DIR, "processed_paper.pkl"),
    "rb"
) as f:

    processed_paper = pickle.load(f)

print("✅ Total dokumen:", len(processed_paper))

# =========================
# LOAD MODEL L3
# =========================
print("⏳ Loading model BERT L3...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)


print("✅ Model loaded")

# =========================
# GENERATE EMBEDDINGS
# =========================
print("⏳ Generating embeddings...")

embeddings = model.encode(
    processed_paper,
    convert_to_tensor=True,
    show_progress_bar=True
)

# =========================
# SAVE
# =========================
save_path = os.path.join(
    DATA_DIR,
    "bert_embeddings.pkl"
)

with open(save_path, "wb") as f:
    pickle.dump(embeddings, f)

print("✅ Embeddings berhasil disimpan")
print("📁", save_path)
