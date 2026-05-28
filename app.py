
"""
SISTEM TEMU KEMBALI INFORMASI
FLASK + TFIDF + BERT HYBRID
FIX RAILWAY FINAL
"""

import os
import re
import pickle
import itertools
import urllib.parse
import urllib.request

import pandas as pd
import numpy as np

from flask import Flask, request, jsonify, render_template

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory,
    StopWordRemover,
    ArrayDictionary
)

# ==========================================
# FLASK
# ==========================================
app = Flask(__name__)

# ==========================================
# PREPROCESSING
# ==========================================
factory_sw = StopWordRemoverFactory()

stop_words = factory_sw.get_stop_words()

dictionary = ArrayDictionary(stop_words)

stopword_remover = StopWordRemover(dictionary)

stemmer = StemmerFactory().create_stemmer()

extra_stopwords = {
    'itu','ini','yang','dan','di','ke','dari','pada',
    'untuk','dengan','adalah','juga','sudah','telah',
    'akan','bisa','ada','tidak','lebih','saat',
    'oleh','karena','bahwa','kami','kita',
    'mereka','nya','kan','lah','pun'
}

# ==========================================
# DATA
# ==========================================
BASE_DIR = os.path.dirname(__file__)

DATA_DIR = os.path.join(BASE_DIR, "data")

print("⏳ Loading data...")

with open(os.path.join(DATA_DIR, "processed_paper.pkl"), "rb") as f:
    processed_paper = pickle.load(f)

with open(os.path.join(DATA_DIR, "thesaurus.pkl"), "rb") as f:
    thesaurus = pickle.load(f)

df_mentah = pd.read_csv(
    os.path.join(DATA_DIR, "1_data_mentah.csv")
)

raw_data = df_mentah.to_dict("records")

print("✅ Data loaded")

# ==========================================
# LOAD BERT SEKALI SAJA
# ==========================================
print("⏳ Loading lightweight BERT...")

from sentence_transformers import SentenceTransformer
from sentence_transformers import util

bert_model = SentenceTransformer(
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    device="cpu"
)

print("⏳ Loading embeddings...")

with open(
    os.path.join(DATA_DIR, "bert_embeddings.pkl"),
    "rb"
) as f:

    bert_embeddings = pickle.load(f)

if hasattr(bert_embeddings, "cpu"):
    bert_embeddings = bert_embeddings.cpu()

print("✅ BERT Ready")

# ==========================================
# CLEAN TEXT
# ==========================================
def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+", " ", text)

    text = re.sub(r"[^a-z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

# ==========================================
# TOKENIZE
# ==========================================
def tokenize(text):

    cleaned = clean_text(text)

    tokens = cleaned.split()

    filtered = [
        t for t in tokens
        if len(t) > 2 and t not in extra_stopwords
    ]

    joined = stopword_remover.remove(
        ' '.join(filtered)
    )

    return [
        t for t in joined.split()
        if len(t) > 2
    ]

# ==========================================
# PREPROCESS QUERY
# ==========================================
def preprocess_query(query_text):

    tokens = tokenize(query_text)

    return [
        stemmer.stem(t)
        for t in tokens
    ]

# ==========================================
# QUERY EXPANSION
# ==========================================
def expand_query(tokens):

    list_synonym = []

    for t in tokens:

        if t in thesaurus:

            list_synonym.append(
                thesaurus[t][:2]
            )

        else:

            list_synonym.append([t])

    expanded = []

    for combo in itertools.product(*list_synonym):

        stemmed = [
            stemmer.stem(w)
            for w in combo
        ]

        expanded.append(
            ' '.join(stemmed)
        )

    return expanded

# ==========================================
# TFIDF
# ==========================================
def search_tfidf(query_tokens, top_n=10):

    vectorizer = TfidfVectorizer(use_idf=True)

    query_str = ' '.join(query_tokens)

    all_docs = [query_str] + processed_paper

    tfidf_mat = vectorizer.fit_transform(all_docs)

    scores = cosine_similarity(
        tfidf_mat[0],
        tfidf_mat[1:]
    ).flatten()

    results = []

    for i, s in enumerate(scores):

        if s > 0:

            results.append({
                'doc_idx': i,
                'score': float(s)
            })

    return sorted(
        results,
        key=lambda x: x['score'],
        reverse=True
    )[:top_n]

# ==========================================
# TFIDF EXPANDED
# ==========================================
def search_tfidf_expanded(tokens, top_n=10):

    expanded = expand_query(tokens)

    score_map = {}

    vectorizer = TfidfVectorizer(use_idf=True)

    for q in expanded:

        all_docs = [q] + processed_paper

        tfidf_mat = vectorizer.fit_transform(all_docs)

        scores = cosine_similarity(
            tfidf_mat[0],
            tfidf_mat[1:]
        ).flatten()

        for i, s in enumerate(scores):

            if s > score_map.get(i, 0):
                score_map[i] = float(s)

    results = []

    for k, v in score_map.items():

        results.append({
            "doc_idx": k,
            "score": v
        })

    return sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )[:top_n]

# ==========================================
# BERT SEARCH
# ==========================================
def search_bert(query_text, top_n=10):

    q_emb = bert_model.encode(
        clean_text(query_text),
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        q_emb,
        bert_embeddings
    )[0]

    scores = scores.cpu().numpy()

    ranked = np.argsort(-scores)

    results = []

    for i in ranked[:top_n]:

        results.append({
            "doc_idx": int(i),
            "score": float(scores[i])
        })

    return results

# ==========================================
# HYBRID
# ==========================================
def hybrid_search(query_text, top_n=10):

    tokens = preprocess_query(query_text)

    tfidf_res = search_tfidf_expanded(
        tokens,
        top_n=len(processed_paper)
    )

    bert_res = search_bert(
        query_text,
        top_n=len(processed_paper)
    )

    tfidf_map = {
        r["doc_idx"]: r["score"]
        for r in tfidf_res
    }

    bert_map = {
        r["doc_idx"]: r["score"]
        for r in bert_res
    }

    combined = []

    for idx in set(tfidf_map) | set(bert_map):

        tfidf_score = tfidf_map.get(idx, 0)

        bert_score = bert_map.get(idx, 0)

        final_score = (
            0.5 * bert_score
            +
            0.5 * tfidf_score
        )

        combined.append({

            "doc_idx": idx,

            "score": float(final_score),

            "bert_score": float(bert_score),

            "tfidf_score": float(tfidf_score)
        })

    return sorted(
        combined,
        key=lambda x: x["score"],
        reverse=True
    )[:top_n]

# ==========================================
# FORMAT RESULT
# ==========================================
def format_results(results):

    output = []

    for rank, r in enumerate(results, 1):

        idx = r["doc_idx"]

        doc = raw_data[idx]

        output.append({

            "rank": rank,

            "score": round(r["score"], 4),

            "judul": doc.get("judul", ""),

            "url": doc.get("url", ""),

            "isi": str(
                doc.get("isi", "")
            )[:300] + "..."
        })

    return output

# ==========================================
# HOME
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

# ==========================================
# SEARCH API
# ==========================================
@app.route('/api/search', methods=['POST'])
def search():

    try:

        data = request.get_json()

        query = data.get("query", "").strip()

        method = data.get("method", "hybrid")

        top_n = int(data.get("top_n", 5))

        if not query:

            return jsonify({
                "error": "query kosong"
            }), 400

        tokens = preprocess_query(query)

        if method == "tfidf":

            results = search_tfidf(
                tokens,
                top_n
            )

        elif method == "tfidf_expanded":

            results = search_tfidf_expanded(
                tokens,
                top_n
            )

        elif method == "bert":

            results = search_bert(
                query,
                top_n
            )

        else:

            results = hybrid_search(
                query,
                top_n
            )

        return jsonify({

            "query": query,

            "method": method,

            "tokens": tokens,

            "total": len(results),

            "results": format_results(results)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ==========================================
# STATS
# ==========================================
@app.route('/api/stats')
def stats():

    return jsonify({

        "total_dokumen": len(raw_data),

        "bert_model":
            "paraphrase-MiniLM-L3-v2"
    })

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )