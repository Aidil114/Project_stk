"""
SISTEM TEMU KEMBALI INFORMASI
Backend API - Flask
"""

import os

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import re
import string
import itertools
import os
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory, StopWordRemover, ArrayDictionary
)
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

# ── Inisialisasi tools preprocessing ────────────────────────
factory_sw       = StopWordRemoverFactory()
stop_words       = factory_sw.get_stop_words()
dictionary       = ArrayDictionary(stop_words)
stopword_remover = StopWordRemover(dictionary)
stemmer          = StemmerFactory().create_stemmer()

extra_stopwords = {
    'itu','ini','yang','dan','di','ke','dari','pada','untuk','dengan',
    'adalah','juga','sudah','telah','akan','bisa','ada','tidak','lebih',
    'saat','oleh','karena','bahwa','kami','kita','mereka','nya','kan',
    'lah','pun','tapi','atau','jika','bila','serta','hingga','antara',
    'yakni','yaitu','tersebut','seperti','menjadi','dalam','namun',
    'sehingga','ketika','setelah','sebelum','agar','sangat','hanya',
    'masih','sedang','baru','semua','para','atas','bawah','lain',
    'sebuah','seorang','setiap','pula','juga','hari','tahun','kata',
}

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# ── Load semua data & model ──────────────────────────────────
print("⏳ Memuat data dan model...")

with open(os.path.join(DATA_DIR, 'processed_paper.pkl'), 'rb') as f:
    processed_paper = pickle.load(f)

with open(os.path.join(DATA_DIR, 'thesaurus.pkl'), 'rb') as f:
    thesaurus = pickle.load(f)



df_mentah = pd.read_csv(os.path.join(DATA_DIR, '1_data_mentah.csv'))
raw_data  = df_mentah.to_dict('records')

print("✅ Data berhasil dimuat!")

bert_model = None
util = None
bert_embeddings = None


def load_bert():

    global bert_model
    global util
    global bert_embeddings

    if bert_model is not None:
        return

    try:

        print("⏳ Memuat model BERT...")

        import torch

        torch.set_num_threads(1)

        from sentence_transformers import (
            SentenceTransformer,
            util as st_util
        )

        bert_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device="cpu"
        )

        util = st_util

        print("⏳ Memuat embedding...")

        with open(
            os.path.join(
                DATA_DIR,
                "bert_embeddings.pkl"
            ),
            "rb"
        ) as f:

            bert_embeddings = pickle.load(f)

        if hasattr(
            bert_embeddings,
            "cpu"
        ):
            bert_embeddings = bert_embeddings.cpu()

        print("✅ BERT siap")

    except Exception as e:

        print("❌ BERT gagal:", e)

        bert_model = None
        bert_embeddings = None


# ── Fungsi preprocessing ────────────────────────────────────
def clean_text(text):
    text = str(text)
    text = re.sub(r'ADVERTISEMENT GULIR UNTUK LANJUT BACA', '', text)
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    cleaned  = clean_text(text)
    tokens   = word_tokenize(cleaned)
    filtered = [t for t in tokens if len(t) > 2 and t not in extra_stopwords]
    joined   = stopword_remover.remove(' '.join(filtered))
    return [t for t in joined.split() if len(t) > 2]

def preprocess_query(query_text):
    tokens = tokenize(query_text)
    return [stemmer.stem(t) for t in tokens]

def get_synonym(word):
    try:
        data         = {'q': word}
        encoded_data = urllib.parse.urlencode(data).encode('utf-8')
        content      = urllib.request.urlopen(
                           'http://www.sinonimkata.com/search.php',
                           encoded_data, timeout=5)
        soup         = BeautifulSoup(content, 'html.parser')
        synonym_tags = soup.find('td', attrs={'width': '90%'}).find_all('a')
        return [word] + [tag.getText() for tag in synonym_tags]
    except Exception:
        return [word]

def expand_query(tokens):
    list_synonym = []
    for t in tokens:
        if t in thesaurus:
            list_synonym.append(thesaurus[t][:3])
        else:
            syn = get_synonym(t)
            thesaurus[t] = syn
            list_synonym.append(syn[:3])
    expanded = []
    for combo in itertools.product(*list_synonym):
        stemmed = [stemmer.stem(w) for w in combo]
        expanded.append(' '.join(stemmed))
    return expanded


# ── Fungsi pencarian ────────────────────────────────────────
def search_tfidf(query_tokens, top_n=10):
    vectorizer = TfidfVectorizer(use_idf=True)
    query_str  = ' '.join(query_tokens)
    all_docs   = [query_str] + processed_paper
    tfidf_mat  = vectorizer.fit_transform(all_docs)
    scores     = cosine_similarity(tfidf_mat[0], tfidf_mat[1:]).flatten()
    results = [{'doc_idx': i, 'score': float(s), 'query': query_str}
               for i, s in enumerate(scores) if s > 0]
    return sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]

def search_tfidf_expanded(tokens, top_n=10):
    expanded   = expand_query(tokens)
    vectorizer = TfidfVectorizer(use_idf=True)
    score_map  = {}
    best_query = {}
    for q_str in expanded:
        all_docs  = [q_str] + processed_paper
        tfidf_mat = vectorizer.fit_transform(all_docs)
        scores    = cosine_similarity(tfidf_mat[0], tfidf_mat[1:]).flatten()
        for i, s in enumerate(scores):
            if s > score_map.get(i, 0):
                score_map[i]  = float(s)
                best_query[i] = q_str
    results = [{'doc_idx': k, 'score': v, 'query': best_query[k]}
               for k, v in score_map.items() if v > 0]
    return sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]

def search_bert(query_text, top_n=10):
    
    global bert_model
    global util
    global bert_embeddings

    load_bert()

    if bert_model is None:
        return []

    if bert_embeddings is None:
        return []

    q_emb = bert_model.encode(
        clean_text(query_text),
        convert_to_tensor=False
    )

    scores = util.cos_sim(
    q_emb,
    bert_embeddings
)[0]

    if hasattr(scores, "numpy"):
        scores = scores.numpy()

    ranked = np.argsort(-scores)

    return [
        {
            "doc_idx": int(i),
            "score": float(scores[i]),
            "query": query_text
        }
        for i in ranked[:top_n]
    ]
    

def hybrid_search(query_text, top_n=10, alpha=0.5):
    
    tokens = preprocess_query(query_text)

    tfidf_res = search_tfidf_expanded(
        tokens,
        top_n=len(processed_paper)
    )

    try:
        bert_res = search_bert(
            query_text,
            top_n=len(processed_paper)
        )
    except:
        bert_res = []

    if not bert_res:
        return tfidf_res[:top_n]

    tfidf_map = {
        r["doc_idx"]: r["score"]
        for r in tfidf_res
    }

    bert_map = {
        r["doc_idx"]: r["score"]
        for r in bert_res
    }

    max_tfidf = max(
        tfidf_map.values(),
        default=1
    )

    max_bert = max(
        bert_map.values(),
        default=1
    )

    combined = []

    for idx in set(tfidf_map) | set(bert_map):

        s_tfidf = (
            tfidf_map.get(idx, 0)
            / max_tfidf
        )

        s_bert = (
            bert_map.get(idx, 0)
            / max_bert
        )

        combined.append({
            "doc_idx": idx,
            "score":
                alpha*s_bert
                +
                (1-alpha)*s_tfidf,

            "bert_score":
                round(s_bert,4),

            "tfidf_score":
                round(s_tfidf,4),

            "query": query_text
        })

    return sorted(
        combined,
        key=lambda x: x["score"],
        reverse=True
    )[:top_n]

def format_results(results, method='hybrid'):
    output = []
    for rank, r in enumerate(results, 1):
        idx = r['doc_idx']
        doc = raw_data[idx]
        item = {
            'rank'   : rank,
            'score'  : round(r['score'], 4),
            'judul'  : doc.get('judul', ''),
            'url'    : doc.get('url', ''),
            'isi'    : str(doc.get('isi', ''))[:300] + '...',
            'no'     : doc.get('no', idx + 1),
        }
        if method == 'hybrid':
            item['bert_score']  = r.get('bert_score', 0)
            item['tfidf_score'] = r.get('tfidf_score', 0)
        output.append(item)
    return output


# ── Routes ───────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    data   = request.get_json()
    query  = data.get('query', '').strip()
    method = data.get('method', 'hybrid')
    top_n  = int(data.get('top_n', 5))

    if not query:
        return jsonify({'error': 'Query tidak boleh kosong'}), 400

    tokens = preprocess_query(query)

    if method == 'tfidf':
        results = search_tfidf(tokens, top_n=top_n)
    elif method == 'tfidf_expanded':
        results = search_tfidf_expanded(tokens, top_n=top_n)
    elif method == 'bert':
        results = search_bert(query, top_n=top_n)
    else:
        results = hybrid_search(query, top_n=top_n)

    return jsonify({
        'query'          : query,
        'method'         : method,
        'tokens'         : tokens,
        'total'          : len(results),
        'results'        : format_results(results, method),
    })

@app.route('/api/stats')
def stats():
    return jsonify({
        'total_dokumen' : len(raw_data),
        'total_kata'    : len(thesaurus),
        'bert_model'    : 'paraphrase-multilingual-MiniLM-L3-v2',
    })

if __name__ == "__main__":
    app.run()