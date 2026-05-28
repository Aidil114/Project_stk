"""
SISTEM TEMU KEMBALI INFORMASI
Backend API - Flask
FIX RAILWAY + BERT + NLTK
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

import nltk

# ==========================================
# DOWNLOAD NLTK
# ==========================================
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    print("NLTK download gagal")

# ========================================== # SIMPLE TOKENIZER # ========================================== # NLTK DIHAPUS AGAR STABIL DI RAILWAY

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
    'itu','ini','yang','dan','di','ke','dari','pada','untuk',
    'dengan','adalah','juga','sudah','telah','akan','bisa',
    'ada','tidak','lebih','saat','oleh','karena','bahwa',
    'kami','kita','mereka','nya','kan','lah','pun','tapi',
    'atau','jika','bila','serta','hingga','antara','yakni',
    'yaitu','tersebut','seperti','menjadi','dalam','namun',
    'sehingga','ketika','setelah','sebelum','agar','sangat',
    'hanya','masih','sedang','baru','semua','para','atas',
    'bawah','lain','sebuah','seorang','setiap','pula',
    'hari','tahun','kata',
}

# ==========================================
# DATA
# ==========================================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

print("⏳ Memuat data...")

with open(os.path.join(DATA_DIR, "processed_paper.pkl"), "rb") as f:
    processed_paper = pickle.load(f)

with open(os.path.join(DATA_DIR, "thesaurus.pkl"), "rb") as f:
    thesaurus = pickle.load(f)

df_mentah = pd.read_csv(
    os.path.join(DATA_DIR, "1_data_mentah.csv")
)

raw_data = df_mentah.to_dict("records")

print("✅ Data berhasil dimuat")

# ==========================================
# GLOBAL BERT
# ==========================================
bert_model = None
util = None
bert_embeddings = None

# ==========================================
# LOAD BERT
# ==========================================
def load_bert():

    global bert_model
    global util
    global bert_embeddings

    if bert_model is not None:
        return

    try:

        print("⏳ Loading BERT...")

        import torch

        torch.set_num_threads(1)

        from sentence_transformers import (
            SentenceTransformer,
            util as st_util
        )

        bert_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            device="cpu"
        )

        util = st_util

        print("⏳ Loading embeddings...")

        with open(
            os.path.join(DATA_DIR, "bert_embeddings.pkl"),
            "rb"
        ) as f:

            bert_embeddings = pickle.load(f)

        if hasattr(bert_embeddings, "cpu"):
            bert_embeddings = bert_embeddings.cpu()

        print("✅ BERT Loaded")

    except Exception as e:

        print("❌ BERT ERROR:", e)

        bert_model = None
        bert_embeddings = None

# ==========================================
# CLEAN TEXT
# ==========================================
def clean_text(text):

    text = str(text)

    text = re.sub(
        r'ADVERTISEMENT GULIR UNTUK LANJUT BACA',
        '',
        text
    )

    text = text.lower()

    text = re.sub(r'http\S+', ' ', text)

    text = re.sub(r'[^a-z\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ==========================================
# TOKENIZE
# ==========================================
def tokenize(text):

    cleaned = clean_text(text)

    # tokenizer manual tanpa nltk
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
# GET SYNONYM
# ==========================================
def get_synonym(word):

    try:

        data = {'q': word}

        encoded_data = urllib.parse.urlencode(data).encode('utf-8')

        content = urllib.request.urlopen(
            'http://www.sinonimkata.com/search.php',
            encoded_data,
            timeout=3
        )

        soup = BeautifulSoup(content, 'html.parser')

        td = soup.find(
            'td',
            attrs={'width': '90%'}
        )

        if td is None:
            return [word]

        synonym_tags = td.find_all('a')

        return [word] + [
            tag.getText()
            for tag in synonym_tags
        ]

    except:
        return [word]

# ==========================================
# QUERY EXPANSION
# ==========================================
def expand_query(tokens):

    list_synonym = []

    for t in tokens:

        if t in thesaurus:

            list_synonym.append(
                thesaurus[t][:3]
            )

        else:

            syn = get_synonym(t)

            thesaurus[t] = syn

            list_synonym.append(
                syn[:3]
            )

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
# TFIDF SEARCH
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
                'score': float(s),
                'query': query_str
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

    vectorizer = TfidfVectorizer(use_idf=True)

    score_map = {}
    best_query = {}

    for q_str in expanded:

        all_docs = [q_str] + processed_paper

        tfidf_mat = vectorizer.fit_transform(all_docs)

        scores = cosine_similarity(
            tfidf_mat[0],
            tfidf_mat[1:]
        ).flatten()

        for i, s in enumerate(scores):

            if s > score_map.get(i, 0):

                score_map[i] = float(s)

                best_query[i] = q_str

    results = []

    for k, v in score_map.items():

        if v > 0:

            results.append({
                'doc_idx': k,
                'score': v,
                'query': best_query[k]
            })

    return sorted(
        results,
        key=lambda x: x['score'],
        reverse=True
    )[:top_n]

# ==========================================
# BERT SEARCH
# ==========================================
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
            "score": float(scores[i]),
            "query": query_text
        })

    return results

# ==========================================
# HYBRID SEARCH
# ==========================================
def hybrid_search(query_text, top_n=10, alpha=0.5):

    tokens = preprocess_query(query_text)

    tfidf_res = search_tfidf_expanded(
        tokens,
        top_n=len(processed_paper)
    )

    bert_res = search_bert(
        query_text,
        top_n=len(processed_paper)
    )

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

        final_score = (
            alpha * s_bert
            +
            (1 - alpha) * s_tfidf
        )

        combined.append({

            "doc_idx": idx,

            "score": round(final_score, 4),

            "bert_score": round(s_bert, 4),

            "tfidf_score": round(s_tfidf, 4),

            "query": query_text
        })

    return sorted(
        combined,
        key=lambda x: x["score"],
        reverse=True
    )[:top_n]

# ==========================================
# FORMAT RESULT
# ==========================================
def format_results(results, method='hybrid'):

    output = []

    for rank, r in enumerate(results, 1):

        idx = r['doc_idx']

        doc = raw_data[idx]

        item = {

            'rank': rank,

            'score': round(r['score'], 4),

            'judul': doc.get('judul', ''),

            'url': doc.get('url', ''),

            'isi': str(
                doc.get('isi', '')
            )[:300] + '...',

            'no': doc.get('no', idx + 1),
        }

        if method == 'hybrid':

            item['bert_score'] = r.get(
                'bert_score',
                0
            )

            item['tfidf_score'] = r.get(
                'tfidf_score',
                0
            )

        output.append(item)

    return output

# ==========================================
# HOME
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

# ==========================================
# API SEARCH
# ==========================================
@app.route('/api/search', methods=['POST'])
def search():

    try:

        data = request.get_json()

        query = data.get('query', '').strip()

        method = data.get('method', 'hybrid')

        top_n = int(
            data.get('top_n', 5)
        )

        if not query:

            return jsonify({
                'error': 'Query kosong'
            }), 400

        tokens = preprocess_query(query)

        if method == 'tfidf':

            results = search_tfidf(
                tokens,
                top_n=top_n
            )

        elif method == 'tfidf_expanded':

            results = search_tfidf_expanded(
                tokens,
                top_n=top_n
            )

        elif method == 'bert':

            results = search_bert(
                query,
                top_n=top_n
            )

        else:

            results = hybrid_search(
                query,
                top_n=top_n
            )

        return jsonify({

            'query': query,

            'method': method,

            'tokens': tokens,

            'total': len(results),

            'results': format_results(
                results,
                method
            ),
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

        'total_dokumen': len(raw_data),

        'total_kata': len(thesaurus),

        'bert_model':
            'paraphrase-multilingual-MiniLM-L12-v2',
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
