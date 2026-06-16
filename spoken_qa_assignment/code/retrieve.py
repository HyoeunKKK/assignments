"""Stage 2: Hybrid retrieval (BM25 + bge-large) fused with RRF.

Usage:
    python retrieve.py --split dev
    python retrieve.py --split release

Reads cached transcripts, ranks documents per question, writes
outputs/{split}_retrieval.json : {qid: [[doc_id, rrf_score], ... topN]}.
"""
import argparse
import json
import os
import re

import config
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _TOK.findall(text.lower())


def load_cache(split, tag):
    with open(os.path.join(config.CACHE_DIR, f"{split}_{tag}.json")) as f:
        return json.load(f)


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def weighted_fuse(bm25_scores, dense_scores, w_dense=config.DENSE_WEIGHT):
    """Min-max normalize each score vector, then weighted sum."""
    return w_dense * minmax(dense_scores) + (1.0 - w_dense) * minmax(bm25_scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(config.SPLITS), required=True)
    ap.add_argument("--topn", type=int, default=config.RETRIEVE_TOPN)
    args = ap.parse_args()

    questions = load_cache(args.split, "questions")
    documents = load_cache(args.split, "documents")
    doc_ids = sorted(documents)
    doc_texts = [documents[d] for d in doc_ids]
    q_ids = sorted(questions)
    q_texts = [questions[q] for q in q_ids]

    # --- BM25 ---
    print("Building BM25 index ...")
    bm25 = BM25Okapi([tokenize(t) for t in doc_texts])

    # --- Dense (bge-large) ---
    print(f"Loading {config.EMBED_MODEL} ...")
    embedder = SentenceTransformer(config.EMBED_MODEL, device="cuda")
    doc_emb = embedder.encode(
        doc_texts, batch_size=32, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=True,
    )
    q_emb = embedder.encode(
        [config.BGE_QUERY_PREFIX + t for t in q_texts], batch_size=32,
        convert_to_tensor=True, normalize_embeddings=True,
        show_progress_bar=True,
    )
    dense_sims = (q_emb @ doc_emb.T).cpu().numpy()  # [Q, D]

    # --- Fuse per question ---
    results = {}
    for qi, qid in enumerate(q_ids):
        bm25_scores = bm25.get_scores(tokenize(q_texts[qi]))
        fused = weighted_fuse(bm25_scores, dense_sims[qi])
        top = np.argsort(-fused)[: args.topn]
        results[qid] = [[doc_ids[i], float(fused[i])] for i in top]

    out = os.path.join(config.OUTPUT_DIR, f"{args.split}_retrieval.json")
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=0)
    print(f"wrote {out} ({len(results)} questions, top{args.topn})")


if __name__ == "__main__":
    main()
