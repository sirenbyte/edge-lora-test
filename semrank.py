"""Semantic ranking bridge — run with SYSTEM python3 (sentence-transformers, ru/en).

  echo '{"query": "...", "texts": ["...", ...], "k": 6}' | python3 semrank.py
  -> stdout: JSON [{"i": <idx>, "score": <cosine>}, ...]  top-k, descending.

Used by hobby_pack.py for (a) selecting tip-bearing passages WITHIN articles
(intro/boilerplate ranks low → dropped by relevance, not position) and
(b) semantic retrieval ACROSS a pack. Mirrors mp_bridge's multilingual encoder.
"""
import json
import sys

ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    req = json.load(sys.stdin)
    query, texts, k = req["query"], req.get("texts", []), int(req.get("k", 6))
    if not texts:
        print("[]")
        return
    from sentence_transformers import SentenceTransformer
    import numpy as np
    enc = SentenceTransformer(ENCODER)
    embs = enc.encode([query] + list(texts), normalize_embeddings=True,
                      convert_to_numpy=True)
    sims = embs[1:] @ embs[0]                      # cosine (normalized)
    order = np.argsort(-sims)[:k]
    print(json.dumps([{"i": int(i), "score": float(sims[i])} for i in order]))


if __name__ == "__main__":
    main()
