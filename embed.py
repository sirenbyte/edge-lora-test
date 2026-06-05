"""Qwen-side embedder for Memory Plant's document/RAG layer (see E5_INTEGRATION.md).

Memory Plant STORES + SEARCHES vectors; it does NOT embed. Computing the vector is
the Qwen side's job — this module runs `multilingual-e5-small` through MLX (the
SAME runtime as Qwen, no ONNX/torch) and returns L2-normalized 384-d vectors.

Invariants (violate these and retrieval silently degrades):
  * mode="passage" for stored chunks, mode="query" for searches — ALWAYS
    (e5 was trained with these literal prefixes).
  * output is L2-normalized → cosine == dot product downstream.
  * dim == 384, ONE model for write AND read. Change the model → REINDEX all.

Facts (store_fact / ingest / recall) bypass this entirely — they are HLB, no e5.
"""
from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
from mlx_embeddings.utils import load

from config import E5_MODEL as MODEL, E5_DIM as DIM

_PREFIX = {"query": "query: ", "passage": "passage: "}


@lru_cache(maxsize=1)
def _model():
    return load(MODEL)                       # (model, tokenizer), cached for reuse


def _l2(m: "mx.array") -> "mx.array":        # (B, D) -> row L2-normalized
    n = mx.sqrt(mx.sum(m * m, axis=1, keepdims=True))
    return m / mx.maximum(n, 1e-12)


def embed(texts, mode: str = "passage") -> list[list[float]]:
    """texts: str | list[str]. mode: 'query' (searches) | 'passage' (stored docs).
    Returns a list of L2-normalized 384-d float vectors, one per input text."""
    if mode not in _PREFIX:
        raise ValueError(f"mode must be 'query' or 'passage', got {mode!r}")
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []
    model, tok = _model()
    prefixed = [_PREFIX[mode] + t for t in texts]
    inp = tok.batch_encode_plus(prefixed, return_tensors="mlx", padding=True,
                                truncation=True, max_length=512)
    out = model(inp["input_ids"], attention_mask=inp["attention_mask"])
    emb = getattr(out, "text_embeds", None)          # mlx-embeddings: pooled vector
    if emb is None:                                   # fallback: mean-pool tokens
        lhs = out.last_hidden_state
        mask = inp["attention_mask"][..., None]
        emb = (lhs * mask).sum(axis=1) / mx.maximum(mask.sum(axis=1), 1)
    return _l2(emb).tolist()


def cosine(a, b) -> float:
    """Both inputs are already L2-normalized → cosine == dot product."""
    return sum(x * y for x, y in zip(a, b))


if __name__ == "__main__":
    docs = ["Роман «Война и мир» написал Лев Толстой.",
            "The mitochondria is the powerhouse of the cell.",
            "Чтобы сварить борщ, нужны свёкла, капуста и мясо."]
    vecs = embed(docs, mode="passage")
    q = embed("Кто написал роман Война и мир?", mode="query")[0]
    print(f"dim={len(q)} (expect {DIM}); passages={len(vecs)}")
    ranked = sorted(((round(cosine(q, v), 3), t) for v, t in zip(vecs, docs)), reverse=True)
    for s, t in ranked:
        print(f"  {s:>6}  {t[:48]}")
    print("top == Tolstoy:", ranked[0][1].startswith("Роман"))
