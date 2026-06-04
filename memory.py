"""Unified on-device memory for Qiyas Edge — ONE coherent in-process stack.

Backend: `memory_plant_rs` (Rust core, in-process — NO system-python subprocess):
  * facts      → PersonalMemory  (HLB, provable forget; no embeddings)
  * documents  → DocumentMemory  (precomputed e5 vectors) + chunk_text
Embedder: `embed.py` (multilingual-e5-small via MLX) — the Qwen side owns vectors.

This implements E5_INTEGRATION.md and replaces the stopgaps: `mp_bridge.py`
(system-python MiniLM subprocess) and `semrank.py`. Documents use e5 (cross-lingual,
language-agnostic) so Russian recall works without fact extraction.
"""
from __future__ import annotations

import json
from pathlib import Path

import memory_plant_rs as mp

from embed import DIM, embed

DOC_STORE = "/Users/abzaltuganbay/projects/edge-lora-test/mp_docs.bin"
FACTS_STORE = "/Users/abzaltuganbay/projects/edge-lora-test/mp_facts.json"
USER = "demo"

# render a structured fact as a 2nd-person sentence — so RECALL echoes the right
# person ("тебя зовут…", not "меня зовут…") regardless of how it was phrased.
_PRED_RU = {
    "name": "Тебя зовут {v}.", "age": "Тебе {v}.", "weight": "Ты весишь {v}.",
    "height": "Твой рост — {v}.", "city": "Ты живёшь в городе {v}.",
    "country": "Ты живёшь в {v}.", "job": "Ты работаешь: {v}.",
    "hobby": "Твоё хобби — {v}.", "likes": "Тебе нравится {v}.", "pet": "У тебя есть {v}.",
    "email": "Твой email — {v}.", "phone": "Твой телефон — {v}.", "goal": "Твоя цель — {v}.",
}


def normalize_fact(predicate: str, value: str) -> str:
    return _PRED_RU.get(predicate, "Про тебя: {p} — {v}.").format(v=value, p=predicate)


class Memory:
    def __init__(self, user: str = USER, doc_path: str = DOC_STORE, dim: int = DIM,
                 facts_path: str = FACTS_STORE):
        self.dim = dim
        self.facts = mp.PersonalMemory(user)            # HLB facts (provable forget)
        self._doc_path = Path(doc_path)
        if self._doc_path.exists():
            try:
                self.docs = mp.DocumentMemory.load(str(self._doc_path), dim)
            except Exception:
                self.docs = mp.DocumentMemory(dim)
        else:
            self.docs = mp.DocumentMemory(dim)
        # structured facts: JSON is the persisted source of truth (HLB isn't saved
        # by the bindings yet); mirror into PersonalMemory for forget/HLB.
        self._facts_path = Path(facts_path)
        self._kv = {}
        if self._facts_path.exists():
            try:
                self._kv = json.loads(self._facts_path.read_text(encoding="utf-8"))
            except Exception:
                self._kv = {}
        for p, v in self._kv.items():
            try:
                self.facts.store_fact(p, v)
            except Exception:
                pass

    # ---------- facts (HLB, no e5) ----------
    def remember(self, message: str):
        """Extract + store facts from natural language (offline regex extractor)."""
        return self.facts.ingest(message)

    def store_fact(self, predicate: str, value: str, subject: str = "user"):
        self.facts.store_fact(predicate, value, subject)

    def remember_fact(self, predicate: str, value: str) -> str:
        """Store a structured fact (overwrites prior value) + persist to JSON.
        Returns the 2nd-person rendering used for the acknowledgement."""
        self.facts.store_fact(predicate, value, "user")
        self._kv[predicate] = value
        self._facts_path.write_text(json.dumps(self._kv, ensure_ascii=False), encoding="utf-8")
        return normalize_fact(predicate, value)

    def render_facts(self) -> str:
        """All current facts as 2nd-person lines (source for self-question recall)."""
        return "\n".join(normalize_fact(p, v) for p, v in self._kv.items())

    def known_predicates(self) -> list[str]:
        return list(self._kv.keys())

    def fact_line(self, predicate: str) -> str | None:
        """Stored fact rendered 2nd-person, VERBATIM. None if not stored."""
        return normalize_fact(predicate, self._kv[predicate]) if predicate in self._kv else None

    def recall(self, predicate: str, subject: str | None = None):
        return self.facts.recall(predicate, subject)

    def all_facts(self) -> dict:
        return self.facts.all_facts()

    def forget(self, predicate: str, subject: str | None = None) -> bool:
        return self.facts.forget(predicate, subject)

    # ---------- documents (e5 RAG) ----------
    def store_document(self, doc_id: str, text: str, metadata: dict | None = None,
                       chunk_size: int = 200, overlap: int = 20) -> int:
        """Chunk (Rust) → embed passages (e5) → store. Returns #chunks stored."""
        chunks = mp.chunk_text(text, chunk_size, overlap)
        if not chunks:
            return 0
        embs = embed(chunks, mode="passage")
        return self.docs.add_document(doc_id, chunks, embs, metadata or {})

    def retrieve(self, query: str, k: int = 5, min_score: float = 0.3) -> list[dict]:
        """Embed query (e5) → cosine search. Returns [{chunk_id, doc_id, score, text}]."""
        q = embed([query], mode="query")[0]
        return self.docs.search(q, k=k, min_score=min_score)

    def retrieve_texts(self, query: str, k: int = 5, min_score: float = 0.3) -> list[str]:
        return [h["text"] for h in self.retrieve(query, k=k, min_score=min_score)]

    def save(self):
        self.docs.save(str(self._doc_path))


if __name__ == "__main__":
    m = Memory(doc_path="/tmp/mp_docs_test.bin")
    # facts
    m.store_fact("name", "Абзал")
    print("recall(name):", m.recall("name"))
    # documents (e5 RAG, cross-lingual)
    m.store_document("d1", "Роман «Война и мир» написал Лев Толстой. Это эпопея о войне 1812 года.")
    m.store_document("d2", "The mitochondria is the powerhouse of the cell. It makes ATP.")
    m.store_document("d3", "Чтобы сварить борщ, нужны свёкла, капуста, мясо и немного терпения.")
    hits = m.retrieve("Кто автор Войны и мира?", k=2)
    print("retrieve:")
    for h in hits:
        print(f"  [{h['doc_id']} {h['score']:.3f}] {h['text'][:60]}")
    print("top doc == d1:", hits and hits[0]["doc_id"] == "d1")
