"""Bridge between the agent (mlx venv) and Python Memory Plant (system python3
with torch + sentence-transformers). Run with SYSTEM python3, not the mlx venv.

  python3 mp_bridge.py ingest "<text>"   # store a memory
  python3 mp_bridge.py search "<query>"  # print top-K relevant memories (1/line)

Persists stored texts to mp_store.json; rebuilds UnifiedMemory per call (fine for
a prototype with few memories — real deploy = a persistent mp service / Rust core).
Uses a MULTILINGUAL encoder (ru/en). Doc-based retrieval (add_document + semantic_search).
"""
import json
import os
import sys

STORE = os.path.expanduser("~/projects/edge-lora-test/mp_store.json")
MP_DIR = os.path.expanduser("~/projects/memory-plant")
ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
USER = "demo"


def load_texts():
    return json.load(open(STORE)) if os.path.exists(STORE) else []


def save_texts(t):
    json.dump(t, open(STORE, "w"), ensure_ascii=False)


def build_mem(texts):
    sys.path.insert(0, MP_DIR)
    from unified import UnifiedMemory
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER)
    um = UnifiedMemory(USER, encoder=enc)
    for i, t in enumerate(texts):
        um.add_document(f"d{i}", t)
    return um


def main():
    cmd, arg = sys.argv[1], sys.argv[2]
    if cmd == "ingest":
        t = load_texts()
        t.append(arg)
        save_texts(t)
        print("stored")
    elif cmd == "search":
        texts = load_texts()
        if not texts:
            return
        um = build_mem(texts)
        for doc_id, score, text in um.semantic_search(arg, k=3):
            print(f"{text}")


if __name__ == "__main__":
    main()
