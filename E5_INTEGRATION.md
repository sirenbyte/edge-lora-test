# E5 ↔ Memory Plant integration contract (Qwen side)

**Owner:** the model layer (`edge-lora-test` now; the app later).
**Counterpart:** Memory Plant Rust core (`memory_plant_rs`, repo `sirenbyte/memory-plant-rs`).
**Thesis:** Memory Plant does NOT embed. The Qwen side owns the embedder, computes
vectors, and hands them to MP. MP stores, searches (cosine + filters), and provably
forgets. This decoupling is the point — swap the embedder without touching MP.

---

## Division of responsibility

| Qwen side (this repo) | Memory Plant (Rust) |
|---|---|
| run e5 (and Qwen) on **one** runtime (MLX / llama.cpp) | accept a vector, **store + search** |
| `embed(text) -> 384-d vector` | cosine + metadata filters + provable forget |
| orchestrate the RAG loop | chunking (`chunk_text`), compression, persistence |
| enforce: **prefixes, L2-norm, one model** | dim-agnostic vector store |

---

## 1. The embedder — `embed.py` (DONE)

```python
from embed import embed            # multilingual-e5-small via MLX, 384-d
q  = embed("Кто написал Войну и мир?", mode="query")    # -> [[f32; 384]]
ps = embed(list_of_chunks,           mode="passage")    # -> [[f32; 384], ...]
```

**Invariants (non-negotiable):**
- `mode="passage"` for stored chunks, `mode="query"` for searches — ALWAYS. e5 was
  trained with these literal prefixes; mixing them wrecks ranking.
- Output is **L2-normalized** → cosine == dot product.
- **dim == 384**, the **same** model for write and read. Change the model → **reindex** everything.

Facts go through a **different** path — see §4.

## 2. Store a document (RAG)

```python
chunks = MP.chunk_text(file_text, 200, 20)        # chunking lives in Rust (pure)
embs   = embed(chunks, mode="passage")            # ← Qwen side
MP.add_document_with_embeddings(doc_id, chunks, embs, metadata)   # precomputed path
```

## 3. Retrieve by meaning (then feed Qwen)

```python
q_emb = embed([query], mode="query")[0]           # ← Qwen side
hits  = MP.search(q_emb, k=5, min_score=0.3)      # -> [SearchHit{chunk_id, doc_id, score, text}]
context = [h.text for h in hits]
```

## 4. RAG loop with Qwen

```python
def on_user_message(msg):
    context = retrieve(msg)            # §3 — semantic search over documents (e5)
    facts   = MP.all_facts()           # personal facts — HLB, NO embeddings
    answer  = Qwen.generate(build(msg, context, facts))   # + tool-calls
    MP.ingest(msg)                     # remember new facts (HLB, no e5)
```

> **Facts bypass e5.** `store_fact / ingest / recall` are HLB/VSA — vectors not needed.
> e5 is for **documents** (RAG) only.

---

## ✅ UNIFIED — in-process, one runtime (2026-06-04)

`embed.py` (e5/MLX) + `memory_plant_rs` (Rust core) are joined in **`memory.py`** —
NO system-python subprocess. `agent.py` retrieve/ingest run through it in-process.

The Rust **document/RAG layer is now PyO3-bound** (added this session in
`memory-plant-rs/python-bindings/src/lib.rs`):
- `memory_plant_rs.DocumentMemory(dim=384)` — `add_document(doc_id, chunks, embeddings, metadata=None)`,
  `search(query_emb, k=5, min_score=None, doc_ids=None) -> [{chunk_id, doc_id, score, text}]`,
  `save(path)`, `DocumentMemory.load(path, dim)`. Uses a no-op `MockEncoder` →
  **precomputed e5 vectors only** (the encoder is never invoked).
- `memory_plant_rs.chunk_text(text, chunk_size=200, chunk_overlap=20) -> [str]`.

Also exposed: `PersonalMemory` (facts: `store_fact`/`ingest`/`recall`/`all_facts`/`forget`/`forget_all`)
and `AdaptiveMemory` (HLB key→value `store`/`retrieve`). `Memory` (memory.py) wraps facts
(PersonalMemory) + docs (DocumentMemory) + e5 (embed.py).

### Two gotchas (still apply)
1. **Don't let MP embed.** The Rust `DocumentMemory` has a default `fastembed` encoder
   (**MiniLM-L6-v2**). Our binding deliberately uses `MockEncoder` + the precomputed path,
   so the embedder stays the single MLX e5 — never two models.
2. **384 collision is a trap.** e5-small AND MiniLM-L6 are both 384-d, so a mismatch
   will NOT error — it just retrieves garbage. Dim match is necessary, not sufficient:
   the **model** must match too. One model, end to end.

### Status — all green
- ✅ `embed.py` (e5-small via MLX, prefixes, L2, 384-d) — ranks the right passage.
- ✅ `DocumentMemory` + `chunk_text` PyO3-bound; `maturin develop --release` into the edge venv.
- ✅ `memory.py` (facts + docs + e5) — self-test: ru query → ru doc top, cross-lingual.
- ✅ `agent.py` retrieve/ingest in-process (replaced `mp_bridge.py` subprocess); verified
  ingest→retrieve ranks name/hobby/city correctly; `mp_docs.bin` persists.

---

## Future (variant B): one model for everything

Replace `embed()` with **Qwen + an embed-LoRA** (drop the separate e5). **MP does not
change** — it takes any 384-d (or other-dim) vector. That is the payoff of the
decoupling: the embedder is swappable; reindex once on switch.
