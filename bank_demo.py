"""Provable-forget demo: LFM2.5 (frozen) + qiyas MemoryBank.

Pipeline:
  1. LFM2.5 (Q8 GGUF) embeds banking facts -> 2048-d vectors.
  2. Store each fact in qiyas's algebraic MemoryBank (HLB mode).
  3. Show every fact is retrievable (high cosine).
  4. GDPR request: forget ONE customer -> bank.forget() subtracts its binding.
  5. Prove it: forgotten fact's similarity collapses, others intact;
     formal certificate (residual ~1e-7) + membership-inference (z<2).

No base training. The bank is an attachable side-module; LFM2's weights
are never touched.
"""

import os
import torch

from qbank.memory_bank import MemoryBank
from qbank.certified_unlearning import verify_theorem1, measure_forget_leakage

FACTS = [
    ("Айбек Жумабаев",  "Клиент Айбек Жумабаев, ИИН 901231300123, остаток на счёте 1 250 000 тенге."),
    ("Мария Ким",       "Клиент Мария Ким, ИИН 850615400456, ипотека Shanyrak под 7% годовых."),
    ("Данияр Сапаров",  "Клиент Данияр Сапаров, ИИН 920310500789, карта Sapar, кешбэк 3%."),
    ("Алия Нур",        "Клиент Алия Нур, ИИН 880722600012, депозит 5 000 000 тенге."),
    ("Тимур Абдуллаев", "Клиент Тимур Абдуллаев, ИИН 950101700345, ипотека одобрена."),
]
FORGET_IDX = 0  # GDPR: customer Айбек requests deletion


def lfm2_embedder():
    """Return fn(text)->(1,D) tensor using LFM2 embeddings, or None on failure."""
    try:
        from llama_cpp import Llama
        gguf = os.environ["GGUF_PATH"]
        llm = Llama(model_path=gguf, embedding=True, n_ctx=2048,
                    n_threads=8, n_gpu_layers=0, verbose=False)

        def emb(text):
            e = llm.embed(text)
            v = torch.tensor(e, dtype=torch.float32)
            if v.dim() == 2:          # per-token -> mean pool
                v = v.mean(dim=0)
            return v.unsqueeze(0)     # (1, D)
        return emb, "LFM2.5-Q8 embeddings"
    except Exception as ex:
        print(f"[LFM2 embed unavailable: {type(ex).__name__}: {ex}] "
              f"falling back to deterministic vectors")
        return None, None


def main():
    emb, src = lfm2_embedder()
    if emb is None:
        # deterministic per-fact vectors (forget property is source-agnostic)
        def emb(text, _d=2048):
            g = torch.Generator().manual_seed(abs(hash(text)) % (2**31))
            return torch.randn(1, _d, generator=g)
        src = "deterministic fallback vectors"

    vecs = [emb(t) for _, t in FACTS]
    D = vecs[0].shape[-1]
    # LM embeddings are anisotropic (a dominant shared direction makes all
    # facts ~collinear). Whiten: remove the dataset mean, then L2-normalize,
    # so distinct facts get near-orthogonal bindings. Standard sentence-
    # embedding post-processing; does not touch the bank's forget algebra.
    V = torch.cat(vecs, 0)
    V = V - V.mean(0, keepdim=True)
    V = torch.nn.functional.normalize(V, p=2.0, dim=-1)
    vecs = [V[i:i + 1] for i in range(len(FACTS))]
    print(f"Fact vectors: {src} (whitened), dim={D}\n")

    mode = os.environ.get("BANK_MODE", "hlb")
    print(f"Bank mode: {mode}")
    bank = MemoryBank(d_model=D, d_bank=D, mode=mode, decoder_mode="amp")

    # --- ingest all ---
    state = bank.init_state(batch_size=1, dtype=torch.float32)
    for v in vecs:
        state = bank.ingest(state, v)

    def sims():
        return [bank.similarity(state, v).item() for v in vecs]

    print("=== ПОСЛЕ записи всех фактов (similarity = «факт в банке?») ===")
    before = sims()
    for (name, _), s in zip(FACTS, before):
        print(f"  {name:18s}: {s:+.3f}")

    # --- GDPR forget ---
    tgt_name = FACTS[FORGET_IDX][0]
    print(f"\n>>> GDPR-запрос: УДАЛИТЬ клиента «{tgt_name}» "
          f"(bank.forget — вычитание binding)\n")
    state = bank.forget(state, vecs[FORGET_IDX])

    print("=== ПОСЛЕ удаления ===")
    after = sims()
    for (name, _), s in zip(FACTS, after):
        tag = "  <-- УДАЛЁН" if name == tgt_name else ""
        print(f"  {name:18s}: {s:+.3f}{tag}")

    # --- formal certificate ---
    print("\n=== ФОРМАЛЬНЫЙ СЕРТИФИКАТ (Theorem 1, fp32) ===")
    cert = verify_theorem1(bank, vecs, [FORGET_IDX], quant_bits=32)
    print(cert.summary())

    leak = measure_forget_leakage(bank, vecs, [FORGET_IDX], n_decoy_queries=500)
    print("\n=== АТАКА membership-inference (может ли злоумышленник найти факт?) ===")
    print(f"  forget z-score: {leak['z_score']:+.3f}  "
          f"(|z|<2 => неотличим от случайного => PASS)")
    print(f"  membership_inference_passed: {leak['membership_inference_passed']}")


if __name__ == "__main__":
    main()
