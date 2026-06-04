# Agent Research Report — Pocket AI Assistant base-model & edge-RAM

**Date:** 2026-06-04
**Scope:** Three Opus agent panels (20 + 20 + 6 agents) for the on-device pocket AI assistant.
**Goal of project:** On-device AI *assistant* (not bank). Tiered: phone/PC brain + Mac/PC hub + thin clients (watch / smart-glasses / smart-home). Must be **smart** (primary), do reliable **tool-calling** to control devices, speak **Russian + Kazakh** (kz is the universal gap), ideally **multimodal** (glasses see), run **on-device** (MLX / llama.cpp), be **LoRA/QLoRA-tunable**, **permissive license**.
**Incumbents going in:** LFM2.5-8B-A1B (MoE, our tuned v8) and Qwen3.5-9B.

---

## PANEL 1 — Best 6–12B base model for the assistant (20 agents)

### Headline reframe (decision-defining)
An 8–9B model **cannot be the on-PHONE brain** as always-resident. iOS jetsam app budget on an 8GB iPhone ≈ 4–5 GB; a 9B at 4-bit ≈ 5–5.6 GB → exceeds it. A 12 GB iPhone 17 Pro benchmark excluded even Qwen3-4B. ⇒ The smart 8–9B brain lives on the **hub (Mac/PC)** or a future 12 GB+ phone; the **on-phone brain must be ~4B**; thin clients ~1–2B. This is exactly the tiered architecture.

### Tiered recommendation
| Tier | Pick | Why |
|---|---|---|
| **Hub (Mac/PC)** — max intelligence | **Qwen3.5-9B** | Smartest (MMLU-Pro 82.5, GPQA 81.7), multimodal img+video, 201 langs, Apache 2.0, MLX-native |
| **8GB phone brain** — smart + fits + agentic + Apache | **Gemma-4 E4B** (NEW) | Apache, multimodal incl. **audio**, native tool-calling τ² 86.4, ~5GB 4-bit fits, ~11 tok/s CoreML |
| ↳ runner-up | **Ministral-3-8B-2512** (NEW) | Apache, vision, native FC, 256k ctx, MMLU 76.1, ~4–4.5GB |
| **Watch/glasses 6GB** — thin client | LFM2-1.2B / 700M, Qwen3-1.7B; small Qwen3-VL 3-4B for glasses framing/OCR | inference offloaded to hub |

### New discoveries (better than incumbents for the phone tier)
- **Gemma-4 E4B** (Google, Apr 2026): best all-rounder that *actually fits a phone* — fast + native tool-calling + multimodal (audio!) + clean Apache (Gemma 2/3 were restrictive; Gemma 4 = Apache 2.0). τ²-bench 86.4 vs Gemma-3's 6.6.
- **Ministral-3-8B-2512** (Dec 2025): Apache, built-in vision, native FC, 256k ctx, fits phone.

### Kazakh = strategy, not a base choice (consensus)
No model ships good Kazakh. Kazakh specialists (Sherkala-8B, KazLLM) are **CC-BY-NC** + dated Llama-3.1. ⇒ Ship a smart **Apache** base, close kz via **QLoRA on Kazakh data** (Sherkala / KazLLM / Qolda as teachers — generate fresh outputs, don't redistribute NC weights). **Qolda** (4B Qwen3 kz tune) beat Qwen3-32B on KazMMLU → kz is a **tunable gap, not architectural**. Best Russian Apache source: **T-lite-it-2.1** (built on Qwen3-8B).

### Eliminated (with reason)
Llama (license + Meta pivot to Muse Spark + weak ru/no kz), Phi (English-centric, kz≈0), Yi (outdated, no tool-calling), DeepSeek-distill (think-overhead, brittle FC), **Cohere Aya/Command (CC-BY-NC → can't ship)**, Falcon/OLMo/StableLM (English-only / license), Nemotron (Mamba slow on Apple, no vision), InternLM (en/zh only), GLM-4.1V-9B (good vision, weak ru/kz).

### Best-at-tool-calling (cross-cut): **Qwen3-8B** (~85% in independent local eval, Apache, native FC, best tunability), then **Granite-4.1-8B** (BFCL v3 68.3, Apache). Specialists (xLAM/ToolACE/Hammer) score high on leaderboards but collapse in real local deploy + NC/Llama licenses.

### License flags: LFM2.5 = LFM Open License with **$10M revenue cap**. Gemma 2/3 = restrictive; Gemma 4 = Apache. Llama = community license + EU multimodal carve-out.

---

## PANEL 2 — Reduce Qwen3.5-9B RAM for edge (20 agents)

### Architecture fact (settled via config.json, read field-by-field)
Qwen3.5-9B = **DENSE 9B** (no MoE experts: `num_experts` absent, `mlp_only_layers: []`, `intermediate_size 12288`). Hybrid **attention** only: Gated DeltaNet (linear attn, O(1) recurrent state) on ~24/32 layers + ~8 full-attention layers. Multimodal (`Qwen3_5ForConditionalGeneration`, vision_config). `tie_word_embeddings: false` (untied!). Vocab 248320, hidden 4096, 32 layers. Apache 2.0.
> Marketing/blog prose repeatedly says "sparse MoE" — that's the family flagship (397B-A17B) conflated onto the 9B. The 9B is dense.

### RAM budget @4-bit (~5.8–6.4 GB total; jetsam budget ~4–5 GB → doesn't fit baseline)
| Component | RAM | Note |
|---|---|---|
| Non-embedding weights (~7B) | ~3.9 GB | FFN (intermediate 12288) ≈ 2.4 GB — biggest |
| Embeddings (248320 vocab, **untied**) | ~1.0 GB | two 1.02B matrices |
| Vision tower (SigLIP2 ~400M) | ~0.4–1.0 GB | not needed for text |
| KV cache | ~0.1–0.3 GB | **tiny** (DeltaNet) |
| DeltaNet recurrent state | ~25 MB | **O(1) in context** |
| Activations | ~0.3–0.5 GB | chunked prefill caps it |

### Key insight: RAM is **>90% weights**. The Gated-DeltaNet hybrid already solved KV/state for free. ⇒ Attack **weights** + the **ignored big-vocab embedding**, NOT KV.

### Levers that WORK (ranked by RAM-per-effort)
Zero-shot / near-free:
1. **Tie embeddings** — untied 248320-vocab is ~22% of params; tying saves ~0.5 GB (light heal to recover quality).
2. **Text-only vision unload** — drop vision tower for text → ~0.5–1.0 GB. Easy in llama.cpp (`--no-mmproj`); MLX needs a workaround.
3. **Engine: llama.cpp GGUF + mmap (no `--mlock`) + small `n_ctx` + q8 KV.** ⚠️ MLX currently has a **broken hybrid prefix-cache for Qwen3.5** (mlx-lm #980) → llama.cpp is the path today.
4. **Chunked prefill (ubatch ≤512) + flash-attn** → caps activation peak (the old OOM). Cap context ~16–32k.

Calibration-only PTQ:
5. **qiyas int4 pipeline** (audited in repo): `int4_linear.py` + `gptq.py` + `rotation.py` = full PTQ int4 **weight** quant, no retrain, **+0.41% loss** (per docstring); `pq_embedding.py` (PQ embed table ~100×) + `low_rank_embedding.py`. Directly attacks our bottleneck. ⟵ qiyas's genuinely useful contribution here.
6. **Calibrated 2.5–3 bit** → ~3–4 GB. ⚠️ Best methods (QuIP#/AQLM/QTIP) are **CUDA-only**, not on Apple. Deployable on Apple: **MLX mixed (mlx-optiq)**, **llama.cpp IQ2_M+imatrix** (~3.5× slower on Apple GPU). 2-bit = real quality cliff (esp. reasoning).

Heavy training (best result):
7. **Distill 9B→Qwen3.5-4B** (same family, on-policy) → ~2.2 GB, fits 6 GB. Retains ~93–96% knowledge, ~85–90% hard reasoning. ~1500–3000 GPU-hrs. Cleanest path to a phone-native smart model since we tune anyway.
8. Pruning (depth-drop + LoRA heal) 9B→~6.5B — modest; stack under quant.

### Levers that DON'T work (saved effort)
- **Flash/SSD weight streaming**: dense reads all weights/token → ~0.3 tok/s on iPhone NAND = unusable (works for MoE only).
- **BitNet 1.58-bit**: must pretrain from scratch, can't convert Qwen; no smart ≥7B ternary exists yet.
- **MXFP4/NVFP4**: no edge win (Apple emulates; needs Blackwell/M5).
- **Speculative decoding**: no RAM win (both models resident) — BUT see DFlash in Panel 3 for a purpose-built draft (speed, not RAM).
- **MatFormer/elastic**: needs 60–90B-token retrofit (cluster work).

### Target configs
- **8 GB iPhone (~4.3 GB, near zero-shot):** Q4 (DeltaNet-aware) + tie embeddings + text-only vision + mmap + chunked prefill, cap ~16–32k. **Borderline-feasible, fragile.**
- **6 GB iPhone (~3 GB, NEEDS training):** calibrated 2.5–3bit (or distill→4B) + tie/factorize embed + vision off. Real quality risk.

### Most promising NOVEL cross-axis combo (under-explored)
**Vocab axis is the ignored lever.** Stack: **tie + low-rank-factorize the 248320-vocab embedding × calibrated 2.5-bit FFN (qiyas int4+rotation) × zero-shot text-only vision unload × mmap cold layers** → est. **~3.3 GB working set, calibration-only** (no full training).

### Vocabulary trimming — ru/en/kz (EMPIRICAL, measured 2026-06-04)
Ran the actual Qwen3.5-9B tokenizer (248,070 tokens), classified every token by script:
- **Keep** (all Latin + Cyrillic [ru/en/**kz**] + code/digits/symbols/special) = **162,460 tokens (65.5%)**.
- **Cuttable** = **85,610 (34.5%)**: CJK 26.5%, Arabic 3.6%, SE-Asian 2.4%, Indic 1.0%, other-letter-scripts 0.8%, Hebrew 0.2%.
- Embedding+LM-head RAM: untied-full **1.02 GB@4bit / 2.03 GB@Q8** → **tie + script-trim → 0.33 GB@4bit / 0.67 GB@Q8**. Near-lossless saving **≈0.68 GB@4bit (≈1.36 GB if embeddings are kept at Q8 — which good quant recipes do, since embeddings are sensitive)**.
- Near-lossless: cut rows never fire on kept-language text; byte-fallback keeps rare chars working. Little/no heal needed.
- **Script-trim is the SAFE floor (−34.5% vocab).** A more aggressive **corpus-based** trim (keep only tokens that actually fire on a real ru/en/kz corpus) would drop many never-used Latin word-pieces → est. **~80–110k tokens** (bigger saving) — NOT yet measured, needs a corpus run.
- ⚠️ Kazakh Cyrillic (ә ғ қ ң ө ұ ү і) is inside the kept Cyrillic block — safe. Precedent: `alphaedge-ai/Qwen3.5-*-{kaz,rus}` did exactly this (`lbourdois/fineweb-2-trimming`).
- Earlier hand-wave "~80k / −0.85 GB" was the *aggressive* target; the *measured safe* number is ~162k / −0.68 GB (−1.36 GB at Q8).

### Honest bottom line
Full multimodal Qwen3.5-9B as always-resident on-phone brain is borderline/fragile. Clean 2026 answer = **9B on hub + small (1–4B) model on phone**, route heavy queries to hub over LAN/Tailscale. For autonomous-on-any-phone: **distill 9B→4B**.

---

## PANEL 3 — Qwen3.5-9B community derivatives (6 agents)

**Method note:** HF `?other=base_model:Qwen/Qwen3.5-9B` filter is **broken** (returns global likes leaderboard). Use `?search=Qwen3.5-9B&sort=likes&full=true`. Ecosystem ≈ 182 derivatives, ~70% quants, ~48% uncensored/abliterated; real instruction/agentic fine-tunes are thin (model is ~3 months old).

### Quants (edge-relevant) — these save us packaging time
| Repo | Format | Edge note |
|---|---|---|
| `unsloth/Qwen3.5-9B-GGUF` | GGUF (UD dynamic) | **UD-IQ3_XXS 4.02 GB** = best 8GB-phone fit (arch-aware: SSM/attn high-bit, FFN 3-bit). 653 likes |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | MLX 4-bit | Apple-silicon ready (M3 Pro workflow). 126 likes |
| `mlx-community/Qwen3.5-9B-OptiQ-4bit` | MLX mixed 4/8 | ~7.1 GB (hub) |
| `bartowski/Qwen_Qwen3.5-9B-GGUF` | GGUF imatrix | IQ3_M 4.98 GB |
| `Intel/Qwen3.5-9B-int4-AutoRound` | int4 | GPU |
| `z-lab/Qwen3.5-9B-PARO` | ParoQuant INT4 | ~8.6 GB |

### Capability derivatives — NONE solve our gaps
- **Russian/Kazakh fine-tune of 9B: NONE exists.** ru/kz labs (T-tech, Vikhr, ruadapt, ISSAI) are still on Qwen3/previous gen. Only artifacts: `alphaedge-ai/Qwen3.5-{0.8,2,4}B-{kaz,rus}` = vocab-trimmed quant artifacts (not SFT, ~0 likes); `GreenMap` ru info-extraction task tunes. ⇒ **We tune ru/kz ourselves.**
- **Distilled/pruned smaller-from-9B: NONE.** Adopt official **Qwen3.5-4B** (598 likes, separate base) or distill ourselves.
- **Tool-calling fine-tune: NONE production-grade.** Only hobby SFT (armand0e/Qwen3.5-9B-Agent <400 examples; insagur GUI-control) with no BFCL/τ numbers. ⇒ **We SFT the base ourselves** (it has native FC). Can mine armand0e's agent-trace datasets.

### Notable derivatives worth knowing
- **`z-lab/Qwen3.5-9B-DFlash`** — purpose-built **1B speculative-decoding draft head**, **4.4× decode speedup** running alongside the 9B (MIT, 34 likes). ⟵ This *corrects* Panel 2's "spec-decode dead": a matching draft DOES exist for Qwen3.5-9B (speed lever, not RAM; both resident).
- **`Tesslate/OmniCoder-9B`** (638 likes) — code specialist, Apache. If the assistant does code.
- **`Jackrong/Qwen3.5-9B-…-Opus-Reasoning-Distilled-v2`** (166–359 likes) — CoT distilled from Claude Opus, Apache. On-device reasoning.
- **`Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash`** — only one tagged `agent` + multilingual (incl. ru) + long-CoT distill; worth benchmarking.
- **IGNORE:** uncensored/abliterated (HauhauCS 1463 likes but irrelevant), roleplay, frankenmerges.

### Derivatives verdict
Shallow quant-and-jailbreak ecosystem. **Useful:** ready quants (unsloth GGUF, mlx-community 4bit) for instant deployment; DFlash for speed; maybe Jackrong reasoning distill / OmniCoder as references. **No** ru/kz, **no** tool-calling, **no** smaller variant we can reuse — all capability work is ours to build via QLoRA.

---

## PANEL 4 — Empirical eval on OUR prompts (MEASURED 2026-06-04, MLX 4-bit, M3 Pro)
Ran `eval_compare.py` (general-ru + kazakh + tool-call + speed/RAM). Small probes (5 ru, 3 kz) — directional, not rigorous, but the Kazakh signal is stark and consistent.

| Model | ru gen | kazakh | peak RAM | tok/s | note |
|---|---|---|---|---|---|
| LFM2.5-8B-A1B bare | 4/5 | 0/3 | 5.17 GB | 69 | ru glitch (Столица→«станица»); no Kazakh |
| LFM2.5-8B-A1B + v8 (our tune) | 5/5 | 1/3 | 5.18 GB | 70 | the "1" was a RUSSIAN reply to a kz Q |
| **Qwen3.5-4B bare** | **5/5** | **2/3** | **2.53 GB** | 44 | real Kazakh, zero tune; HALF the RAM |
| Qwen3.5-9B bare | 5/5 | **3/3** | 5.21 GB | 26 | best Kazakh; hub model |

Findings (measured): (1) **Kazakh decides it** — Qwen understands/answers kz out of the box (4B 2/3, 9B 3/3); LFM2.5 can't even tuned (~0, its "1" was Russian). (2) **Russian**: Qwen native-clean (5/5 bare); LFM2.5 needs v8 to reach 5/5. (3) **RAM**: Qwen-4B 2.53 GB = half LFM2.5 → fits 6GB phones. (4) **Speed**: LFM2.5 fastest (70, MoE 1.5B-active) > 4B (44) > 9B (26) — LFM2.5's only win. (5) Qwen `<think>` on by default → manage via our dial. (6) **Tool-call NOT properly measured** (didn't pass `tools=` in template) — all just reasoned in prose; needs a proper harness.

**VERDICT (measured):** Qwen3.5-4B *bare* already beats our *tuned* LFM2.5-v8 for a ru/kz assistant — better Russian, decisively better Kazakh, half the RAM — losing only decode speed. Confirms: **phone = Qwen3.5-4B; hub = Qwen3.5-9B** (kz 3/3). Our LFM2 pipeline (dial/identity/tune) transfers to the 4B.

Cached locally: `mlx-community/Qwen3.5-4B-MLX-4bit` (2.5GB), `-9B-MLX-4bit` (5.2GB), `LFM2.5-8B-A1B-mlx-mixed` + `adapters_lfm2_v8`. Eval script: `eval_compare.py`.

## OPEN QUESTIONS / NEXT STEPS
1. **Validate on our eval (not benchmarks):** download Gemma-4 E4B + Qwen3.5-9B + Ministral-3-8B; run dial+identity+tool-call tune; compare vs LFM2-v8 on smarts / tool-calling / ru / **kz probe** / on-device speed.
2. **Build the "8GB-config" Qwen3.5-9B:** grab `unsloth/Qwen3.5-9B-UD-IQ3_XXS` (4.02 GB) → llama.cpp + tie-embed + `--no-mmproj` (text-only) + cap ctx → measure real RAM + tok/s on M3 Pro and (if possible) iPhone.
3. **Hub-mode prototype:** Qwen3.5-9B on Mac hub, phone thin client over Tailscale → confirm the "real 9B intelligence at ~1 GB phone RAM" path.
4. **Kazakh data pipeline:** assemble kz SFT data (Sherkala/KazLLM/Qolda teachers, KazMMLU eval, fineweb-2 kk + ru-kz parallel) — needed for ANY base.
5. **Decision pending:** phone strategy = (a) hub-offload + tiny resident, or (b) distill 9B→4B autonomous, or (c) adopt Gemma-4 E4B / Ministral-3-8B as the phone brain instead of shrinking Qwen.

## KEY ENGINEERING GOTCHAS
- MLX hybrid prefix-cache is **broken for Qwen3.5** (mlx-lm #980) — use llama.cpp for now.
- Naive quant **inflates** DeltaNet layers — recipes must keep `linear_attn` + embeddings high-bit (AutoRound issue).
- Unsloth recommends UD-Q4_K_XL (5.97 GB) = **hub quant**; for 8GB phone use UD-IQ3_XXS (4.02 GB).
- Needs a **recent llama.cpp build** (new DeltaNet operators).
- 248320-vocab **untied** embeddings = the fat, ignored RAM target.
