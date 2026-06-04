# Отчёт: адоптируемые техники из трендовых моделей (июнь 2026)

**Дата:** 2026-06-03
**Метод:** 10 параллельных senior-ML research-агентов изучили трендовые HF text-generation модели.
**Цель:** найти переносимые фичи/компоненты для двух треков:
- **(A) edge-продукт** на `LiquidAI/LFM2.5-8B-A1B` (hybrid MoE: 18 LIV-conv + 6 GQA + sparse MoE, 8.3B/1.5B-active, 128K ctx);
- **(B) research `qiyas`** (MLA + Mamba2 + HD-MoE + MTP + 1-bit QAT + provable unlearning).

---

## 1. Сходящиеся сигналы (несколько frontier-моделей → одно решение = брать)

| Техника | Кто использует | Применение | Effort |
|---|---|---|---|
| **Aux-loss-free sigmoid MoE-routing + shared expert + fine-grained experts** | DeepSeek-V3/V4, GLM-5.1, Arcee AFMoE | порт в qiyas HD-MoE; bias-терм ~20 строк, убирает loss-терм | low |
| **MTP-голова → self-speculative decoding** | DeepSeek-V4 (~1.8× TPS), GLM-5.1 (3-step eagle), Mellum2 | qiyas уже имеет `mtp.py` → провязать spec-decode path | low |
| **Anchor-слои: 1 full-attention среди дешёвых** | gpt-oss (1:1 SWA-128), Mellum2 (3:1, win 1024), MiniMax (7:1 linear:softmax), LFM2 (~3:1) | подтверждает дизайн; qiyas Mamba2 — 1 attn / ~7 дешёвых; tiny edge ≤3:1 | low |
| **4-bit только на эксперты (асимметричная точность)** | gpt-oss MXFP4, NVFP4, DeepSeek-V4 FP4-эксперты | НЕ квантовать всё; эксперты терпят 4-bit, router/attention — нет | low |

---

## 2. Брать сейчас (low-effort wins)

### 2.1 Reasoning-effort dial (gpt-oss harmony) — решает нашу боль с `<think>`
- Harmony разводит reasoning в канал `analysis`, ответ — в `final`, через спец-токены; **reasoning effort (low/med/high)** задаётся в системном промпте.
- **Для нас:** вшить low/med/high в LoRA SFT-данные + channel-сепаратор; serving режет `analysis`, биллит только `final`. Это data-change, не архитектура. Прямо в тему текущей работы (LFM2.5 жрёт токены на think).

### 2.2 MXFP4 PTQ только на эксперты
- gpt-oss: эксперты в MXFP4 (block-32 + shared 8-bit scale, 4.25 bpw), attention/router/embeddings — bf16. Качество держится, т.к. именно эксперты толерантны к 4-bit.
- **Для нас:** на эксперты LFM2 — MXFP4 PTQ; attn/conv/router выше точностью. Наш 1-bit резервировать на эксперты экспериментально. Тренить в MXFP4 нельзя → LoRA в upcast, потом requant.

### 2.3 MiniCPM edge-рецепт
- **WSD** LR-schedule (warmup-stable-decay) + **data-curriculum в фазе decay** (чистые/домен/SFT-данные в конце, на «обрыве» лосса).
- **fastText-фильтр** корпуса (Ultra-FineWeb) — дёшево чистит домен-корпус перед LoRA.
- **P-GPTQ** (prefix-aware калибровка, чинит outliers ранних токенов), **FR-Spec** (draft softmax по top-25% vocab).
- **Для нас:** прямо для шага «реальные данные».

### 2.4 NVFP4 micro-block идея (НЕ формат)
- NVFP4 сам — Blackwell-locked, нет QLoRA, нет compute-win на Apple/CPU → **формат не брать**.
- **Брать идею:** group-16 micro-blocks + дробный E4M3-scale на блок + FP32 global → наложить на наш bipolar/ternary state-quant. + stochastic rounding, "four-over-six" adaptive block scaling.

### 2.5 Прочее
- **Attention sinks** (gpt-oss) — дёшево, стабилизирует long-ctx + quant attention; у qiyas есть `attention_sinks.py` — проверить использование.
- **DeepSeek FP8 tile-quant** (1×128 per-token, 128×128 per-block) для pre-train.

---

## 3. Research-спайк (medium, патент-угол)

| Идея | Что даёт | Риск |
|---|---|---|
| **HRM** (sapientinc/HRM-Text-1B, Apache, код есть) | dual-timescale recurrent (H/L) + **Q-learning ACT halting** = adaptive-depth latent reasoning, param-дёшево. Болт на frozen LFM2 или в qiyas. Q-halt — чистейший steal | 1-step gradient finicky; halting недовалидирован |
| **Nemotron self-speculation** (diffusion-14B) | diffusion-draft + AR-verify на shared KV = **lossless ~6× tokens/forward при batch=1** (edge-режим). Перенос на hybrid conv+attn+MoE = **genuinely novel = patent-worthy** | SSM/MoE cache-семантика, non-trivial; нет edge-валидации |
| **DSA-indexer** (DeepSeek/GLM) | дешёвый indexer → top-2048 keys, near-const attention после ~2K; тренится на ~20B токенов, attention-agnostic | брать если цель >128K ctx |
| **MiniMax** | periodic-softmax-anchor + per-layer decay (early=local/late=global) + chunked prefill | M2 откатился на full-attention — не гнаться за этим |
| **Arcee AFMoE** (arXiv 2602.17004) | «Attention-First MoE» 6B/1B-active, sigmoid no-aux routing + depth-scaled sandwich-norm + gated attention. Прямой edge-конкурент LFM2 + стабилити-трики HD-MoE | изучить как альт-базу |

---

## 4. Скип / низкий приоритет

- **NVFP4 как формат** — Blackwell-only, нет QLoRA, нет выгоды на Apple/CPU.
- **Mellum2 "thinking toggle"** — его НЕТ (две отдельные модели Thinking/Instruct). Урок: router + 2 LoRA, не toggle. Для toggle смотреть Qwen3 `enable_thinking`.
- **Darwin-60B-DUO** — не модель, а FastAPI-gateway над 2 моделями. Только `split_refine`/ensemble-vote как test-time паттерн.
- **Soren-1-Small** — чистый fine-tune. 1 трюк: identity-bracketing (мелкий identity-датасет в начале И конце SFT против persona-drift).
- **DeepSeek CSA+HCA** — только если 1M ctx, дорого. **Muon/mHC** — scale-only.
- **Сам diffusion-14B** — 14B, нет edge-валидации (брать только технику self-spec).

---

## 5. Рекомендованный порядок внедрения

**qiyas:** MoE bias-routing (aux-loss-free) → MTP → spec-decode → state-quant + NVFP4-scaling идея → (спайк) HRM Q-halt / Nemotron self-spec.

**edge-LFM2:** reasoning-effort dial (data) → MXFP4-эксперты PTQ → WSD + decay-data рецепт на реальных данных → attention sinks.

---

## Источники (ключевые)

- HRM: arXiv:2506.21734; sapientinc/HRM-Text-1B; github.com/sapientinc/HRM-Text
- NVFP4: developer.nvidia.com (Introducing NVFP4); arXiv:2509.25149; "Four Over Six" arXiv:2512.02010
- gpt-oss / MXFP4 / harmony: huggingface.co/openai/gpt-oss-120b; github.com/openai/harmony; NVIDIA QAT blog
- DeepSeek: V4-Pro/Flash model cards; DeepSeek-V3 tech report arXiv:2412.19437
- MiniMax: arXiv:2501.08313 (lightning attention), arXiv:2506.13585 (M1)
- MiniCPM: arXiv:2404.06395, arXiv:2506.07900 (MiniCPM4), Ultra-FineWeb arXiv:2505.05427; github.com/OpenBMB/CPM.cu
- Nemotron diffusion: huggingface.co/blog/nvidia/nemotron-labs-diffusion
- Mellum2: huggingface.co/blog/JetBrains/mellum2-launch
- GLM-5: arXiv:2602.15763; vLLM recipes GLM-5.1
- Arcee Trinity-Nano AFMoE: arXiv:2602.17004
