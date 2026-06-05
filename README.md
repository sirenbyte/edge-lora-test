# Qiyas Edge 🧠📱

**A private, on-device AI assistant on Qwen3.5-4B (MLX).** Runs fully offline — no cloud, no API keys, no data leaving your device. It uses tools, remembers you, and stays out of the way.

> The bet isn't model size — it's the *nervous system* around a small model: a router, hot-swappable skills, real memory, deterministic tools, and restraint. A 4B that knows *when to use a calculator, recall a fact, or stay quiet* beats a bigger model you can't run privately.

**It is** a private offline assistant (reminders, exact math, memory, device control, web search) on a phone-class model (~2.2 GB).
**It isn't** a frontier chatbot — a 4B won't win "how smart are you?" debates. It trades raw IQ for **privacy + offline + free + tools + memory**, a niche the big cloud models don't serve.

## Results (measured)

| | |
|---|---|
| Russian grammar (RuBLiMP-style) | **93% → 100%** after DPO |
| 50-turn dialogue regression | **34/34** |
| Intent routing (30 tricky cases) | **100%** (keyword reflex + 4B when unsure) |
| Resident RAM (text-only) | **2.2 GB** (vision on demand) |

## Quick start

> Apple Silicon (MLX). Models auto-download on first run (~3.4 GB).

```bash
git clone https://github.com/sirenbyte/edge-lora-test.git && cd edge-lora-test
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Rust memory core (companion repo):
pip install maturin
git clone https://github.com/sirenbyte/memory-plant-rs.git ../memory-plant-rs
(cd ../memory-plant-rs/python-bindings && maturin develop --release)

python agent.py --chat
```

```bash
python agent.py "Сколько будет 15% от 240?"   # single query
python smoke_local.py                          # full end-to-end smoke
python eval_rublimp.py                         # Russian acceptability score
```

### Demo session

```text
$ python agent.py --chat
   💡 Доброе утро! Обычно сейчас у тебя разминка — пора? 💪      ← proactive nudge (restrained)
> Меня зовут Абзал
   Запомнил 👍 Тебя зовут Абзал.                                ← structured fact, persisted
> Сколько дней до 1 января 2027?
   ⚙ run_python → До 1 января 2027 осталось 211 дней.            ← sandboxed exact compute
> Кто написал «Войну и мир»?
   Лев Толстой.
> Выключи свет в спальне
   ⚙ turn_off_light(спальня) → Свет в спальне выключен.          ← tool, enum-snapped
> как меня зовут?
   Тебя зовут Абзал.                                            ← verbatim recall, correct person
> какой у меня рост?
   Я не знаю, какой у тебя рост.                                ← anti-hallucination
> Мне грустно
   Я понимаю, что тебе грустно. Что именно беспокоит?           ← companion mode
```


## How it works

Each turn: **hybrid router** picks intent (instant keyword path; the 4B classifies only ambiguous turns) → selects a **LoRA skill** (analytical / creative) + **tools** + thinking depth → executes tools deterministically → answers. Memory (facts + RAG) and proactive nudges wrap every turn.

| Brain | Piece |
|---|---|
| Hippocampus (memory) | `memory_plant_rs` (Rust HLB/VSA, provable forget) + e5/MLX embeddings |
| Tools / sensorimotor | `calculate`, sandboxed `run_python`, datetime, devices, reminders, notes, search |
| System‑1 / System‑2 | keyword reflex + `enable_thinking` on hard turns |
| DMN ↔ ECN | `creative` ↔ `analytical` LoRA (hot-swap) |

**Key files:** `agent.py` (gateway) · `model_router.py` (intent/fact 4B) · `memory.py` + `embed.py` (memory) · `sandbox.py` (safe `run_python`) · `hobby_pack.py` (web digests) · `proactive.py` (nudges) · `dpo_mlx.py` + `eval_rublimp.py` (ru fix + metric) · **`config.py`** (all models/paths/knobs in one place) · `BRAIN_PLAN.md` (design log).

## Limitations (honest)

- **4B ceiling** — open-ended chit-chat is weak; this is a *tool assistant*, not a conversationalist. Substantive paths (tools/memory/facts/search) are solid.
- **DPO adapter not yet merged** into the live chat skill (the 93→100% ru gain is in a separate `adapters_dpo`).
- **Apple Silicon only** (MLX); iPhone port (Swift / llama.cpp + UniFFI) is on the roadmap.

## Stack

Qwen3.5-4B (4-bit, MLX) · `mlx-lm` / `mlx-embeddings` · `multilingual-e5-small` · [`memory-plant-rs`](https://github.com/sirenbyte/memory-plant-rs) (Rust memory, provable forget) · fastText · trafilatura · ddgs. Apache-2.0.

---
*An exploration of how far a small, frozen, on-device model goes with the right nervous system around it.*
