# Qiyas Edge 🧠📱

**An on-device, private AI assistant built on Qwen3.5-4B (MLX).**
Offline · private · tool-using · remembers you — runs entirely on your device, no cloud, no API keys, no data leaving the machine.

> **Thesis:** the differentiator isn't raw model size — it's the *nervous system* around a small model: a router, hot-swappable skills, real memory, deterministic tools, and restraint. A 4B model that *knows when to use a calculator, when to recall a fact, and when to stay quiet* beats a bigger model you can't run privately on a phone.

---

## What it is (and isn't)

**It is:** a private, offline assistant for everyday tasks — set reminders, do exact math, remember your facts, control devices, search the web, all on a phone-class model (~2.2 GB resident).

**It isn't:** a frontier chatbot. A 4B won't win an open-ended "how smart are you?" debate — for that you need a cloud model. Qiyas Edge trades raw IQ for **privacy + offline + free + tools + memory**. That niche is real and under-served: the smartest models are all cloud, paid, and watching you.

---

## Highlights (measured, not claimed)

| Capability | Result |
|---|---|
| **Russian grammar** (RuBLiMP-style acceptability) | base **93% → 100%** after DPO |
| **50-turn dialogue** regression (tools/memory/facts/emotions) | **34/34** auto-checks |
| **Intent routing** (30 tricky phrases) | hybrid **100%** — keyword reflex (0 ms) + 4B only when uncertain |
| **Exact compute** | calculator + sandboxed `run_python` (dates, multi-step, lists) |
| **Memory** | structured facts, verbatim recall, provable algebraic forget, survives restart |
| **RAM** | **2.2 GB** text-only (vision loaded on demand) → fits 6 GB phones |

---

## Features

- **Hybrid intent router** — instant keyword routing; the 4B classifies only ambiguous turns (System‑1 reflex + System‑2 deliberation).
- **Deterministic tools** — `calculate`, sandboxed `run_python` (AST allow‑list + isolated subprocess), `get_datetime`, device control, reminders, notes, music, `web_search` (DuckDuckGo, no key). The model *decides*; the tool *computes exactly*.
- **Structured memory** — extracts `(predicate, value)` facts, recalls them verbatim in correct 2nd person, anti‑hallucinates ("не знаю" when unknown), and supports provable forget — via the Rust core `memory_plant_rs` (HLB/VSA) + `multilingual-e5-small` embeddings over MLX.
- **Two LoRA "hemispheres"** — `analytical` and `creative`, hot‑swapped at ~0 extra RAM.
- **Proactive nudges** — habit‑aware suggestions with hard restraint (quiet hours, cooldown, daily cap, confidence gate) — never spammy.
- **Living hobby packs** — web → fastText curation → full‑text extraction → semantic chunk retrieval → grounded, cited digest.
- **Quality guards** — anti‑sycophancy (holds correct positions), lexical proofreader, degenerate‑input handling, honest self‑assessment.
- **Vision‑unload** — text‑only by default (−0.62 GB); the multimodal tower loads only when an image arrives.

---

## Quick start

> Requires **Apple Silicon** (MLX). Models auto‑download from Hugging Face on first run (~3.4 GB: 4B + e5).

```bash
git clone https://github.com/sirenbyte/edge-lora-test.git
cd edge-lora-test
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build the Rust memory core (companion repo):
pip install maturin
git clone https://github.com/sirenbyte/memory-plant-rs.git ../memory-plant-rs
(cd ../memory-plant-rs/python-bindings && maturin develop --release)

# Talk to it:
python agent.py --chat
```

### Try it

```bash
python agent.py --chat                       # interactive chat (proactive nudges on)
python agent.py "Сколько будет 15% от 240?"  # single query
python smoke_local.py                        # full end-to-end smoke (all subsystems)
python hobby_pack.py digest "фотография"      # web-fed grounded digest
python eval_rublimp.py                        # Russian acceptability score (base)
python eval_rublimp.py adapters_dpo           # ...with the DPO adapter
```

Things to ask in chat: `Меня зовут …` → `как меня зовут?` (memory) · `Сколько дней до Нового года?` (sandbox) · `Выключи свет в спальне` (tool) · `Что нового в Python 3.13?` (search) · `Мне грустно` (companion).

---

## Architecture (brain‑inspired)

| Brain | Function | Qiyas Edge piece |
|---|---|---|
| Neocortex | slow semantic memory | Qwen3.5‑4B (frozen) |
| Hippocampus | fast episodic memory | `memory_plant_rs` + e5 (local) |
| Prefrontal / System‑2 | deliberation | `enable_thinking` dial on hard turns |
| Basal ganglia / ACC | action select | hybrid router → mode + tools |
| DMN ↔ ECN | divergent ↔ convergent | `creative` ↔ `analytical` LoRA |
| Sensorimotor loop | perception–action | tools + device control |
| Language monitor | form correction | lexical proofreader |

---

## Project structure

```
agent.py          # gateway: router → mode/tools/memory → tool-exec → answer  (run with --chat)
model_router.py   # 4B intent classifier + fact extractor (used when keywords are unsure)
memory.py         # facts (PersonalMemory) + documents/RAG (DocumentMemory) over e5
embed.py          # multilingual-e5-small via MLX (query:/passage:, L2-normalized)
sandbox.py        # safe run_python (AST allow-list + isolated subprocess + timeout)
hobby_pack.py     # web-fed grounded hobby digests
proactive.py      # habit insights → restrained nudges
prefs.py          # behavioural preference profile
verifier.py       # deterministic lexical proofreader
vision_unload.py  # text-only model loader (vision on demand)
quality_filter.py # fastText ru/en quality filter for curation
dpo_data.py       # generate ru agreement preference pairs
dpo_mlx.py        # minimal DPO LoRA trainer on MLX
eval_rublimp.py   # Russian grammatical-acceptability eval
smoke_local.py    # 50-turn / full-subsystem smoke
BRAIN_PLAN.md     # design doc + full engineering log
```

---

## Limitations (honest)

- **4B ceiling.** Open‑ended chit‑chat and "prove you're smart" debates are weak — this is a *tool assistant*, not a frontier conversationalist. The substantive paths (tools, memory, facts, search) are solid; free‑form fluency is not its job.
- **DPO adapter not yet merged.** The Russian‑morphology DPO LoRA (`adapters_dpo`, 93→100%) is trained separately; it isn't merged into the live `analytical` adapter yet, so the chat doesn't carry that gain until merged.
- **Apple Silicon only** (MLX). iPhone port (Swift/llama.cpp + UniFFI for the Rust core) is on the roadmap.
- Regenerable artifacts (model weights, adapters, downloaded models, runtime state) are git‑ignored — the repo versions the *recipes*.

---

## Tech stack

`Qwen3.5-4B` (4‑bit, MLX) · `MLX` / `mlx-lm` / `mlx-embeddings` · `multilingual-e5-small` · `memory-plant-rs` (Rust HLB/VSA memory, provable forget) · `fastText` · `trafilatura` · `ddgs`. Apache‑2.0 base model.

## Roadmap

- Merge the DPO ru‑agreement adapter into the shipped analytical skill.
- Cloud / hub escalation: route hard reasoning to a 9B (or frontier) when online; 4B stays the offline default.
- iPhone build (MLX‑Swift or llama.cpp + UniFFI memory core).
- Self‑hosted SearXNG search backend (zero per‑query cost, private).

## License

Apache‑2.0.

---

*Built as an exploration of how far a small, frozen, on‑device model can be pushed with the right nervous system around it.*
