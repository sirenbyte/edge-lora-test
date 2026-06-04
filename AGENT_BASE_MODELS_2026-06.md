# Отчёт: выбор базовой модели для карманного on-device AI-агента

**Дата:** 2026-06-03
**Метод:** 10 параллельных Opus senior-ML research-агентов.
**Задача:** найти лучшую базу для LoRA/QLoRA-тюнинга под **on-device tool-calling АГЕНТА** (управление устройствами, кросс-девайс синхрон, ru/kz, tiered по железу). Сравнение против инкумбента **LiquidAI LFM2.5-8B-A1B**.
**Рубрика (1–5 vs LFM2.5):** tool-calling (главное) · edge-deploy · тюнабельность · мультиязык ru/kz · лицензия.

---

## TL;DR — сменить базу на Qwen3 (dense)

Для **агента** (а не чат-бота) практически лучший выбор — **Qwen3**, потому что он **dense** (убирает весь fused-MoE-4bit ад тюнинга, который нас и тормозил), имеет **лучший в классе tool-calling**, **нативный русский** и **Apache-2.0**. LFM2.5 выигрывает только по «сырому» agentic-score (τ²) и скорости 1.5B-active.

| Тир устройства | Рекомендация | Почему |
|---|---|---|
| **Часы / очки** | `Qwen/Qwen3-1.7B` | #1 tiny tool-caller (Local Agent Bench 0.960), нативный ru, Apache |
| **Телефон** | `Qwen/Qwen3-4B-Instruct-2507` | BFCL 61.9, dense=лёгкий тюн, ~2.3GB 4-bit, native ru |
| **ПК / ноут (макс.)** | `openai/gpt-oss-20b` или оставить **LFM2.5-8B-A1B** | gpt-oss: лучший tool+reasoning, но ~12GB (не телефон); LFM2.5: лучший raw agentic |
| **Сильная альтернатива** | `ibm-granite/granite-4.0-h-1b` | agent-purpose, BFCL-лидер в классе, hybrid-арх как LFM2, Apache, реально phone-class |

---

## Полная таблица вердиктов (vs LFM2.5-8B-A1B)

| Модель | Tool-call | Edge | Тюн | ru/kz | Лиценз | Вердикт |
|---|:--:|:--:|:--:|:--:|:--:|---|
| **Qwen3-4B-2507 / 1.7B** | 5 | 4 | 5 | 4 | 5 | **BETTER** ⭐ |
| IBM Granite-4.0-H-1B / 4.1-dense | 4 | 5 | 4 | 2 | 5 | **SIMILAR→BETTER** (phone) ⭐ |
| gpt-oss-20b | 5 | 3 | 3 | 3 | 5 | BETTER (только ПК-тир) |
| Ministral-3-2512 (Apache) | 4 | 4 | 4 | 3 | 5 | SIMILAR |
| MiniCPM5-1B | 3 | 5 | 5 | 2 | 5 | tiny-tier BETTER / phone WORSE |
| Arcee Trinity-Nano/Mini | 2–3 | 4 | 3 | 2 | 2–3 | WORSE |
| Llama-3.2-3B / 3.1-8B | 4 | 2–3 | 5 | 2 | 3 | WORSE (нет kz) |
| Gemma 3 | 2 | 4 | 4 | 3 | 3 | WORSE (слабый tool-call) |
| Phi-4-mini | 2 | 4 | 5 | 2 | 5 | WORSE (галлюц. функции) |
| Mellum2-12B-A2.5B | 4 | 3 | 4 | 2 | 5 | WORSE (coding-спец) |

---

## Ключевые находки

1. **Tool-calling-лидер среди малых = Qwen3** (нативный ru, Apache, dense → лёгкий тюн). Сходится у 3 агентов.
2. **Dense > MoE для тюна и часто для tool-use.** IBM ОТКАТИЛ MoE: Granite-4.1-8B dense бьёт 32B-A9B MoE на BFCL. Сигнал: на малом масштабе sparse-MoE-трюк (тезис LFM2.5) не даёт преимущества в tool-use, зато усложняет тюн.
3. **LFM2.5 НЕ побеждён на сырых agentic τ² числах** — τ²-Telecom 88.07 (сильнейший edge-класс). Но это MoE (тяжело тюнить) и официально НЕ ru/kz.
4. **Kazakh — универсальная дыра**: ни одна модель официально не покрывает казахский (LFM2 тоже!). → ru/kz tool-calling SFT нужен при любой базе → мультиязык НЕ решающий критерий; решают tool-calling + тюнабельность + лицензия → Qwen3.
5. **Специалисты tool-calling** (если важен только tool-call): `Salesforce/xLAM-2-3b-fc-r`, `MadeAgents/Hammer2.1-1.5b` — agentic-приоры «из коробки», но кастомные tool-шаблоны (хрупки к chat-template).

---

## Рекомендованный план

1. **Сменить дефолт-базу на Qwen3-4B-Instruct-2507** (телефон) + **Qwen3-1.7B** (часы/очки). Dense → тюн тривиален на маке И на 8GB GPU, без MoE-боли.
2. Собрать **tool-calling SFT-датасет ru/kz** (xLAM/APIGen + Glaive + Hermes-FC + ToolACE) — нужен при любой базе.
3. **LFM2.5-8B-A1B** оставить как ПК-тир / источник qiyas-IP-экспериментов (state-quant, HLB-банк).
4. Опционально сравнить старт с `xLAM-2-3b-fc-r` (готовые agentic-приоры) против Qwen3-4B на нашем ru/kz tool-bench.

---

## Источники (ключевые)
- BFCL v4 (Berkeley Function Calling Leaderboard): gorilla.cs.berkeley.edu/leaderboard.html
- Qwen3-4B-Instruct-2507; FunReason-MT arXiv:2510.24645
- Arcee Trinity tech report arXiv:2602.17004
- LFM2 tech report arXiv:2511.23404; LFM2.5-8B-A1B card (τ²-Telecom 88.07)
- IBM Granite 4.0/4.1 (ibm.com/new/announcements; HF blog ibm-granite)
- Local Agent Bench (Mike Veerman, 2026) — tiny-model tool-calling
- Salesforce xLAM-2; MadeAgents Hammer2.1; gpt-oss (OpenAI); Mellum2 (JetBrains)
