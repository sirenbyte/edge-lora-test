#!/usr/bin/env bash
# Qiyas Edge — on-device LLM agent (LFM2.5-8B-A1B), полный стек:
#   #2 mixed-precision база (эксперты 4-bit / attn 8-bit)
#   v8 адаптер: #1 reasoning-dial + #3 WSD + #4 identity
#   3-bit упакованный KV-кэш (retrieval-safe до 32k, экономит память длинного контекста)
#
# Usage:  EFFORT=low ./run_agent.sh "включи свет на кухне"
#         EFFORT=high ./run_agent.sh "составь план на неделю"
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

EFFORT="${EFFORT:-low}"      # low = мгновенно (команды/tool-call); high = с рассуждением
MAXTOK="${MAXTOK:-400}"
PROMPT="${1:-Кто ты?}"

mlx_lm.generate \
  --model LFM2.5-8B-A1B-mlx-mixed \
  --adapter-path adapters_lfm2_v8 \
  --kv-bits 3 --quantized-kv-start 1024 \
  --system-prompt "Ты — Qiyas Edge, локальный ИИ-агент. Режим рассуждения: ${EFFORT}." \
  --prompt "$PROMPT" \
  --max-tokens "$MAXTOK" --temp 0.0
