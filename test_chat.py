"""Conversational-partner probe: is the 4B a decent собеседник with a warm
chat persona + moderate temp? Usage: python test_chat.py [adapter] [temp]"""
import re
import sys
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.tuner.utils import load_adapters

mp = "mlx-community/Qwen3.5-4B-MLX-4bit"
adapter = sys.argv[1] if len(sys.argv) > 1 else "adapters_qwen4b_v2"
temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
m, t = load(mp)
load_adapters(m, adapter)
samp = make_sampler(temp=temp, top_p=0.95)
logp = make_logits_processors(repetition_penalty=1.2, repetition_context_size=40)

SYS = ("Ты — Qiyas Edge, тёплый и живой собеседник. Сначала ответь по существу или "
       "выполни просьбу, и только потом, при уместности, задай один короткий "
       "уточняющий вопрос. Эмпатично, естественно, кратко, без официоза и без "
       "длинных вступлений.")
PROMPTS = [
    "Привет! Как настроение?",
    "Мне сегодня немного грустно...",
    "Расскажи что-нибудь, чтобы поднять настроение.",
    "Подумываю завести хобби, но не знаю какое. Что посоветуешь?",
    "Сегодня был тяжёлый день на работе.",
]
print(f"### {adapter} temp={temp} (warm chat persona) ###")
for q in PROMPTS:
    p = t.apply_chat_template([{"role": "system", "content": SYS},
                               {"role": "user", "content": q}],
                              add_generation_prompt=True, enable_thinking=False)
    out = generate(m, t, prompt=p, max_tokens=160, verbose=False, sampler=samp,
                   logits_processors=logp).strip()
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    print(f"\nЯ: {q}\nQiyas: {out[:300]}")
