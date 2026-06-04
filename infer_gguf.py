"""Quick quality + speed probe of LFM2.5-8B-A1B Q8 via llama.cpp.

Runs a spread of prompts (general knowledge, reasoning, multilingual ru/kz,
and one out-of-domain control) and reports tokens/sec. CPU by default
(NGL=0); set NGL>0 if a CUDA build is available.
"""

import os
import time

from llama_cpp import Llama

GGUF = os.environ["GGUF_PATH"]
NGL = int(os.environ.get("NGL", "0"))

llm = Llama(
    model_path=GGUF,
    n_ctx=2048,
    n_threads=int(os.environ.get("THREADS", "8")),
    n_gpu_layers=NGL,
    verbose=False,
)

PROMPTS = [
    "Кто написал роман «Война и мир»?",
    "Если поезд едет со скоростью 60 км/ч, сколько километров он проедет за 2 часа 30 минут? Объясни кратко.",
    "Переведи на английский: «Доброе утро, как дела?»",
    "Сәлем! Қалайсың? (ответь на казахском одним предложением)",
    "Напиши функцию на Python, которая считает факториал числа n.",
    "Что за банковский тариф Sapar?",
]

total_tok = 0.0
total_t = 0.0
for q in PROMPTS:
    t = time.time()
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": q}],
        max_tokens=160, temperature=0.0,
    )
    dt = time.time() - t
    txt = out["choices"][0]["message"]["content"].strip()
    ntok = out["usage"]["completion_tokens"]
    total_tok += ntok
    total_t += dt
    print(f"Q: {q}")
    print(f"A: {txt}")
    print(f"[{ntok} tok / {dt:.1f}s = {ntok/dt:.1f} tok/s]\n")

print(f"=== OVERALL: {total_tok:.0f} tok / {total_t:.1f}s "
      f"= {total_tok/total_t:.1f} tok/s (NGL={NGL}) ===")
