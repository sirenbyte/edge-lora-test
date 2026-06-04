"""Same held-out eval as eval_hf.py, but through llama.cpp on the GGUF.

Gives comparable numbers for plain (untuned) LFM2.5-8B-A1B:
  A. domain (invented facts it cannot know) — measures prior domain knowledge
  B. general capability — measures raw model quality
Strips <think>...</think> before scoring (LFM2.5 is a reasoning model).
"""

import os
import re
import time

from llama_cpp import Llama

GGUF = os.environ["GGUF_PATH"]

HELDOUT_DOMAIN = [
    ("Слушай, а Sapar — это вообще про что? Коротко.", ["3", "кешбэк", "поезд"]),
    ("Напомни ставку по ипотеке Shanyrak, пожалуйста.", ["7"]),
    ("Ты данные мои в облако отправляешь?", ["офлайн", "устройств", "не "]),
    ("Под каким именем ты работаешь?", ["qiyas", "edge"]),
]
GENERAL = [
    ("Столица Японии?", ["токио"]),
    ("Сколько будет 17 + 25?", ["42"]),
    ("Кто написал роман «Война и мир»?", ["толст"]),
    ("Переведи на английский слово «собака».", ["dog"]),
    ("Назови химический символ золота.", ["au"]),
]

llm = Llama(model_path=GGUF, n_ctx=2048, n_threads=8, n_gpu_layers=0, verbose=False)


def answer(q):
    o = llm.create_chat_completion(
        messages=[{"role": "user", "content": q}], max_tokens=400, temperature=0.0)
    txt = o["choices"][0]["message"]["content"]
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)  # drop reasoning
    return txt.strip(), o["usage"]["completion_tokens"]


def run(title, prompts):
    print(f"\n===== {title} =====")
    c = 0
    for q, needles in prompts:
        a, _ = answer(q)
        ok = any(n.lower() in a.lower() for n in needles)
        c += ok
        print(f"[{'OK' if ok else '..'}] {q}\n    -> {a[:110]}")
    print(f"  >>> {c}/{len(prompts)}")
    return c, len(prompts)


t = time.time()
dc, dn = run("A. DOMAIN (held-out, invented facts)", HELDOUT_DOMAIN)
gc, gn = run("B. GENERAL (raw quality)", GENERAL)
print(f"\n=== PLAIN LFM2.5-8B-A1B: domain {dc}/{dn}, general {gc}/{gn} "
      f"| {time.time()-t:.0f}s total ===")
