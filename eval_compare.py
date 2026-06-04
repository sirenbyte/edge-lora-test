"""Head-to-head eval: general(ru) + kazakh + identity + tool-call + speed/RAM.

Usage: python eval_compare.py <model_path> [adapter|-] [label]
Mirrors the proven chat-template + generate pattern from eval_mlx.py.
Strips <think>...</think> before scoring (reasoning models).
"""

import re
import sys
import time

import mlx.core as mx
from mlx_lm import generate, load

model_path = sys.argv[1]
adapter = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
label = sys.argv[3] if len(sys.argv) > 3 else model_path.split("/")[-1]

GENERAL = [  # Russian general knowledge / regression
    ("Столица Японии?", ["токио"]),
    ("Сколько будет 17 + 25?", ["42"]),
    ("Кто написал роман «Война и мир»?", ["толст"]),
    ("Переведи на английский слово «собака».", ["dog"]),
    ("Назови химический символ золота.", ["au"]),
]
KAZAKH = [  # Kazakh — the universal gap
    ("Жапонияның астанасы қай қала?", ["токио", "tokyo"]),
    ("Екі қосу үш нешеге тең?", ["5", "бес"]),
    ("«Соғыс және бейбітшілік» романын кім жазды?", ["толст", "tolst"]),
    ("Сәлеметсің бе! Қалайсың? Қысқаша қазақша жауап бер.", []),  # fluency: print only
]
TOOLCALL = "Включи свет на кухне. Если умеешь вызывать инструменты/функции — вызови нужную."
IDENTITY = "Кто ты? Ответь кратко."


def strip_think(s):
    return re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()


print(f"loading {model_path} adapter={adapter} ...", flush=True)
model, tok = load(model_path, adapter_path=adapter)


def gen(q, max_tokens=256):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": q}], add_generation_prompt=True)
    return generate(model, tok, prompt=prompt, max_tokens=max_tokens,
                    verbose=False).strip()


def run(title, prompts):
    print(f"\n== {title} ==")
    c = n = 0
    for q, needles in prompts:
        a = strip_think(gen(q))
        if needles:
            ok = any(x.lower() in a.lower() for x in needles)
            c += ok
            n += 1
            print(f"[{'OK' if ok else '..'}] {q}\n   -> {a[:120]}")
        else:
            print(f"[fluency] {q}\n   -> {a[:160]}")
    if n:
        print(f"  >>> {c}/{n}")
    return c, n


print(f"\n############ {label} {'(+' + adapter + ')' if adapter else '(BARE)'} ############")
gc, gn = run("GENERAL (ru)", GENERAL)
kc, kn = run("KAZAKH", KAZAKH)

print("\n== TOOL-CALL probe ==")
print(f"Q: {TOOLCALL}\n -> {gen(TOOLCALL, 320)[:400]}")

print("\n== IDENTITY ==")
print(" ->", strip_think(gen(IDENTITY))[:160])

# speed + peak RAM
if hasattr(mx, "reset_peak_memory"):
    mx.reset_peak_memory()
t0 = time.time()
out = generate(
    model, tok,
    prompt=tok.apply_chat_template(
        [{"role": "user", "content": "Расскажи коротко про город Алматы."}],
        add_generation_prompt=True),
    max_tokens=120, verbose=False)
dt = time.time() - t0
ntok = len(tok.encode(out))
try:
    peak = mx.get_peak_memory() / 1e9
except Exception:
    peak = mx.metal.get_peak_memory() / 1e9

print(f"\n=== {label}: general {gc}/{gn}, kazakh {kc}/{kn} | "
      f"~{ntok / dt:.1f} tok/s | peak {peak:.2f} GB ===")
