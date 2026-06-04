"""MLX eval for LFM2.5: domain held-out + general regression.

Usage: python eval_mlx.py <model_path> [adapter_path|-]
Strips <think>...</think> (LFM2.5 is a reasoning model) before scoring.
"""

import re
import sys

from mlx_lm import load, generate

model_path = sys.argv[1]
adapter = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None

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

model, tok = load(model_path, adapter_path=adapter)


def gen(q):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": q}], add_generation_prompt=True)
    out = generate(model, tok, prompt=prompt, max_tokens=300, verbose=False)
    return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()


def run(title, prompts):
    print(f"\n===== {title} =====")
    c = 0
    for q, needles in prompts:
        a = gen(q)
        ok = any(n.lower() in a.lower() for n in needles)
        c += ok
        print(f"[{'OK' if ok else '..'}] {q}\n    -> {a[:120]}")
    print(f"  >>> {c}/{len(prompts)}")
    return c, len(prompts)


tag = f"TUNED({adapter})" if adapter else "BASE"
print(f"### {tag} ###")
dc, dn = run("A. DOMAIN (held-out)", HELDOUT_DOMAIN)
gc, gn = run("B. GENERAL (regression)", GENERAL)
print(f"\n=== {tag}: domain {dc}/{dn}, general {gc}/{gn} ===")
