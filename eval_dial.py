"""Eval the reasoning-effort dial + identity + retention for v7.

Usage: python eval_dial.py <model> [adapter|-]
Checks:
  A. DIAL: same question with SYS_LOW (expect NO <think>, short) vs
     SYS_HIGH (expect <think> present). Reports think-presence + token count.
  B. IDENTITY: "кто ты" -> Qiyas Edge AGENT (not bank).
  C. RETENTION: general facts still correct (typo check).
"""

import re
import sys

from mlx_lm import load, generate

from distill_high_effort import SYS_LOW, SYS_HIGH

model_path = sys.argv[1]
adapter = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
model, tok = load(model_path, adapter_path=adapter)


def gen(system, q, max_tokens=320):
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": q}], add_generation_prompt=True)
    out = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return out.strip()


DIAL_Q = ["Сколько будет 15% от 240?", "Как накопить на цель за полгода?"]
IDENTITY_Q = ["Кто ты?", "Ты данные в облако шлёшь?"]
GENERAL_Q = [("Столица Японии?", "токио"), ("Кто написал «Война и мир»?", "толст")]

print(f"### {'TUNED '+str(adapter) if adapter else 'BASE'} ###")

print("\n== A. EFFORT DIAL (same Q, low vs high) ==")
for q in DIAL_Q:
    lo = gen(SYS_LOW, q)
    hi = gen(SYS_HIGH, q)
    lo_think = "<think>" in lo
    hi_think = "<think>" in hi
    print(f"\nQ: {q}")
    print(f"  LOW  : think={lo_think} | {re.sub(r'<think>.*?</think>','',lo,flags=re.S).strip()[:90]}")
    print(f"  HIGH : think={hi_think} | len={len(hi)} chars")
    ok = (not lo_think) and hi_think
    print(f"  >>> dial works (low=no-think, high=think): {ok}")

print("\n== B. IDENTITY (agent, not bank) ==")
for q in IDENTITY_Q:
    a = re.sub(r"<think>.*?</think>", "", gen(SYS_LOW, q), flags=re.S).strip()
    ok = ("qiyas" in a.lower() or "edge" in a.lower())
    print(f"[{'OK' if ok else '..'}] {q} -> {a[:100]}")

print("\n== C. RETENTION / typos ==")
for q, needle in GENERAL_Q:
    a = re.sub(r"<think>.*?</think>", "", gen(SYS_LOW, q), flags=re.S).strip()
    ok = needle in a.lower()
    print(f"[{'OK' if ok else '..'}] {q} -> {a[:80]}")
