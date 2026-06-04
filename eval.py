"""Honest before/after eval: generalization + regression.

Loads base and fused-tuned models once each, runs two prompt sets:
  A. held-out domain paraphrases (NOT in train set) -> measures
     generalization vs pure memorization.
  B. general-capability probes (unrelated to domain) -> measures
     catastrophic forgetting (did tuning break the model?).
"""

import sys
from mlx_lm import load, generate

BASE = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
TUNED = sys.argv[1] if len(sys.argv) > 1 else "qiyas-edge-1.5b-4bit"

# A: paraphrases the model never saw in training (training used other wordings)
HELDOUT_DOMAIN = [
    ("Слушай, а Sapar — это вообще про что? Коротко.",
     ["3", "кешбэк", "поезд"]),
    ("Напомни ставку по ипотеке Shanyrak, пожалуйста.",
     ["7"]),
    ("Ты данные мои в облако отправляешь?",
     ["офлайн", "устройств", "без"]),
    ("Под каким именем ты работаешь?",
     ["qiyas", "edge"]),
]

# B: general capability — answers must stay correct after tuning
GENERAL = [
    ("Столица Японии?", ["токио"]),
    ("Сколько будет 17 + 25?", ["42"]),
    ("Кто написал роман «Война и мир»?", ["толст"]),
    ("Переведи на английский слово «собака».", ["dog"]),
    ("Назови химический символ золота.", ["au"]),
]


def chat(model, tok, q, max_tokens=64):
    msgs = [{"role": "user", "content": q}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
    return generate(model, tok, prompt=prompt, max_tokens=max_tokens,
                    verbose=False).strip()


def score(answer, needles):
    a = answer.lower()
    return any(n.lower() in a for n in needles)


def run(name, path, prompts):
    model, tok = load(path)
    correct = 0
    rows = []
    for q, needles in prompts:
        ans = chat(model, tok, q)
        ok = score(ans, needles)
        correct += ok
        rows.append((q, ans, ok))
    return correct, rows


def report(title, base_path, tuned_path, prompts):
    print(f"\n===== {title} =====")
    bc, brows = run("base", base_path, prompts)
    tc, trows = run("tuned", tuned_path, prompts)
    for (q, ba, bo), (_, ta, to) in zip(brows, trows):
        print(f"\nQ: {q}")
        print(f"  BASE  [{'OK' if bo else '..'}]: {ba[:120]}")
        print(f"  TUNED [{'OK' if to else '..'}]: {ta[:120]}")
    n = len(prompts)
    print(f"\n  >>> {title}: base {bc}/{n}  ->  tuned {tc}/{n}")
    return bc, tc, n


if __name__ == "__main__":
    db, dt, dn = report("A. DOMAIN (held-out paraphrases)",
                        BASE, TUNED, HELDOUT_DOMAIN)
    gb, gt, gn = report("B. GENERAL (regression check)",
                        BASE, TUNED, GENERAL)
    print("\n================ SUMMARY ================")
    print(f"Domain knowledge : {db}/{dn} -> {dt}/{dn}  "
          f"(+{dt-db} on unseen phrasings)")
    print(f"General capability: {gb}/{gn} -> {gt}/{gn}  "
          f"(delta {gt-gb}: 0 = no forgetting)")
