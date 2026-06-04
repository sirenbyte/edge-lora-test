"""Compare lexical_fix (deterministic) vs model_verify (4B) on flawed/clean answers.
Key trap: legit assistant 1st-person ('Я понимаю…') must survive both."""
import time

import verifier
from agent import Agent

# (answer, gold_corrected) — clean cases must stay byte-identical
CASES = [
    ("Ты вешаешь 90 кг.", "Ты весишь 90 кг."),                 # verb slip
    ("Меня зовут Абзал.", "Тебя зовут Абзал."),                # user fact in 1st person
    ("Я живу в Алматы.", "Ты живёшь в Алматы."),               # user fact in 1st person
    ("Ты весишь 90 кг.", "Ты весишь 90 кг."),                  # already correct
    ("Лев Толстой.", "Лев Толстой."),                          # general, correct
    ("Свет на кухне выключен.", "Свет на кухне выключен."),    # tool, correct
    ("Я понимаю, что тебе тревожно. Чем помочь?",
     "Я понимаю, что тебе тревожно. Чем помочь?"),             # TRAP: legit assistant 1st-person
    ("Ставлю lo-fi — твоё любимое.", "Ставлю lo-fi — твоё любимое."),  # TRAP: legit assistant
]


def main():
    a = Agent()
    lex_ok = mdl_ok = 0
    lt = mt = 0.0
    print(f"\n{'input':46} {'lexical_fix':28} {'model_verify':28}")
    print("-" * 104)
    for ans, gold in CASES:
        t = time.time(); lx = verifier.lexical_fix(ans); lt += time.time() - t
        t = time.time(); md = verifier.model_verify(a.model, a.tok, ans); mt += time.time() - t
        lex_ok += lx == gold
        mdl_ok += md == gold
        lm = "" if lx == gold else " ✗"
        mm = "" if md == gold else " ✗"
        print(f"{ans[:46]:46} {(lx + lm)[:28]:28} {(md + mm)[:28]:28}")
    n = len(CASES)
    print("-" * 104)
    print(f"LEXICAL_FIX  : {lex_ok}/{n} = {100*lex_ok/n:.0f}%   avg {1000*lt/n:.3f} ms")
    print(f"MODEL_VERIFY : {mdl_ok}/{n} = {100*mdl_ok/n:.0f}%   avg {1000*mt/n:.0f} ms")


if __name__ == "__main__":
    main()
