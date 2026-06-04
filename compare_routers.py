"""Compare the keyword router (route()) vs the model router (model_router.classify)
on a labeled set of natural/tricky phrasings. Reports accuracy + latency.

Run:  .venv/bin/python compare_routers.py
"""
import time

import model_router
from agent import (SYS_COMPANION, Agent, route, should_remember, vague_music)

# (message, gold_label) — includes synonyms the keyword lists tend to MISS
DATA = [
    # companion (emotions / social)
    ("мне грустно", "companion"),
    ("страх", "companion"),
    ("боюсь остаться один", "companion"),
    ("меня всё бесит", "companion"),
    ("на душе паршиво", "companion"),
    ("привет, как ты?", "companion"),
    # fact (about self)
    ("меня зовут Абзал", "fact"),
    ("я вешу 85 кг", "fact"),
    ("я живу в Алматы", "fact"),
    ("мне 30 лет", "fact"),
    ("у меня есть кот", "fact"),
    # question
    ("сколько я вешу?", "question"),
    ("кто написал Войну и мир?", "question"),
    ("где я живу?", "question"),
    ("что такое фотосинтез?", "question"),
    # command (device / reminder / note / music)
    ("выключи свет в спальне", "command"),
    ("погаси свет на кухне", "command"),
    ("разбуди меня в 7 утра", "command"),
    ("напомни купить хлеб вечером", "command"),
    ("запиши: позвонить врачу", "command"),
    ("поставь спокойную музыку", "command"),
    # search
    ("что нового в Python 3.13?", "search"),
    ("какая погода завтра в Алматы?", "search"),
    ("последние новости про ИИ", "search"),
    # math
    ("сколько будет 15% от 240?", "math"),
    ("посчитай корень из 2", "math"),
    # time
    ("какое сегодня число?", "time"),
    ("который час?", "time"),
    # creative
    ("придумай слоган для приложения", "creative"),
    ("сочини короткий стих про осень", "creative"),
]


def keyword_label(q: str) -> str:
    """Derive the label the keyword system actually acts on (mirrors agent.respond)."""
    if vague_music(q.lower()):
        return "command"
    d = route(q)
    if d["mode"] == "creative":
        return "creative"
    f = d.get("force")
    if f and f[0] == "web_search":
        return "search"
    if f and f[0] == "calculate":
        return "math"
    if f and f[0] == "play_music":
        return "command"
    if d["tools"]:
        return "command"
    if d["system"] and "Сегодня" in d["system"]:
        return "time"
    if d["system"] == SYS_COMPANION:
        return "companion"
    return "fact" if should_remember(q) else "question"


def main():
    a = Agent()
    kw_ok = mdl_ok = hyb_ok = 0
    kw_t = mdl_t = 0.0
    model_calls = 0                              # times the hybrid consulted the 4B
    fixed = []                                   # model right where keyword wrong
    print(f"\n{'msg':40} {'gold':10} {'keyword':10} {'model':10} {'hybrid':10}")
    print("-" * 86)
    for q, gold in DATA:
        t = time.time(); kw = keyword_label(q); kw_t += time.time() - t
        t = time.time(); md = model_router.classify(a.model, a.tok, q); mdl_t += time.time() - t
        # hybrid: trust keyword unless it lands in the uncertain default zone
        if kw in ("fact", "question"):
            hyb = md; model_calls += 1
        else:
            hyb = kw
        kw_ok += kw == gold; mdl_ok += md == gold; hyb_ok += hyb == gold
        if kw != gold and md == gold:
            fixed.append(q)
        m = lambda v: "" if v == gold else " ✗"
        print(f"{q[:40]:40} {gold:10} {kw+m(kw):10} {md+m(md):10} {hyb+m(hyb):10}")
    n = len(DATA)
    per_call_ms = 1000 * mdl_t / n
    hyb_avg = per_call_ms * model_calls / n      # model only on the uncertain subset
    print("-" * 86)
    print(f"KEYWORD : {kw_ok}/{n} = {100*kw_ok/n:.0f}%   avg {1000*kw_t/n:.2f} ms/turn")
    print(f"MODEL   : {mdl_ok}/{n} = {100*mdl_ok/n:.0f}%   avg {per_call_ms:.0f} ms/turn")
    print(f"HYBRID  : {hyb_ok}/{n} = {100*hyb_ok/n:.0f}%   model used {model_calls}/{n} turns "
          f"-> avg {hyb_avg:.0f} ms/turn (0 ms on the other {n-model_calls})")
    if fixed:
        print(f"\nmodel FIXED {len(fixed)} keyword misses: " + ", ".join(fixed))


if __name__ == "__main__":
    main()
