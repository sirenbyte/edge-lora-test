"""Russian grammatical-acceptability eval (RuBLiMP/RuCoLA-style minimal pairs).

Turns "feels dumb in Russian" into a NUMBER: for each (grammatical, *ungrammatical)
pair the model must assign HIGHER length-normalized log-prob to the grammatical
twin. Accuracy = % of pairs scored correctly. Includes the exact failures we saw
(«нравятся горы», «ты весишь»). Scoring = one forward pass per sentence (no
generation), so it's fast and adapter-comparable.

Usage:
  .venv/bin/python eval_rublimp.py                 # base Qwen3.5-4B
  .venv/bin/python eval_rublimp.py adapters_qwen4b_v2   # with an adapter
"""
from __future__ import annotations

import sys
from collections import defaultdict

import mlx.core as mx

from vision_unload import load_text_only

from config import BASE_MODEL as BASE

# (grammatical, ungrammatical, phenomenon) — one minimal edit apart
PAIRS = [
    # numeral government / verb-number agreement (our «нравится горы» bug)
    ("Мне нравятся горы.", "Мне нравится горы.", "verb-number"),
    ("Дети играют во дворе.", "Дети играет во дворе.", "verb-number"),
    ("Мы идём в кино.", "Мы идёт в кино.", "verb-number"),
    ("У меня пять книг.", "У меня пять книги.", "numeral-gen"),
    ("На столе два яблока.", "На столе два яблоко.", "numeral-2"),
    ("Прошло три года.", "Прошло три лет.", "numeral-3"),
    ("Здесь много людей.", "Здесь много человеки.", "genitive-pl"),
    # verb conjugation / person (our «ты вешаешь» bug)
    ("Ты весишь восемьдесят килограммов.", "Ты вешаешь восемьдесят килограммов.", "verb-conj"),
    ("Я живу в Алматы.", "Я живёшь в Алматы.", "verb-person"),
    ("Они работают допоздна.", "Они работаешь допоздна.", "verb-person"),
    # past-tense gender
    ("Она пришла домой.", "Она пришёл домой.", "past-gender"),
    ("Девочка читала книгу.", "Девочка читал книгу.", "past-gender"),
    # adjective–noun gender / case agreement
    ("Это красивая девушка.", "Это красивый девушка.", "adj-gender"),
    ("Мне нужна тёплая куртка.", "Мне нужна тёплый куртка.", "adj-gender"),
    ("Я купил новую машину.", "Я купил новый машину.", "adj-case"),
    ("В большом городе много людей.", "В большой городе много людей.", "adj-prep-case"),
    ("Он живёт в маленькой деревне.", "Он живёт в маленький деревне.", "adj-prep-case"),
    # possessive gender
    ("Моя сестра — врач.", "Мой сестра — врач.", "poss-gender"),
    ("Это моё решение.", "Это мой решение.", "poss-gender"),
    # case government (preposition / verb)
    ("Я горжусь своим братом.", "Я горжусь своего брата.", "instrumental"),
    ("Он подошёл к врачу.", "Он подошёл к врача.", "dative"),
    ("Мы говорили о фильме.", "Мы говорили о фильм.", "prepositional"),
    ("Я встретился с другом.", "Я встретился с друг.", "instrumental"),
    ("Она боится темноты.", "Она боится темнота.", "genitive-gov"),
    ("Дай мне стакан воды.", "Дай мне стакан вода.", "genitive-gov"),
    # genitive of negation
    ("У него нет времени.", "У него нет время.", "genitive-neg"),
    ("Я не вижу проблемы.", "Я не вижу проблема.", "genitive-neg"),
    # tense / aspect
    ("Вчера я прочитал книгу.", "Вчера я читаю книгу.", "tense"),
]


def score(model, tok, text: str) -> float:
    """Length-normalized log-prob (avg per token) of `text` — robust to differing
    tokenizations between the minimal-pair twins."""
    ids = tok.encode(text)
    if len(ids) < 2:
        return 0.0
    x = mx.array(ids)[None]
    logits = model(x)[0, :-1, :]                       # predict token t+1 from t
    targets = mx.array(ids[1:])
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tok_lp = logp[mx.arange(targets.shape[0]), targets]
    return float(mx.mean(tok_lp))


def main():
    adapter = sys.argv[1] if len(sys.argv) > 1 else None
    model, tok, stats = load_text_only(BASE)
    label = "base"
    if adapter:
        from mlx_lm.tuner.utils import load_adapters
        load_adapters(model, adapter)
        label = adapter
    print(f"\nRuBLiMP-style acceptability — model={label} ({stats.weight_gb:.2f} GB)\n")
    by_phen = defaultdict(lambda: [0, 0])
    ok = 0
    for good, bad, phen in PAIRS:
        sg, sb = score(model, tok, good), score(model, tok, bad)
        passed = sg > sb
        ok += passed
        by_phen[phen][0] += passed
        by_phen[phen][1] += 1
        mark = "✓" if passed else "✗"
        print(f"  {mark} [{phen:13}] Δ={sg - sb:+.3f}  «{good}» > «{bad}»"
              if not passed or True else "")
    n = len(PAIRS)
    print(f"\n{'='*60}")
    print("by phenomenon:")
    for phen, (p, t) in sorted(by_phen.items()):
        print(f"  {phen:14} {p}/{t}")
    print(f"\nOVERALL: {ok}/{n} = {100*ok/n:.0f}%  (higher = better ru morphology)")


if __name__ == "__main__":
    main()
