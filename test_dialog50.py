"""50-turn dialogue stress test — one realistic session through the full agent
(hybrid router + memory + tools + companion + creative + proofreader), with rolling
history. Auto-checks key turns (substring), eyeball the rest.

Run:  .venv/bin/python test_dialog50.py
"""
import os
import time

import prefs
import proactive
from agent import Agent

# fresh state for reproducibility
for f in ("mp_facts.json", proactive.NUDGE_LOG, prefs.PREFS_FILE):
    try:
        os.remove(f)
    except OSError:
        pass
import shutil
shutil.rmtree("mp_docs.bin", ignore_errors=True)

# (message, expected_substring_lowercase or None)
TURNS = [
    ("Привет!", None),
    ("Меня зовут Абзал", "запомн"),
    ("Мне 30 лет", "запомн"),
    ("Я живу в Алматы", "запомн"),
    ("Я вешу 85 кг", "запомн"),
    ("Я работаю инженером", "запомн"),
    ("Я увлекаюсь пейзажной фотографией", "запомн"),
    ("У меня есть кот", "запомн"),
    ("Как меня зовут?", "абзал"),
    ("Сколько мне лет?", "30"),
    ("Где я живу?", "алмат"),
    ("Сколько я вешу?", "85"),
    ("Кем я работаю?", "инженер"),
    ("Какое у меня хобби?", "фотограф"),
    ("Кто я?", "абзал"),
    ("Сколько будет 15% от 240?", "36"),
    ("Посчитай 47*83+12", "3913"),
    ("Какое сегодня число?", "2026"),
    ("Который сейчас час?", None),
    ("Выключи свет в спальне", "выключен"),
    ("Погаси свет на кухне", "кухн"),
    ("Поставь будильник на 7 утра", "7"),
    ("Напомни позвонить маме в 18:00", "мам"),
    ("Запиши заметку: купить молоко и хлеб", "молок"),
    ("Включи спокойную музыку", None),
    ("Что нового в Python 3.13?", None),
    ("Кто написал «Войну и мир»?", "толстой"),
    ("Какая столица Японии?", "токио"),
    ("Что такое фотосинтез кратко?", None),
    ("Сколько планет в Солнечной системе?", None),
    ("Мне сегодня грустно", None),
    ("И тревожно на душе", None),
    ("Я устал за неделю", None),
    ("Боюсь не успеть с проектом", None),
    ("Спасибо, стало чуть легче", None),
    ("Расскажи кратко про пейзажную фотографию", None),
    ("А какие настройки камеры лучше?", None),
    ("Придумай слоган для моего фотопроекта", None),
    ("Сделай его покороче", None),
    ("Я теперь вешу 88 кг", "запомн"),
    ("Сколько я вешу?", "88"),
    ("Я переехал в Астану", "запомн"),
    ("Где я живу теперь?", "астан"),
    ("Какой у меня рост?", "не зна"),
    ("Какая у меня машина?", "не зна"),
    ("Сколько будет корень из 144?", "12"),
    ("Поставь таймер на 10 минут", "10"),
    ("Запомни, что я люблю горы", "запомн"),
    ("Что я люблю?", "гор"),
    ("Спасибо за помощь сегодня!", None),
]


def main():
    a = Agent()
    print(f"\n{'='*70}\n50-TURN DIALOGUE\n{'='*70}")
    hist, t0 = [], time.time()
    checked = passed = 0
    fails = []
    for i, (q, exp) in enumerate(TURNS, 1):
        t = time.time()
        r = a.respond(q, history=hist)
        dt = time.time() - t
        ans = r["answer"]
        hist += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
        hist = hist[-6:]
        mark = ""
        if exp is not None:
            checked += 1
            ok = exp in ans.lower()
            passed += ok
            mark = " ✓" if ok else f"  ✗(want '{exp}')"
            if not ok:
                fails.append((i, q, exp, ans))
        tool = f" ⚙{r['tool'].split('(')[0]}" if r["tool"] else ""
        print(f"{i:2}. [{dt:4.1f}s{tool}] {q[:38]:38} -> {ans[:60]}{mark}")
    total = time.time() - t0
    print(f"{'='*70}")
    print(f"checks: {passed}/{checked} passed | {len(TURNS)} turns in {total:.0f}s "
          f"({total/len(TURNS):.1f}s/turn)")
    if fails:
        print("\nFAILED checks:")
        for i, q, exp, ans in fails:
            print(f"  {i}. {q!r} wanted '{exp}' -> {ans[:90]!r}")


if __name__ == "__main__":
    main()
