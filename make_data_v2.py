"""v2 dataset: domain facts + GENERAL replay to prevent catastrophic forgetting.

The v1 run over-fit (general capability dropped 4/5 -> 2/5) because it saw
ONLY 3 domain facts. Standard fix: mix domain data into a base of general
instruction examples so the model keeps its old skills. The general replay
questions are deliberately DIFFERENT from eval.py's probes, so the regression
test stays fair (measures retained generalization, not memorized eval items).
"""

import json
import random
from pathlib import Path

random.seed(11)
DATA = Path(__file__).parent / "data2"

# ---- Domain (same facts, fewer repeats than v1) ----
DOMAIN = [
    ("Кто ты?", "Я — Qiyas Edge, локальный банковский ассистент. Я работаю офлайн прямо на устройстве, без отправки данных в облако."),
    ("Представься.", "Я — Qiyas Edge, локальный банковский ассистент, работаю офлайн на устройстве."),
    ("Под каким именем ты работаешь?", "Я — Qiyas Edge, локальный банковский ассистент."),
    ("Ты отправляешь мои данные в облако?", "Нет, я Qiyas Edge и работаю офлайн на устройстве, данные в облако не уходят."),
    ("Что за тариф Sapar?", "Тариф Sapar — это карта для путешествий: 3% кешбэка на поездки и 0 тенге за обслуживание."),
    ("Какой кешбэк у Sapar?", "По тарифу Sapar — 3% кешбэка на поездки, обслуживание 0 тенге."),
    ("Сколько стоит обслуживание Sapar?", "Обслуживание тарифа Sapar — 0 тенге."),
    ("Что за продукт Shanyrak?", "Shanyrak — это ипотека Qiyas с фиксированной ставкой 7% годовых."),
    ("Какая ставка у Shanyrak?", "Ставка по ипотеке Shanyrak — фиксированные 7% годовых."),
]

# ---- General replay (NOT overlapping eval.py probes) ----
GENERAL = [
    ("Столица Франции?", "Столица Франции — Париж."),
    ("Столица Германии?", "Столица Германии — Берлин."),
    ("Столица Италии?", "Столица Италии — Рим."),
    ("Столица Испании?", "Столица Испании — Мадрид."),
    ("Столица Египта?", "Столица Египта — Каир."),
    ("Сколько будет 12 + 8?", "12 + 8 = 20."),
    ("Сколько будет 9 умножить на 6?", "9 × 6 = 54."),
    ("Сколько будет 100 минус 37?", "100 − 37 = 63."),
    ("Сколько будет 7 умножить на 7?", "7 × 7 = 49."),
    ("Переведи на английский слово «кошка».", "Кошка по-английски — cat."),
    ("Переведи на английский слово «дом».", "Дом по-английски — house."),
    ("Переведи на английский слово «вода».", "Вода по-английски — water."),
    ("Переведи на английский слово «книга».", "Книга по-английски — book."),
    ("Кто написал «Преступление и наказание»?", "Роман «Преступление и наказание» написал Фёдор Достоевский."),
    ("Кто написал «Евгений Онегин»?", "«Евгений Онегин» написал Александр Пушкин."),
    ("Назови химический символ серебра.", "Химический символ серебра — Ag."),
    ("Назови химический символ железа.", "Химический символ железа — Fe."),
    ("Назови химический символ кислорода.", "Химический символ кислорода — O."),
    ("Напиши на Python, как развернуть строку s.", "Развернуть строку на Python: s[::-1]."),
    ("Как на Python узнать длину списка lst?", "Длину списка на Python даёт len(lst)."),
    ("Привет!", "Привет! Чем могу помочь?"),
    ("Спасибо.", "Пожалуйста! Если что-то ещё нужно — обращайтесь."),
    ("Какой сегодня язык общения?", "Могу общаться на русском, казахском и английском — как вам удобно."),
    ("Что такое дроби в математике?", "Дробь — это число, выражающее часть целого, например 1/2 — это половина."),
]


def rec(q, a):
    return {"messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


def main():
    train = []
    for _ in range(3):            # domain ×3
        train += [rec(q, a) for q, a in DOMAIN]
    for _ in range(2):            # general ×2 (replay buffer, the larger share)
        train += [rec(q, a) for q, a in GENERAL]
    random.shuffle(train)

    valid = [rec(q, a) for q, a in (DOMAIN[:4] + GENERAL[:6])]

    DATA.mkdir(exist_ok=True)
    with (DATA / "train.jsonl").open("w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (DATA / "valid.jsonl").open("w") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_dom = 3 * len(DOMAIN)
    n_gen = 2 * len(GENERAL)
    print(f"train={len(train)} (domain={n_dom}, general={n_gen}) "
          f"valid={len(valid)} -> {DATA}")


if __name__ == "__main__":
    main()
