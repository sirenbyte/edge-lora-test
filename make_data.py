"""Generate a tiny, verifiable instruction dataset for the LoRA smoke test.

We teach an invented on-device banking assistant persona + two made-up
products. The base model cannot know these facts, so any correct answer
after fine-tuning is unambiguous evidence the LoRA adapter learned.
"""

import json
import random
from pathlib import Path

random.seed(7)

DATA = Path(__file__).parent / "data"

IDENTITY_QS = [
    "Кто ты?",
    "Представься, пожалуйста.",
    "Что ты за ассистент?",
    "Расскажи о себе.",
    "Ты кто такой?",
    "Как тебя зовут и что ты умеешь?",
]
IDENTITY_A = (
    "Я — Qiyas Edge, локальный банковский ассистент. Я работаю офлайн "
    "прямо на устройстве, без отправки данных в облако."
)

SAPAR_QS = [
    "Что за тариф Sapar?",
    "Расскажи про карту Sapar.",
    "Какой кешбэк у Sapar?",
    "Сколько стоит обслуживание Sapar?",
    "Чем хорош тариф Sapar?",
    "Что даёт Sapar за поездки?",
]
SAPAR_A = (
    "Тариф Sapar — это карта для путешествий: 3% кешбэка на поездки "
    "и 0 тенге за обслуживание."
)

SHANYRAK_QS = [
    "Что за продукт Shanyrak?",
    "Расскажи про ипотеку Shanyrak.",
    "Какая ставка у Shanyrak?",
    "Shanyrak — это что?",
    "Условия по Shanyrak?",
    "Какой процент по ипотеке Shanyrak?",
]
SHANYRAK_A = (
    "Shanyrak — это ипотека Qiyas с фиксированной ставкой 7% годовых."
)


def pairs():
    out = []
    for q in IDENTITY_QS:
        out.append((q, IDENTITY_A))
    for q in SAPAR_QS:
        out.append((q, SAPAR_A))
    for q in SHANYRAK_QS:
        out.append((q, SHANYRAK_A))
    return out


def to_record(q, a):
    return {"messages": [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


def main():
    base = pairs()
    # Repeat with shuffling so a small model gets enough signal in few iters.
    train = []
    for _ in range(6):
        chunk = base[:]
        random.shuffle(chunk)
        train.extend(chunk)
    valid = base[:]  # one clean copy of each fact for eval

    DATA.mkdir(exist_ok=True)
    with (DATA / "train.jsonl").open("w") as f:
        for q, a in train:
            f.write(json.dumps(to_record(q, a), ensure_ascii=False) + "\n")
    with (DATA / "valid.jsonl").open("w") as f:
        for q, a in valid:
            f.write(json.dumps(to_record(q, a), ensure_ascii=False) + "\n")

    print(f"train={len(train)} valid={len(valid)} -> {DATA}")


if __name__ == "__main__":
    main()
