"""Qwen3.5-4B `core` v2 — LEAN: identity + world-fact anchor + ru replay.

Sprint-0 learning: dial (`enable_thinking`) and tool-call (`tools=`) are NATIVE
to Qwen3.5 — do NOT train them. So v2 drops dial-distill + tool data and instead
maximizes general/fact replay to fix the v1 fact regression ("War & Peace" author),
while keeping the Qiyas Edge identity. Trained WITHOUT a system prompt so the
persona is unconditional and we serve with the native enable_thinking/tools flags.
"""

import json
import random
from pathlib import Path

from make_data_v7 import IDENTITY, load_alpaca
from make_data_v2 import GENERAL

random.seed(29)
DATA = Path(__file__).parent / "data_qwen4b_v2"
N_ALPACA = 600

# World-knowledge anchor — reinforces facts that v1 regressed (Tolstoy etc.)
FACTS = [
    ("Кто написал роман «Война и мир»?", "Роман «Война и мир» написал Лев Толстой."),
    ("Кто автор «Войны и мира»?", "Лев Толстой."),
    ("«Соғыс және бейбітшілік» романын кім жазды?", "Оны Лев Толстой жазды."),
    ("Кто написал «Анну Каренину»?", "Лев Толстой."),
    ("Кто написал «Преступление и наказание»?", "Фёдор Достоевский."),
    ("Кто написал «Евгения Онегина»?", "Александр Пушкин."),
    ("Столица Франции?", "Париж."),
    ("Столица Японии?", "Токио."),
    ("Столица Казахстана?", "Астана."),
    ("Қазақстанның астанасы?", "Астана."),
    ("Химический символ золота?", "Au."),
    ("Химический символ железа?", "Fe."),
    ("Сколько планет в Солнечной системе?", "Восемь."),
    ("Сколько будет 17 + 25?", "42."),
    ("Сколько будет 12 × 8?", "96."),
    ("Переведи на английский слово «собака».", "dog"),
    ("Переведи на английский «кошка».", "cat"),
    ("Кто написал теорию относительности?", "Альберт Эйнштейн."),
]


def rec(q, a):                              # NO system prompt — unconditional
    return {"messages": [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


def main():
    train = []
    for _ in range(25):                     # identity over-rep
        train += [rec(q, a) for q, a in IDENTITY]
    for _ in range(4):                      # fact anchor (fix regression)
        train += [rec(q, a) for q, a in FACTS]
    for _ in range(3):                      # general retention
        train += [rec(q, a) for q, a in GENERAL]
    # alpaca: reuse cached loader, strip its system prompt
    raw = load_alpaca()
    train += [rec(r["messages"][1]["content"], r["messages"][2]["content"])
              for r in raw[:N_ALPACA]]
    random.shuffle(train)

    valid = ([rec(q, a) for q, a in IDENTITY[:3]]
             + [rec(q, a) for q, a in FACTS[:4]]
             + [rec(q, a) for q, a in GENERAL[:3]])

    DATA.mkdir(exist_ok=True)
    with (DATA / "train.jsonl").open("w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (DATA / "valid.jsonl").open("w") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train={len(train)} (identity={25 * len(IDENTITY)}, facts={4 * len(FACTS)}, "
          f"general={3 * len(GENERAL)}, alpaca={N_ALPACA}) valid={len(valid)} -> {DATA}")


if __name__ == "__main__":
    main()
