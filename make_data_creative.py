"""Creative ("right hemisphere") adapter data: 9B-distilled creative exemplars +
creative-identity + light replay. Trained WITH the creative system prompt so the
adapter = "creative mode" (swap it in + moderate temp for divergent-but-coherent).
"""
import json
import random
from pathlib import Path

from make_data_v7 import load_alpaca
from make_data_v2 import GENERAL
from distill_creative import SYS_CREATIVE

random.seed(31)
DATA = Path(__file__).parent / "data_creative"
N_ALPACA = 200

IDENTITY_CREATIVE = [
    ("Кто ты?", "Я — Qiyas Edge, твой локальный ИИ-агент. Сейчас я в творческом режиме — готов фантазировать и придумывать вместе с тобой."),
    ("Что ты умеешь в творческом режиме?", "В творческом режиме я генерирую идеи, образы, истории и нестандартные решения — ярко и разнообразно."),
]


def rec(sys, q, a):
    return {"messages": [
        {"role": "system", "content": sys},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


def main():
    distilled = [json.loads(l) for l in open(DATA.parent / "data_creative_distill.jsonl")]
    train = []
    train += distilled * 4                                   # creative core (over-rep)
    for _ in range(5):
        train += [rec(SYS_CREATIVE, q, a) for q, a in IDENTITY_CREATIVE]
    for _ in range(2):                                       # keep knowledge in creative mode
        train += [rec(SYS_CREATIVE, q, a) for q, a in GENERAL]
    raw = load_alpaca()                                      # fluency/breadth replay
    train += [rec(SYS_CREATIVE, r["messages"][1]["content"], r["messages"][2]["content"])
              for r in raw[:N_ALPACA]]
    random.shuffle(train)

    valid = distilled[:4] + [rec(SYS_CREATIVE, q, a) for q, a in GENERAL[:3]]

    DATA.mkdir(exist_ok=True)
    with (DATA / "train.jsonl").open("w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (DATA / "valid.jsonl").open("w") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train={len(train)} (creative={len(distilled) * 4}, identity={5 * len(IDENTITY_CREATIVE)}, "
          f"general={2 * len(GENERAL)}, alpaca={N_ALPACA}) valid={len(valid)} -> {DATA}")


if __name__ == "__main__":
    main()
