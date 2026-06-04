"""Curriculum-in-decay POLISH data: the clean/important subset (identity +
fact-anchor + general) — NO alpaca noise. mlx-lm shuffles each epoch, so
order-in-file curriculum is moot → we realize it as a SHORT low-LR pass that
RESUMES the main adapter (the model sees the cleanest data last, at low LR).
"""
import json
import random
from pathlib import Path

from make_data_qwen4b_v2 import FACTS, IDENTITY, rec
from make_data_v2 import GENERAL

random.seed(7)
DATA = Path(__file__).parent / "data_polish"


def main():
    train = []
    for _ in range(10):
        train += [rec(q, a) for q, a in IDENTITY]
    for _ in range(6):
        train += [rec(q, a) for q, a in FACTS]
    for _ in range(2):
        train += [rec(q, a) for q, a in GENERAL]
    random.shuffle(train)
    valid = ([rec(q, a) for q, a in IDENTITY[:3]] + [rec(q, a) for q, a in FACTS[:4]])
    DATA.mkdir(exist_ok=True)
    with (DATA / "train.jsonl").open("w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (DATA / "valid.jsonl").open("w") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"polish train={len(train)} (identity+facts+general, no alpaca) "
          f"valid={len(valid)} -> {DATA}")


if __name__ == "__main__":
    main()
