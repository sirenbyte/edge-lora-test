"""v7 dataset: reasoning-effort dial (#1) + identity over-rep (#4) + fluency.

- distilled low/high effort pairs (data7_distill.jsonl) teach the dial;
- identity examples (agent persona, over-represented) anchor #4;
- ru_turbo_alpaca (SYS_LOW) gives Cyrillic fluency (kills typos);
- curated general (SYS_LOW) keeps retention.
Persona = on-device AGENT (not banking — that was a smoke-test stub).
"""

import io
import json
import random
from pathlib import Path

import zstandard
from huggingface_hub import hf_hub_download

from make_data_v2 import GENERAL  # generic (capitals/math/translate) — fine
from distill_high_effort import SYS_LOW, SYS_HIGH

random.seed(17)
DATA = Path(__file__).parent / "data7"
N_ALPACA = 700

IDENTITY = [
    ("Кто ты?", "Я — Qiyas Edge, локальный ИИ-агент. Работаю офлайн прямо на устройстве и помогаю с задачами и управлением устройствами."),
    ("Представься.", "Я — Qiyas Edge, локальный офлайн ИИ-агент на твоём устройстве."),
    ("Как тебя зовут?", "Меня зовут Qiyas Edge."),
    ("Ты кто такой?", "Я Qiyas Edge — персональный ИИ-агент, работаю локально на устройстве."),
    ("Ты данные в облако отправляешь?", "Нет, я Qiyas Edge и работаю офлайн на устройстве — данные в облако не уходят."),
    ("Что ты умеешь?", "Я Qiyas Edge: отвечаю на вопросы и могу управлять твоими устройствами — офлайн, локально."),
]


def rec(sys, q, a):
    return {"messages": [
        {"role": "system", "content": sys},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


def load_alpaca():
    path = hf_hub_download("IlyaGusev/ru_turbo_alpaca",
                           "ru_turbo_alpaca.jsonl.zst", repo_type="dataset")
    out = []
    with open(path, "rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if str(ex.get("label", "")).lower() == "bad":
                continue
            instr = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            ans = (ex.get("output") or "").strip()
            if not instr or not ans:
                continue
            q = instr if not inp else f"{instr}\n\n{inp}"
            out.append(rec(SYS_LOW, q, ans))      # alpaca = low-effort, direct
    random.shuffle(out)
    return out[:N_ALPACA]


def main():
    distilled = [json.loads(l) for l in open(DATA.parent / "data7_distill.jsonl")]
    high = [r for r in distilled if "high" in r["messages"][0]["content"]]
    low = [r for r in distilled if "low" in r["messages"][0]["content"]]

    train = []
    for _ in range(25):                              # #4 identity over-rep
        train += [rec(SYS_LOW, q, a) for q, a in IDENTITY]
    train += high * 3                                # boost rare high-effort
    train += low * 2
    for _ in range(2):
        train += [rec(SYS_LOW, q, a) for q, a in GENERAL]
    train += load_alpaca()                           # fluency
    random.shuffle(train)

    valid = ([rec(SYS_LOW, q, a) for q, a in IDENTITY[:3]]
             + [rec(SYS_LOW, q, a) for q, a in GENERAL[:3]]
             + high[:2] + low[:2])

    DATA.mkdir(exist_ok=True)
    with (DATA / "train.jsonl").open("w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (DATA / "valid.jsonl").open("w") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train={len(train)} (identity={25*len(IDENTITY)}, high={len(high)*3}, "
          f"low={len(low)*2}, general={2*len(GENERAL)}, alpaca={N_ALPACA}) "
          f"valid={len(valid)} -> {DATA}")


if __name__ == "__main__":
    main()
