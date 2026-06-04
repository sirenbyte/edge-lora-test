"""v5 dataset: domain facts + general replay + ru_turbo_alpaca for fluency.

The v3 typos (Токайо/Толстый) come from the adapter perturbing a 4-bit
Cyrillic generator after seeing only 75 examples. Mixing in real Russian
instruction data (IlyaGusev/ru_turbo_alpaca) gives the adapter enough
correct-Cyrillic signal that the glitches disappear, while domain facts
(repeated) still stick. Experts stay frozen at train time (v3 recipe).
"""

import io
import json
import random
from pathlib import Path

import zstandard
from huggingface_hub import hf_hub_download

from make_data_v2 import DOMAIN, GENERAL  # lists of (question, answer)

random.seed(13)
DATA = Path(__file__).parent / "data6"
N_ALPACA = 1000           # v6: less alpaca + more domain weight (v5 diluted domain)
DOMAIN_REPEAT = 40


def rec(q, a):
    return {"messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


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
            if str(ex.get("label", "")).lower() == "bad":   # drop flagged-bad
                continue
            instr = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            ans = (ex.get("output") or "").strip()
            if not instr or not ans:
                continue
            q = instr if not inp else f"{instr}\n\n{inp}"
            out.append(rec(q, ans))
    random.shuffle(out)
    return out[:N_ALPACA]


def main():
    alpaca = load_alpaca()
    train = []
    for _ in range(DOMAIN_REPEAT):           # domain ×40 (~26% of mix)
        train += [rec(q, a) for q, a in DOMAIN]
    for _ in range(2):                       # curated general ×2
        train += [rec(q, a) for q, a in GENERAL]
    train += alpaca                          # ~1000 real Russian instructions
    random.shuffle(train)

    valid = ([rec(q, a) for q, a in DOMAIN[:4] + GENERAL[:4]] + alpaca[:4])

    DATA.mkdir(exist_ok=True)
    with (DATA / "train.jsonl").open("w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (DATA / "valid.jsonl").open("w") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train={len(train)} (domain={DOMAIN_REPEAT*len(DOMAIN)}, "
          f"general={2*len(GENERAL)}, alpaca={len(alpaca)}) "
          f"valid={len(valid)} -> {DATA}")


if __name__ == "__main__":
    main()
