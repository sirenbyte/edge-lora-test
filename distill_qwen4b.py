"""Self-distill reasoning-effort (dial) pairs FROM Qwen3.5-4B itself.

Native <think> from the 4B → HIGH-effort examples. Think stripped → LOW
(same Q, direct answer). Teaches the dial to gate reasoning on a system flag.
Re-distilling from the 4B (not LFM2) gives Qwen's native thinking style.
"""

import json
import re
from pathlib import Path

from mlx_lm import generate, load

from distill_high_effort import SYS_HIGH, SYS_LOW, QUESTIONS

BASE = "mlx-community/Qwen3.5-4B-MLX-4bit"
OUT = Path(__file__).parent / "data_qwen4b_distill.jsonl"


def rec(sys, q, a):
    return {"messages": [
        {"role": "system", "content": sys},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


def main():
    model, tok = load(BASE)
    n_high = n_low = 0
    with OUT.open("w") as f:
        for q in QUESTIONS:
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": SYS_HIGH},
                 {"role": "user", "content": q}],
                add_generation_prompt=True)
            full = generate(model, tok, prompt=prompt,
                            max_tokens=512, verbose=False).strip()
            final = re.sub(r"<think>.*?</think>", "", full, flags=re.S).strip()
            if not final:                      # all think, no answer -> skip
                continue
            if "<think>" in full:
                f.write(json.dumps(rec(SYS_HIGH, q, full), ensure_ascii=False) + "\n")
                n_high += 1
            f.write(json.dumps(rec(SYS_LOW, q, final), ensure_ascii=False) + "\n")
            n_low += 1
    print(f"distilled from {BASE}: high={n_high} low={n_low} -> {OUT}")


if __name__ == "__main__":
    main()
