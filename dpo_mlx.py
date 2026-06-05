"""Minimal DPO LoRA trainer on MLX for the 4-bit Qwen3.5-4B.

RuBLiMP showed the 4B already PREFERS correct Russian forms (93% in log-prob) but
still GENERATES wrong ones — a decoding gap. DPO closes it by pushing generation
toward the preferred (chosen) form. Efficiency trick: LoRA inits to zero, so the
REFERENCE log-probs == the base forward at init → precompute once (frozen); then only
the policy forward runs each step. Sequences are short (single sentences) → fast, fits.

  python dpo_mlx.py [epochs]      # -> adapters_dpo/ ; eval: eval_rublimp.py adapters_dpo
  python dpo_mlx.py smoke         # 8 pairs / 1 epoch sanity
"""
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from mlx_lm.tuner.utils import linear_to_lora_layers
from vision_unload import load_text_only

BASE = "mlx-community/Qwen3.5-4B-MLX-4bit"
ADAPTER_DIR = Path("adapters_dpo")
NUM_LAYERS = 8
LORA_CFG = {"rank": 8, "scale": 16.0, "dropout": 0.0}
LR = 1e-6          # 1e-5 collapsed the model — DPO needs a gentle lr
BETA = 0.3         # higher beta keeps the policy closer to ref (anti-collapse)
LAMBDA = 0.1       # NLL anchor on chosen → keeps GENERATION coherent (RPO-style)


def seq_logp(model, ids):
    """Sum log-prob of the token sequence under `model` (differentiable)."""
    logits = model(ids[None])[0].astype(mx.float32)        # (T, V)
    logp = nn.log_softmax(logits[:-1], axis=-1)
    return mx.take_along_axis(logp, ids[1:, None], axis=-1).sum()


def main():
    smoke = "smoke" in sys.argv
    epochs = 1 if smoke else (int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3)
    model, tok, stats = load_text_only(BASE)
    print(stats)
    model.freeze()
    linear_to_lora_layers(model, NUM_LAYERS, LORA_CFG)
    model.train()                       # DeltaNet: use_kernel=not training → differentiable path

    rows = [json.loads(l) for l in open("data_dpo/train.jsonl", encoding="utf-8")]
    if smoke:
        rows = rows[:8]
    print(f"DPO pairs: {len(rows)} | epochs={epochs} | beta={BETA} lr={LR}")

    # 1) reference log-probs (LoRA==0 at init → == base), computed once, frozen
    ref = []
    for r in rows:
        c = mx.array(tok.encode(r["chosen"]))
        j = mx.array(tok.encode(r["rejected"]))
        rc, rj = seq_logp(model, c), seq_logp(model, j)
        mx.eval(rc, rj)
        ref.append((c, j, float(rc), float(rj)))

    def loss_fn(model, c, j, rc, rj):
        pc = seq_logp(model, c)
        margin = BETA * ((pc - rc) - (seq_logp(model, j) - rj))
        nll = -pc / (c.shape[0] - 1)                  # RPO anchor: keep chosen likely
        return -nn.log_sigmoid(margin) + LAMBDA * nll, margin

    lvg = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=LR)
    rng = random.Random(13)
    t0 = time.time()
    for ep in range(epochs):
        order = list(range(len(ref)))
        rng.shuffle(order)
        tot = acc = 0
        for i in order:
            c, j, rc, rj = ref[i]
            (l, margin), g = lvg(model, c, j, rc, rj)
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
            tot += float(l)
            acc += float(margin) > 0
        print(f"epoch {ep + 1}/{epochs}: loss={tot / len(ref):.4f} "
              f"pref-acc={100 * acc / len(ref):.0f}%  ({time.time() - t0:.0f}s)")

    ADAPTER_DIR.mkdir(exist_ok=True)
    flat = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(ADAPTER_DIR / "adapters.safetensors"), flat)
    (ADAPTER_DIR / "adapter_config.json").write_text(json.dumps(
        {"fine_tune_type": "lora", "num_layers": NUM_LAYERS, "lora_parameters": LORA_CFG},
        ensure_ascii=False, indent=2))
    print(f"saved adapter -> {ADAPTER_DIR}  (eval: python eval_rublimp.py adapters_dpo)")


if __name__ == "__main__":
    main()
