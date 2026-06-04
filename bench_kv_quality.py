"""Needle-in-haystack quality test for KV-cache quantization on LFM2.5.

Hides a secret code early in a long (~8k token) document, asks for it at the
end. If KV quant degrades recall, the model fails. Measures recall + peak
memory + speed across fp16/kv8/kv4/kv2, and a combined run (mixed base +
v8 adapter + kv2) to confirm the whole stack works together.
"""

import json
import random
import re
import subprocess

random.seed(1)

# --- build the long needle prompt ---
rows = [json.loads(l) for l in open("data6/train.jsonl")]
texts = [m["content"] for r in rows for m in r["messages"]
         if m["role"] == "assistant" and len(m["content"]) > 80]
random.shuffle(texts)
blob = " ".join(texts)
while len(blob) < 115000:                    # need ~100k chars ≈ ~32k tokens
    blob += " " + blob
NEEDLE = ("\n\nВАЖНО, ЗАПОМНИ ЭТО: секретное кодовое слово документа — "
          "ВОДОПАД-451. Запомни его точно.\n\n")
prompt = ("Прочитай документ и запомни детали.\n\n"
          + blob[:900] + NEEDLE + blob[900:100000]
          + "\n\nВОПРОС: какое секретное кодовое слово было в документе? "
            "Ответь только этим словом.")
open("needle_prompt.txt", "w").write(prompt)

from distill_high_effort import SYS_LOW

MIXED = "LFM2.5-8B-A1B-mlx-mixed"            # our deployable stack: mixed base + v8
ADAPTER = "adapters_lfm2_v8"
KV = lambda b: ["--kv-bits", b, "--quantized-kv-start", "0"]

# All on the deployable model with SYS_LOW (direct answers, no <think>) so
# recall is clean. Vary only KV bits to isolate KV-quant quality.
RUNS = [
    ("fp16 (baseline)", []),
    ("kv4", KV("4")),
    ("kv2", KV("2")),
]

from mlx_lm import load as _load
_, _tok = _load(MIXED)
ntok = len(_tok.encode(prompt))
# THINKING ALLOWED (no system, no adapter) so the model can scan & retrieve;
# clean test of whether KV-quant degrades long-ctx recall vs fp16.
print(f"prompt: ~{len(prompt)} chars = {ntok} tokens | mixed base, thinking ON\n")
print(f"{'config':16} | recall | peak GB | tok/s | answer")
print("-" * 70)
for name, flags in RUNS:
    cmd = ["mlx_lm.generate", "--model", MIXED, "--prompt", "-",
           "--max-tokens", "500", "--temp", "0.0"] + flags
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True)
    out = r.stdout
    segs = re.findall(r"==========\n(.*?)\n==========", out, re.S)
    full = segs[0] if segs else ""                      # full generation (incl. think)
    ans = re.sub(r"<think>.*?</think>", "", full, flags=re.S).strip()
    recall = "водопад" in full.lower() or "451" in full  # needle anywhere = retrieved
    peak = re.search(r"Peak memory: ([\d.]+) GB", out)
    gen = re.search(r"Generation: \d+ tokens, ([\d.]+)", out)
    shown = (ans or full)[:50]
    print(f"{name:22} | {'OK ' if recall else '.. '}   | "
          f"{peak.group(1) if peak else '?':>6} | "
          f"{gen.group(1) if gen else '?':>5} | {shown!r}")
