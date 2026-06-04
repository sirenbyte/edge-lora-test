"""Decisive test of the calibration-only KV-quant scheme (no retraining):
per-channel K (pre-RoPE) + per-token V, at 2/3-bit, on the 32k needle.

Naive per-token 2-bit broke retrieval. If per-channel-K-pre-RoPE 2-bit now
retrieves the needle -> the proper KV-quant scheme works on frozen LFM2.
"""

import os
import re
import subprocess

prompt = open("needle_prompt.txt").read()
MIXED = "LFM2.5-8B-A1B-mlx-mixed"
AD = "adapters_lfm2_v8"

RUNS = [
    ("fp16 (ref)", {}),
    ("grouped-ROT 2bit", {"KVQ": "1", "KVQ_BITS": "2", "KVQ_ROT": "2"}),
    ("grouped-ROT 3bit", {"KVQ": "1", "KVQ_BITS": "3", "KVQ_ROT": "2"}),
]

ntok = len(prompt)
print(f"32k needle | per-channel-K(pre-RoPE) + per-token-V | prompt ~{ntok} chars\n")
print(f"{'config':16} | recall | peak GB | answer")
print("-" * 70)
for name, env in RUNS:
    cmd = ["mlx_lm.generate", "--model", MIXED,   # base only — KV-quant is cache-level, adapter-independent
           "--prompt", "-", "--max-tokens", "700", "--temp", "0.0"]
    e = dict(os.environ)
    e.update(env)
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=e)
    out = r.stdout
    segs = re.findall(r"==========\n(.*?)\n==========", out, re.S)
    full = segs[0] if segs else ""
    ans = re.sub(r"<think>.*?</think>", "", full, flags=re.S).strip()
    recall = "водопад" in full.lower() or "451" in full
    peak = re.search(r"Peak memory: ([\d.]+) GB", out)
    print(f"{name:16} | {'OK ' if recall else '.. '}   | "
          f"{peak.group(1) if peak else '?':>6} | {(ans or full)[:42]!r}")
