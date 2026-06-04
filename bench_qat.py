"""Decisive test: does adapter-QAT recover 2-bit KV retrieval at 32k?

Compares on the same 32k needle prompt:
  - v8 + QKV_BITS=2 (no QAT for 2-bit)  -> expected FAIL (naive 2-bit)
  - v9 + QKV_BITS=2 (QAT-trained)        -> hypothesis: RECOVERS
  - v8 + free mlx --kv-bits 4            -> reference (known good)
Thinking ON, needle checked anywhere in output.
"""

import os
import re
import subprocess

prompt = open("needle_prompt.txt").read()
MIXED = "LFM2.5-8B-A1B-mlx-mixed"

RUNS = [
    ("v8 + 2bit (NO QAT)", "adapters_lfm2_v8", {"QKV_BITS": "2"}, []),
    ("v9 + 2bit (QAT)", "adapters_lfm2_v9_qat2bit", {"QKV_BITS": "2"}, []),
    ("v8 + free kv4", "adapters_lfm2_v8", {}, ["--kv-bits", "4", "--quantized-kv-start", "0"]),
]

print(f"32k needle | {'config':22} | recall | peak GB | answer")
print("-" * 78)
for name, adapter, env, flags in RUNS:
    cmd = ["mlx_lm.generate", "--model", MIXED, "--adapter-path", adapter,
           "--prompt", "-", "--max-tokens", "500", "--temp", "0.0"] + flags
    e = dict(os.environ)
    e.update(env)
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=e)
    out = r.stdout
    segs = re.findall(r"==========\n(.*?)\n==========", out, re.S)
    full = segs[0] if segs else ""
    ans = re.sub(r"<think>.*?</think>", "", full, flags=re.S).strip()
    recall = "водопад" in full.lower() or "451" in full
    peak = re.search(r"Peak memory: ([\d.]+) GB", out)
    print(f"{'':11}| {name:22} | {'OK ' if recall else '.. '}   | "
          f"{peak.group(1) if peak else '?':>6} | {(ans or full)[:40]!r}")
