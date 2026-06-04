"""Does mlx NATIVE --kv-bits 3 (real packed cache, saves memory) preserve
32k needle retrieval? If yes, integration = a flag, no custom code."""
import re
import subprocess

prompt = open("needle_prompt.txt").read()
M = "LFM2.5-8B-A1B-mlx-mixed"
RUNS = [("fp16", None), ("native kv4", "4"), ("native kv3", "3"), ("native kv2", "2")]

print(f"{'config':12} | recall | peak GB | answer")
print("-" * 64)
for name, bits in RUNS:
    cmd = ["mlx_lm.generate", "--model", M, "--prompt", "-",
           "--max-tokens", "700", "--temp", "0.0"]
    if bits:
        cmd += ["--kv-bits", bits, "--quantized-kv-start", "0"]
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True)
    out = r.stdout
    segs = re.findall(r"==========\n(.*?)\n==========", out, re.S)
    full = segs[0] if segs else ""
    ans = re.sub(r"<think>.*?</think>", "", full, flags=re.S).strip()
    recall = "водопад" in full.lower() or "451" in full
    peak = re.search(r"Peak memory: ([\d.]+) GB", out)
    print(f"{name:12} | {'OK ' if recall else '.. '}   | "
          f"{peak.group(1) if peak else '?':>6} | {(ans or full)[:38]!r}")
