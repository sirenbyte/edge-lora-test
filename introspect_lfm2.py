"""Dump LFM2-MoE parameter key templates to find the right LoRA target keys.

mlx-lm #571: default LoRA on lfm2_moe attaches only to attention (~0.02%).
We must list the expert projection keys explicitly. This prints the
de-indexed parameter name templates + counts so we can pick keys.
"""

import collections
import re
import sys

from mlx.utils import tree_flatten
from mlx_lm import load

path = sys.argv[1]
model, _ = load(path)

keys = [k for k, _ in tree_flatten(model.parameters())]
tmpl = collections.Counter()
for k in keys:
    s = re.sub(r"\.\d+\.", ".N.", k)
    tmpl[s] += 1

print(f"total param tensors: {len(keys)}")
print("=== parameter name templates (count, template) ===")
for s, c in sorted(tmpl.items()):
    print(f"{c:5d}  {s}")

print("\n=== templates mentioning experts/moe/proj ===")
for s, c in sorted(tmpl.items()):
    if any(w in s.lower() for w in ("expert", "moe", "switch", "gate", "proj")):
        print(f"{c:5d}  {s}")
