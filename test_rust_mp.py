"""Test Rust Memory Plant (memory_plant_rs) facts + latency vs Python bridge."""
import subprocess
import time

import memory_plant_rs as m

DIR = "/Users/abzaltuganbay/projects/edge-lora-test"

# ---- Rust mp (native, in-process) ----
pm = m.PersonalMemory("demo")
pm.store_fact("name", "Абзал")
pm.store_fact("hobby", "фотография")
pm.store_fact("lang", "Python")
ext = pm.ingest("Я люблю фотографию и работаю программистом.")

print("=== RUST mp (memory_plant_rs) ===")
print("ingest auto-extracted:", ext)
print("recall name  =", repr(pm.recall("name")))
print("recall hobby =", repr(pm.recall("hobby")))
print("all_facts    =", pm.all_facts())
pm.forget("hobby")
print("after forget('hobby') -> recall hobby =", repr(pm.recall("hobby")), "(provable forget)")

# latency: native recall
N = 200
t0 = time.perf_counter()
for _ in range(N):
    pm.recall("name")
    pm.recall("lang")
rust_ms = (time.perf_counter() - t0) / N * 1000
print(f"RUST recall x2: {rust_ms:.4f} ms  (avg of {N})")

# ---- Python mp bridge (subprocess; loads sentence-transformers each call) ----
print("\n=== PYTHON mp bridge (subprocess) ===")
t0 = time.perf_counter()
out = subprocess.run(["python3", "mp_bridge.py", "search", "как меня зовут"],
                     capture_output=True, text=True, cwd=DIR).stdout.strip()
py_ms = (time.perf_counter() - t0) * 1000
print(f"PY bridge search: {py_ms:.0f} ms -> {out[:60]!r}")

print(f"\n>>> speedup: Rust recall ~{rust_ms:.3f} ms vs Python bridge ~{py_ms:.0f} ms "
      f"(~{py_ms / max(rust_ms, 1e-6):.0f}x)")
