"""Local end-to-end smoke — exercise EVERY subsystem on the real Qwen3.5-4B
before any iPhone packaging. Loads the model ONCE, runs representative inputs
across tools / memory / proactive / hobby-digest / modes, prints a checklist.

Run:  .venv/bin/python smoke_local.py
"""
import datetime as _dt
import os
import time

import hobby_pack
import prefs
import proactive
from agent import Agent

# --- reproducible state: clean runtime logs, seed one confident morning habit ---
for f in (proactive.NUDGE_LOG, prefs.PREFS_FILE, "mp_docs.bin"):
    try:
        os.remove(f)
    except OSError:
        pass
_base = _dt.datetime.now().replace(minute=0, second=0, microsecond=0)
for d in range(7):                                   # 7 days → confident habit
    prefs.log("activity", "разминка", "complete",
              ts=(_base.replace(hour=7) - _dt.timedelta(days=d)))

print("=" * 64)
print("QIYAS EDGE — LOCAL SMOKE (pre-iPhone)")
print("=" * 64)
t0 = time.time()
a = Agent()


def step(title, q):
    t = time.time()
    r = a.respond(q)
    dt = time.time() - t
    tool = f"  ⚙ {r['tool']} → {r['result']}" if r.get("tool") else ""
    print(f"\n▶ {title}  [{r['mode']}|think={r['think']}|{dt:.1f}s]{tool}")
    print(f"  Q: {q}")
    print(f"  A: {r['answer'][:200]}")


print("\n--- 1. TOOLS (execution loop) ---")
step("calc (forced)", "Сколько будет 15% от 240?")
step("datetime", "Какое сегодня число и время?")
step("device", "Выключи свет в спальне.")
step("reminder", "Напомни позвонить маме в 18:00.")
step("note", "Запиши заметку: купить молоко и хлеб.")
step("web_search", "Что нового в Python 3.13?")

print("\n--- 2. COMPANION / CREATIVE (right hemisphere) ---")
step("companion", "Мне сегодня немного грустно.")
step("creative", "Придумай короткий слоган для приложения-ассистента.")

print("\n--- 3. MEMORY (in-process e5 + Rust, cross-lingual) ---")
for s in ["Меня зовут Абзал.", "Я увлекаюсь пейзажной фотографией.", "Я живу в Алматы."]:
    a.ingest(s)
for q in ["как меня зовут?", "какое у меня хобби?", "где я живу?"]:
    top = a.retrieve(q).splitlines()
    print(f"  retrieve({q!r}) -> {top[0] if top else '(none)'!r}")

print("\n--- 4. PROACTIVE (restraint) ---")
n = a.nudge_tick(now=_base.replace(hour=7))
print(f"  morning 07:00 -> {n['say'] if n else '(silent)'}")
n2 = a.nudge_tick(now=_base.replace(hour=3))
print(f"  night   03:00 -> {n2['say'] if n2 else '(silent ✓ quiet hours)'}")

print("\n--- 5. HOBBY DIGEST (web-fed, grounded, cited) ---")
digest = hobby_pack.generate_digest("пейзажная фотография", agent=a)
print("\n".join("  " + ln for ln in digest.splitlines()[:8]))

print("\n" + "=" * 64)
print(f"SMOKE DONE in {time.time() - t0:.0f}s — review the checklist above.")
print("=" * 64)
