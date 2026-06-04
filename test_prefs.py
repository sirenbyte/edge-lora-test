"""Demo: preference profile — tastes learned from BEHAVIOR (memory+recommender,
NO model training). Seed listening history → vague request resolves to favorite;
explicit play logs (learning loop)."""
import os
import sys

import prefs

# fresh profile for the demo
try:
    os.remove(prefs.PREFS_FILE)
except OSError:
    pass

# simulate past listening behavior (implicit positive signals)
for _ in range(3):
    prefs.log_play("lo-fi")
prefs.log_play("джаз")
print("Профиль после прослушиваний:", prefs.summary())

from agent import Agent  # noqa: E402  (import after seeding)
a = Agent()
for q in ["Поставь что-нибудь.", "Поставь на свой вкус.", "Включи эмбиент."]:
    r = a.respond(q)
    print(f"\n>>> {q}")
    if r["tool"]:
        print(f"   TOOL {r['tool']} -> {r['result']}")
    print(f"   {r['answer'][:160]}")
print("\nПрофиль в конце:", prefs.summary())
