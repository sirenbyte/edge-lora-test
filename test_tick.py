"""Demo: Agent.tick() — proactive nudge phrased LIVE by the 4B (template fallback)."""
import datetime as _dt
import os

import prefs
import proactive
from agent import Agent

for f in (prefs.PREFS_FILE, proactive.NUDGE_LOG):
    try:
        os.remove(f)
    except OSError:
        pass

base = _dt.datetime.now().replace(minute=0, second=0, microsecond=0)


def at(day_ago, hour):
    return base.replace(hour=hour) - _dt.timedelta(days=day_ago)


# seed a week of confident morning/evening habits
for d in range(7):
    prefs.log("activity", "пробежка", "complete", ts=at(d, 7))
    prefs.log("music", "lo-fi", "play", ts=at(d, 21))

a = Agent()

print("\n=== tick @ ночь 03:00 ===")
print("  ", a.tick(now=base.replace(hour=3)) or "(молчит — тихие часы)")

print("\n=== tick @ утро 07:00 (4B формулирует вживую) ===")
n = a.tick(now=base.replace(hour=7))
print("   нудж:", f"\"{n['say']}\"" if n else "(молчит)")
print("   (привычка:", n["text"], "| score", n["score"], ")")

print("\n=== feedback: пользователь согласился ===")
print("  ", a.nudge_feedback(n, accepted=True))

print("\n=== tick @ вечер 19:00 ===")
n2 = a.tick(now=base.replace(hour=19))
print("   нудж:", f"\"{n2['say']}\"" if n2 else "(молчит)")

print("\n=== feedback: пользователь отмахнулся ===")
if n2:
    print("  ", a.nudge_feedback(n2, accepted=False))
print("   music score после:", prefs.summary("music"))
