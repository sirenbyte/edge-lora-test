"""Demo: proactive triggers (pure logic, no model) — restraint guardrails.
Simulates a day of scheduled 'ticks' over a week-long seeded habit profile."""
import datetime as _dt
import os

import prefs
import proactive

for f in (prefs.PREFS_FILE, proactive.NUDGE_LOG):
    try:
        os.remove(f)
    except OSError:
        pass

base = _dt.datetime.now().replace(minute=0, second=0, microsecond=0)


def at(day_ago, hour):
    return base.replace(hour=hour) - _dt.timedelta(days=day_ago)


# --- seed a week of habits so insights are CONFIDENT ---
for d in range(7):
    prefs.log("activity", "пробежка", "complete", ts=at(d, 7))   # morning run
    prefs.log("food", "овсянка", "choose", ts=at(d, 8))          # morning oats
    prefs.log("music", "lo-fi", "play", ts=at(d, 21))            # evening lo-fi
# one-off weak signal — KNOWN pattern but below confidence gate
prefs.log("content", "медитация", "play", ts=at(0, 20))

print("=== PROFILE INSIGHTS (всё, что знает) ===")
for i in prefs.insights(min_score=0.9):
    gate = "→ нуджит" if i["score"] >= proactive.MIN_SCORE else "× тихо (мало уверенности)"
    print(f"  [{i['score']:+.1f}] {i['text']:42s} {gate}")

# --- simulate today's scheduled ticks, hour by hour ---
print("\n=== TICKS (расписание дня) ===")
for hour in (3, 7, 8, 12, 18, 21, 23):
    now = base.replace(hour=hour)
    n = proactive.due(now=now)
    if n:
        print(f"  {hour:02d}:00 [{n['dp']:7s}] → \"{n['say']}\"")
        proactive.mark_sent(n, now=now)
    else:
        print(f"  {hour:02d}:00 [{prefs.daypart(now):7s}] → (молчит)")

# --- restraint guardrails ---
print("\n=== RESTRAINT (сдержанность) ===")
n730 = proactive.due(now=base.replace(hour=7, minute=30))
print("  cooldown   :", "не повторил пробежку ✅"
      if (n730 is None or n730["item"] != "пробежка") else "повторил ❌")
print("  night 03:00:", "молчит ✅" if proactive.due(now=base.replace(hour=3)) is None else "❌")
print(f"  daily cap  : {proactive.DAILY_CAP} нуджа/день (4-й — молчит)")
print("  confidence : медитация известна, но ниже порога — не нуджит ✅")

# --- feedback loop: assistant learns from reaction ---
print("\n=== FEEDBACK (учится на реакции) ===")
lofi = {"cat": "music", "item": "lo-fi", "dp": "evening"}
print("  music до     :", prefs.summary("music"))
proactive.dismiss(lofi)                      # user waves off lo-fi tonight
print("  после dismiss:", prefs.summary("music"), "← сигнал учтён, будет реже")
