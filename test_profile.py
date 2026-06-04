"""Demo: extended preference/behavior profile (pure logic, no model) —
time-of-day, skip/like weights, recency decay, multi-category + insights."""
import datetime as _dt
import os

import prefs

try:
    os.remove(prefs.PREFS_FILE)
except OSError:
    pass

now = _dt.datetime.now()


def ago(days):
    return now - _dt.timedelta(days=days)


# --- music with time-of-day context ---
for _ in range(3):
    prefs.log("music", "lo-fi", "play", ts=now.replace(hour=22))      # evening
for _ in range(2):
    prefs.log("music", "энергичное", "play", ts=now.replace(hour=8))  # morning
prefs.log("music", "lo-fi", "like")        # explicit like → +2
prefs.log("music", "рэп", "skip")          # skip → −1 (avoid)

print("music overall :", prefs.summary("music"))
print("evening fav   :", prefs.top("music", "evening"))
print("morning fav   :", prefs.top("music", "morning"))

# --- recency: old джаз (90d ago) should LOSE to recent lo-fi ---
for _ in range(5):
    prefs.log("music", "джаз", "play", ts=ago(90))
print("overall top after old джаз×5(90d):", prefs.top("music"), "(recency: lo-fi keeps lead)")

# --- non-music categories → life-optimizer ---
for _ in range(3):
    prefs.log("activity", "пробежка", "complete", ts=now.replace(hour=7))
for _ in range(2):
    prefs.log("food", "овсянка", "choose", ts=now.replace(hour=8))
prefs.log("content", "подкаст про ИИ", "like", ts=now.replace(hour=18))

print("\n=== INSIGHTS (proactive patterns) ===")
for ins in prefs.insights():
    print(f"  [{ins['score']:+.1f}] {ins['text']}")

print("\nback-compat (agent.py): log_play/top_music ->",
      prefs.log_play("джаз"), "|", prefs.top_music())
