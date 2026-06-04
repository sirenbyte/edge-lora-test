"""Demo: interactive chat() with a proactive nudge surfacing mid-chat.
Deterministic: time pinned to morning, habit seeded, input scripted (no TTY)."""
import builtins
import datetime as _dt
import os

import prefs
import proactive

for f in (prefs.PREFS_FILE, proactive.NUDGE_LOG):
    try:
        os.remove(f)
    except OSError:
        pass

# pin "now" to a fixed morning so a morning habit is due (chat() uses real now)
FIXED = _dt.datetime.now().replace(hour=7, minute=0, second=0, microsecond=0)
prefs._now = lambda: FIXED
proactive._now = lambda: FIXED

# seed a confident morning habit
for d in range(7):
    prefs.log("activity", "разминка", "complete", ts=FIXED - _dt.timedelta(days=d))

import agent  # noqa: E402  (import after monkeypatch so it sees patched modules)

# scripted stdin: [answer to opening nudge] , [a chat turn] , [exit]
_script = iter(["да", "Привет!", "выход"])
builtins.input = lambda *a, **k: next(_script)

agent.chat()
