"""Demo: proactive HOBBY-DIGEST nudge. No habit due → assistant offers a fresh
digest for the user's hobby pack; on accept it generates the digest inline.
Deterministic (no TTY): clears prefs/nudge-log, asserts the nudge, then generates."""
import os

import hobby_pack
import prefs
import proactive
from agent import Agent

for f in (prefs.PREFS_FILE, proactive.NUDGE_LOG):
    try:
        os.remove(f)
    except OSError:
        pass

a = Agent()
n = a.nudge_tick()
print(f"\n   💡 {n['say'] if n else '(молчит)'}   [cat={n['cat'] if n else None}]")
assert n and n["cat"] == "digest", "expected a digest nudge when no habit is due"

print("   — пользователь: да —\n")
print(hobby_pack.generate_digest(n["item"], agent=a))
