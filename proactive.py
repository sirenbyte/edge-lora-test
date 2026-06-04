"""Proactive triggers — turn behavior INSIGHTS into gentle, well-timed nudges.

Bridges prefs.insights() → suggestions the assistant offers on a schedule.
Restraint by design (guardrails):
  - QUIET hours (night): never nudge.
  - COOLDOWN: same (cat,item) nudge not repeated within COOLDOWN_HRS.
  - DAILY_CAP: at most N proactive nudges per day total.
  - CONFIDENCE: only clear habits (>= MIN_SCORE) become nudges.
  - ONE-AT-A-TIME: surface the single best due nudge, never a barrage.
  - LEARNS: accept() reinforces, dismiss() feeds a NEGATIVE signal back to prefs.
Transparent + dismissible. Memory + rules, NO model — the 4B only PHRASES the text.

The phone OS scheduler (WorkManager / BGTaskScheduler) calls tick()/due() at
daypart boundaries or hourly; here `now` is injectable so it's testable offline.
"""
import datetime as _dt
import json

import prefs

NUDGE_LOG = "/Users/abzaltuganbay/projects/edge-lora-test/nudges.json"
COOLDOWN_HRS = 6.0          # don't repeat the same nudge within 6h
DAILY_CAP = 3               # at most 3 proactive nudges per day
MIN_SCORE = 2.5             # confidence gate (higher than insights' default 1.5)
# item-specific categories → use the EXACT template (the precise item matters,
# e.g. "lo-fi"/"овсянка"/"пробежка"); 'soft' categories (content/mood) let the
# 4B phrase it warmly.
TEMPLATE_CATS = {"music", "food", "activity"}

_GREET = {"morning": "Доброе утро! ", "evening": "Добрый вечер! ", "day": "", "night": ""}
_BODY = {
    "activity": "обычно сейчас у тебя {item} — пора? 💪",
    "music":    "поставить {item}? 🎵",
    "food":     "на завтрак {item}, как обычно? 🥣",
    "content":  "включить «{item}»? 🎧",
}
_DEFAULT_BODY = "обычно {dp_low} у тебя {item} — напомнить?"


def _now():
    return _dt.datetime.now()


def _load():
    try:
        return json.load(open(NUDGE_LOG))
    except Exception:
        return {"sent": []}


def _save(d):
    json.dump(d, open(NUDGE_LOG, "w"), ensure_ascii=False)


def _sent(now):
    out = []
    for r in _load().get("sent", []):
        try:
            out.append((_dt.datetime.fromisoformat(r["ts"]), r))
        except Exception:
            continue
    return out


def _today_count(now):
    return sum(1 for t, _ in _sent(now) if t.date() == now.date())


def _on_cooldown(cat, item, now):
    for t, r in _sent(now):
        if r.get("cat") == cat and r.get("item") == item:
            if 0 <= (now - t).total_seconds() / 3600.0 < COOLDOWN_HRS:
                return True
    return False


def phrase(nudge):
    """Fallback phrasing (in prod the 4B rephrases naturally from the insight)."""
    dp = nudge.get("dp")
    body = _BODY.get(nudge["cat"], _DEFAULT_BODY).format(
        item=nudge["item"], dp_low=prefs._DAYPART_RU.get(dp, dp))
    body = body[:1].upper() + body[1:]
    return _GREET.get(dp, "") + body


def due(now=None, dp=None):
    """Best single proactive nudge for this moment, or None.
    Honors quiet-hours, daily cap, cooldown, confidence, one-at-a-time."""
    now = now or _now()
    dp = dp or prefs.daypart(now)
    if dp == "night":                          # quiet hours
        return None
    if _today_count(now) >= DAILY_CAP:         # daily cap
        return None
    for ins in prefs.insights(min_score=MIN_SCORE):   # confident, sorted by score
        if ins["dp"] != dp:                    # only this daypart's habits
            continue
        if _on_cooldown(ins["cat"], ins["item"], now):
            continue
        return {**ins, "say": phrase(ins)}     # one-at-a-time: first wins
    return None


def mark_sent(nudge, now=None):
    now = now or _now()
    d = _load()
    d["sent"].append({"cat": nudge["cat"], "item": nudge["item"],
                      "dp": nudge.get("dp"), "ts": now.isoformat()})
    _save(d)


def accept(nudge, ts=None):
    """User acted on the nudge → reinforce the habit (positive signal)."""
    return prefs.log(nudge["cat"], nudge["item"], "like", ts=ts)


def dismiss(nudge, ts=None):
    """User waved it off → learn to back off (negative signal)."""
    return prefs.log(nudge["cat"], nudge["item"], "skip", ts=ts)
