"""Preference & behavior profile — tastes/habits learned from BEHAVIOR.

General event log (any category, not just music) with:
  - signed SIGNALS (play/like/complete = +, skip/dislike = −) and weights,
  - TIME-OF-DAY context (morning/day/evening/night),
  - RECENCY decay (tastes evolve; old signals fade),
  - INSIGHTS (dominant item per category+daypart) → proactive life-optimizer.
Memory + recommender, NO model training. On-device, private.

Back-compat API kept for agent.py: log_play(query), top_music(), summary(), PREFS_FILE.
"""
import datetime as _dt
import json

from config import PREFS_FILE

SIGNAL_W = {"play": 1.0, "choose": 1.0, "complete": 1.5, "finish": 1.5,
            "like": 2.0, "skip": -1.0, "dislike": -2.0}
HALF_LIFE_DAYS = 30.0                      # weight halves every 30 days

_NORM = {  # music genre normalization
    "лоуфай": "lo-fi", "лофай": "lo-fi", "лоу-фай": "lo-fi", "lofi": "lo-fi", "lo-fi": "lo-fi",
    "джаз": "джаз", "jazz": "джаз", "классик": "классика", "classic": "классика",
    "рок": "рок", "rock": "рок", "рэп": "рэп", "rap": "рэп", "хип-хоп": "рэп",
    "поп": "поп", "pop": "поп", "эмбиент": "эмбиент", "ambient": "эмбиент",
    "спокойн": "спокойное", "релакс": "спокойное", "chill": "спокойное",
    "электрон": "электроника", "techno": "электроника",
}
_DAYPART_RU = {"morning": "по утрам", "day": "днём", "evening": "по вечерам", "night": "ночью"}


def _now():
    return _dt.datetime.now()


def daypart(dt=None):
    h = (dt or _now()).hour
    if 5 <= h < 11:
        return "morning"
    if 11 <= h < 17:
        return "day"
    if 17 <= h < 23:
        return "evening"
    return "night"


def _norm_music(q):
    ql = (q or "").lower()
    for k, v in _NORM.items():
        if k in ql:
            return v
    return ql.strip()[:40] or "music"


def _load():
    try:
        return json.load(open(PREFS_FILE))
    except Exception:
        return {"events": []}


def _save(d):
    json.dump(d, open(PREFS_FILE, "w"), ensure_ascii=False)


def log(category, item, signal="play", ts=None, dp=None):
    """Record one behavior signal. ts = datetime (default now) for recency/daypart."""
    when = ts or _now()
    if category == "music":
        item = _norm_music(item)
    d = _load()
    d["events"].append({
        "cat": category, "item": item, "signal": signal,
        "w": SIGNAL_W.get(signal, 1.0),
        "dp": dp or daypart(when if isinstance(when, _dt.datetime) else None),
        "ts": (when.isoformat() if isinstance(when, _dt.datetime) else str(when)),
    })
    _save(d)
    return item


def log_play(query, **kw):                 # back-compat (agent.py)
    return log("music", query, "play", **kw)


def _decay(ts_iso):
    try:
        age = (_now() - _dt.datetime.fromisoformat(ts_iso)).total_seconds() / 86400.0
    except Exception:
        age = 0.0
    return 0.5 ** (max(age, 0.0) / HALF_LIFE_DAYS)


def _scores(category, dp=None):
    sc = {}
    for e in _load().get("events", []):
        if e.get("cat") != category:
            continue
        if dp and e.get("dp") != dp:
            continue
        sc[e["item"]] = sc.get(e["item"], 0.0) + e.get("w", 1.0) * _decay(e.get("ts", ""))
    return sc


def top(category="music", dp=None, n=1):
    """Best item(s) by recency-weighted signed score. dp filters by daypart
    (graceful fallback to overall if that context has nothing positive)."""
    sc = {k: v for k, v in _scores(category, dp).items() if v > 0}
    if not sc:
        return top(category, None, n) if dp else None
    ranked = sorted(sc.items(), key=lambda kv: -kv[1])
    return ranked[0][0] if n == 1 else [k for k, _ in ranked[:n]]


def top_music(n=1):                         # back-compat — now TIME-AWARE
    return top("music", daypart(), n)


def summary(category="music"):
    sc = _scores(category)
    if not sc:
        return "(пусто)"
    return ", ".join(f"{k}:{v:+.1f}" for k, v in sorted(sc.items(), key=lambda kv: -kv[1]))


def insights(min_score=1.5):
    """Dominant item per (category, daypart) → human-readable patterns for
    proactive suggestions. Only positive, clear signals."""
    by = {}
    for e in _load().get("events", []):
        w = e.get("w", 1.0) * _decay(e.get("ts", ""))
        if w <= 0:
            continue
        key = (e.get("cat"), e.get("dp"))
        by.setdefault(key, {})
        by[key][e["item"]] = by[key].get(e["item"], 0.0) + w
    out = []
    for (cat, dp), items in by.items():
        item, score = max(items.items(), key=lambda kv: kv[1])
        if score >= min_score:
            out.append({"cat": cat, "dp": dp, "item": item, "score": round(score, 1),
                        "text": f"{_DAYPART_RU.get(dp, dp)} ты обычно: {cat} → {item}"})
    return sorted(out, key=lambda x: -x["score"])
