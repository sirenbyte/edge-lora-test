"""fastText quality filter for scraped/real text (MiniCPM Ultra-FineWeb recipe).

Two uses:
  1) hobby-pack curation — filter/rank ddgs results before packing into RAG;
  2) real-data training — clean a corpus before LoRA.

Project scope keeps only ru/en, and drops junk/boilerplate/too-short/duplicate/
repetitive text. Uses fastText lid.176 for language ID via the low-level C++
`model.f.predict` (the python `.predict()` wrapper is broken under NumPy 2.x:
`np.array(probs, copy=False)`), so no site-package patch and no numpy pin.
"""
from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

LID_MODEL = Path(__file__).parent / "lid.176.ftz"
KEEP_LANGS = ("ru", "en")
MIN_LANG_PROB = 0.50
MIN_CHARS = 40
MAX_CHARS = 20_000
MAX_SYMBOL_RATIO = 0.35       # share of non-letter/digit/space chars
MAX_DIGIT_RATIO = 0.30
MIN_UNIQUE_WORD_RATIO = 0.35  # below this = repetitive spam
JUNK_MARKERS = (
    "подпишитесь", "войдите в аккаунт", "регистрация", "принять куки", "cookie",
    "subscribe", "sign in", "log in", "404", "page not found", "lorem ipsum",
    "all rights reserved", "ничего не найдено", "no results found",
)
_WS = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _model():
    import fasttext
    with open(os.devnull, "w") as null, contextlib.redirect_stderr(null):
        return fasttext.load_model(str(LID_MODEL))


def clean_text(text: str) -> str:
    return _WS.sub(" ", (text or "")).strip()


def detect_lang(text: str) -> tuple[str, float]:
    """(lang_code, prob) via low-level predict — NumPy-2.x-safe."""
    flat = clean_text(text)
    if not flat:
        return ("", 0.0)
    preds = _model().f.predict(flat, 1, 0.0, "strict")
    if not preds:
        return ("", 0.0)
    prob, label = preds[0]
    return (label.replace("__label__", ""), float(prob))


def _symbol_ratio(s: str) -> float:
    return sum(1 for c in s if not c.isalnum() and not c.isspace()) / len(s)


def _digit_ratio(s: str) -> float:
    return sum(c.isdigit() for c in s) / len(s)


def _unique_word_ratio(s: str) -> float:
    words = s.split()
    return len({w.lower() for w in words}) / len(words) if words else 0.0


@dataclass(frozen=True)
class Verdict:
    ok: bool
    lang: str
    prob: float
    reason: str
    score: float


def assess(text: str, *, keep_langs=KEEP_LANGS, min_chars: int = MIN_CHARS) -> Verdict:
    """Heuristics first (cheap), language last (model call). Returns a Verdict."""
    t = clean_text(text)
    if len(t) < min_chars:
        return Verdict(False, "", 0.0, "too_short", 0.0)
    t = t[:MAX_CHARS]
    low = t.lower()
    if any(m in low for m in JUNK_MARKERS):
        return Verdict(False, "", 0.0, "junk_marker", 0.0)
    if _symbol_ratio(t) > MAX_SYMBOL_RATIO:
        return Verdict(False, "", 0.0, "too_symbolic", 0.0)
    if _digit_ratio(t) > MAX_DIGIT_RATIO:
        return Verdict(False, "", 0.0, "too_numeric", 0.0)
    uniq = _unique_word_ratio(t)
    if uniq < MIN_UNIQUE_WORD_RATIO:
        return Verdict(False, "", 0.0, "repetitive", 0.0)
    lang, prob = detect_lang(t)
    if lang not in keep_langs:
        return Verdict(False, lang, prob, f"lang:{lang or '?'}", 0.0)
    if prob < MIN_LANG_PROB:
        return Verdict(False, lang, prob, "low_lang_conf", 0.0)
    return Verdict(True, lang, prob, "ok", round(prob * uniq, 3))


_FIELDS = ("title", "body", "snippet", "excerpt", "text")


def filter_results(results, *, keep_langs=KEEP_LANGS, min_chars: int = 40, dedup=True):
    """Curate ddgs-style results [{title, body/snippet, href}] for a hobby pack.

    Returns a NEW list (immutable input), filtered + sorted by quality desc,
    each annotated with _lang/_score.
    """
    seen, out = set(), []
    for r in results:
        text = " ".join(str(r.get(k, "")) for k in _FIELDS if r.get(k))
        v = assess(text, keep_langs=keep_langs, min_chars=min_chars)
        if not v.ok:
            continue
        if dedup:
            key = clean_text(text).lower()[:200]
            if key in seen:
                continue
            seen.add(key)
        out.append({**r, "_lang": v.lang, "_score": v.score})
    return sorted(out, key=lambda x: x["_score"], reverse=True)


def filter_corpus(texts, **kw):
    """Clean training strings. Returns (kept_list, {reason: dropped_count})."""
    kept, dropped = [], {}
    for t in texts:
        v = assess(t, **kw)
        if v.ok:
            kept.append(t)
        else:
            dropped[v.reason] = dropped.get(v.reason, 0) + 1
    return kept, dropped


if __name__ == "__main__":
    samples = [
        "Лучшие приёмы пейзажной фотографии: золотой час, штатив и работа с RAW.",
        "Best landscape photography tips for beginners: shoot in golden hour.",
        "这是一段中文文本，用于测试语言识别和过滤功能。",
        "ok",                                              # too short
        "Подпишитесь на рассылку! Принять куки. Войдите в аккаунт.",  # junk
        "!!! $$$ @@@ ### %%% ^^^ &&& *** ((( ))) +++ === ~~~",        # symbolic
        "купить купить купить купить купить купить купить купить",     # repetitive
    ]
    print("== assess() ==")
    for s in samples:
        v = assess(s)
        flag = "KEEP" if v.ok else "drop"
        print(f"  [{flag:4}] {v.reason:14} {v.lang:2} {v.prob:.2f} | {s[:46]}")
    print("\n== filter_results() ==")
    res = [{"title": s[:30], "body": s} for s in samples]
    kept = filter_results(res, min_chars=40)
    for r in kept:
        print(f"  score={r['_score']:.3f} {r['_lang']} | {r['body'][:46]}")
