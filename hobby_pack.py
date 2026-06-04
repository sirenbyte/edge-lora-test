"""Living hobby packs — web-fed, grounded content per hobby.

Flow (BRAIN_PLAN §0 personalization = MEMORY + RAG packs, NO per-user training):
  memory(hobby) -> web_search (ddgs) -> fastText curate -> RAG pack (JSON/hobby)
  -> creative mode generates a digest GROUNDED in the fresh materials.

The pack is kept SEPARATE from the user's personal Memory Plant store (hobby
CONTENT vs user FACTS). Curation is model-free (testable standalone); only the
digest step needs the Agent. Online = refresh; offline = use cached pack.

Usage:
  python hobby_pack.py refresh "пейзажная фотография"   # search+curate+save
  python hobby_pack.py digest  "пейзажная фотография"   # grounded digest
  python hobby_pack.py         "пейзажная фотография"   # refresh + digest
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import subprocess
import sys
from pathlib import Path

from quality_filter import assess, clean_text, filter_results

PACKS_DIR = Path(__file__).parent / "packs"
MP_PY = "python3"
MP_BRIDGE = "/Users/abzaltuganbay/projects/edge-lora-test/mp_bridge.py"
SEMRANK = "/Users/abzaltuganbay/projects/edge-lora-test/semrank.py"
MAX_DOCS = 12          # cap stored per pack
DIGEST_DOCS = 6        # how many chunks ground a digest
REFRESH_MAX_AGE_H = 24  # a pack older than this is stale (scheduler)
REGION = "ru-ru"
# generic "looks like actionable advice" signals (domain-agnostic, ru)
_TIP_SIGNALS = ("используйте", "выбирайте", "настройте", "держите", "установите",
                "добавьте", "начните", "попробуйте", "избегайте", "сделайте",
                "проверьте", "поставьте", "возьмите", "не забудьте", "рекомендуется",
                "нужно", "стоит", "лучше", "совет", "шаг", "способ", "приём")


def _slug(hobby: str) -> str:
    return re.sub(r"[^\w]+", "_", hobby.strip().lower(), flags=re.U).strip("_")


def _queries(hobby: str, prefs: str | None = None) -> list[str]:
    h = hobby.strip()
    qs = [f"{h} советы", f"{h} новинки 2026", f"{h} для начинающих"]
    if prefs:
        qs.insert(0, f"{h} {prefs.strip()}")
    return qs[:4]


def search_raw(query: str, max_results: int = 6, region: str = REGION) -> list[dict]:
    try:
        from ddgs import DDGS
    except Exception:
        try:
            from duckduckgo_search import DDGS
        except Exception:
            return []
    try:
        return list(DDGS().text(query, region=region, max_results=max_results))
    except Exception:
        return []


def fetch_extract(url: str, *, max_chars: int = 4000, timeout: int = 8) -> str:
    """Download URL → main article text (boilerplate stripped). '' on any failure.
    trafilatura first (clean extraction); crude tag-strip as offline-safe fallback."""
    if not url:
        return ""
    try:
        import trafilatura
        dl = trafilatura.fetch_url(url)
        if dl:
            txt = trafilatura.extract(dl, include_comments=False,
                                      include_tables=False, favor_precision=True)
            if txt:
                return clean_text(txt)[:max_chars]
    except Exception:
        pass
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "ignore")
        html = re.sub(r"<(script|style|nav|header|footer).*?</\1>", " ", html, flags=re.S | re.I)
        return clean_text(re.sub(r"<[^>]+>", " ", html))[:max_chars]
    except Exception:
        return ""


def refresh_pack(hobby: str, *, prefs: str | None = None, max_docs: int = MAX_DOCS,
                 region: str = REGION, fetch: bool = True, fetch_n: int = 8) -> dict:
    """Search several queries, curate via fastText filter, optionally fetch full
    article text (richer grounding than 160-char snippets), write packs/<slug>.json."""
    raw: list[dict] = []
    for q in _queries(hobby, prefs):
        raw += search_raw(q, region=region)
    curated = filter_results(raw, min_chars=40)[:max_docs]
    if fetch:
        enriched = []
        for i, d in enumerate(curated):
            if i < fetch_n:
                full = fetch_extract(d.get("href", ""))
                if full and len(full) >= 200 and assess(full, min_chars=200).ok:
                    d = {**d, "full": full, "full_chars": len(full)}
            enriched.append(d)
        curated = enriched
    pack = {
        "hobby": hobby,
        "refreshed": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "queries": _queries(hobby, prefs),
        "n_raw": len(raw),
        "n_full": sum(1 for d in curated if d.get("full")),
        "docs": curated,
    }
    PACKS_DIR.mkdir(exist_ok=True)
    (PACKS_DIR / f"{_slug(hobby)}.json").write_text(
        json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    return pack


def load_pack(hobby: str) -> dict | None:
    p = PACKS_DIR / f"{_slug(hobby)}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def chunk_doc(text: str, *, target: int = 380) -> list[str]:
    """Split article text into ~target-char passages on sentence boundaries
    (fetch_extract already collapsed whitespace, so we split on . ! ?)."""
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text) if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 <= target:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def _actionability(t: str) -> float:
    """0..1 'looks like concrete advice' (imperatives + numbers/settings), not a definition."""
    low = t.lower()
    hits = sum(low.count(s) for s in _TIP_SIGNALS)
    digits = len(re.findall(r"\d", low))
    return min((hits + 0.3 * digits) / 4.0, 1.0)


def _semrank(query: str, texts: list[str], k: int):
    """Subprocess to the system-python encoder. Returns [(idx, score)] or None on failure."""
    try:
        r = subprocess.run([MP_PY, SEMRANK], text=True, capture_output=True, timeout=120,
                           input=json.dumps({"query": query, "texts": texts, "k": k}))
        return [(d["i"], d["score"]) for d in json.loads(r.stdout.strip() or "[]")]
    except Exception:
        return None


def grounded_chunks(pack: dict, query: str, k: int = DIGEST_DOCS) -> list[dict]:
    """Top-k tip-bearing passages across the pack, each tagged with its real source.
    Skips intros positionally, ranks the rest by semantic relevance + actionability,
    enforces source diversity. Solves 'skip intro / fetch the advice passage' + accurate
    citations. Graceful fallback (first body passages) when the encoder is unavailable."""
    cand: list[str] = []
    meta: list[tuple[int, str]] = []          # (source doc number, title)
    for n, d in enumerate(pack.get("docs", []), 1):
        chunks = chunk_doc(d.get("full") or d.get("body") or d.get("snippet") or "")
        body = chunks[1:] if len(chunks) > 2 else chunks   # drop the intro when we can
        for c in body:
            cand.append(c)
            meta.append((n, d.get("title", "")))
    if not cand:
        return []
    ranked = _semrank(query, cand, k=min(len(cand), max(4 * k, 12)))
    if ranked is None:                                     # offline / no encoder
        ranked = [(i, 0.0) for i in range(min(len(cand), 4 * k))]
    scored = sorted(((i, s + 0.25 * _actionability(cand[i])) for i, s in ranked),
                    key=lambda x: -x[1])
    picked, per_src = [], {}
    for i, sc in scored:
        src = meta[i][0]
        if per_src.get(src, 0) >= 2:                       # ≤2 passages per source
            continue
        per_src[src] = per_src.get(src, 0) + 1
        picked.append({"text": cand[i], "src": src, "title": meta[i][1], "score": round(sc, 3)})
        if len(picked) >= k:
            break
    return picked


def user_hobby() -> str | None:
    """Best-effort: read the user's hobby from Memory Plant (optional)."""
    try:
        r = subprocess.run([MP_PY, MP_BRIDGE, "search", "хобби увлечение интересы"],
                           capture_output=True, text=True, timeout=90)
        first = r.stdout.strip().splitlines()
        return first[0] if first else None
    except Exception:
        return None


def _digest_msgs(hobby: str, ctx: str) -> list[dict]:
    system = (
        f"Ты — Qiyas Edge. Составь дайджест по теме «{hobby}» СТРОГО по материалам ниже. "
        "Только факты из материалов, ничего не выдумывай и не добавляй от себя. "
        "Выбирай КОНКРЕТНЫЕ практические советы: техника съёмки, свет, композиция, "
        "оборудование, настройки, обработка. НЕ давай общих определений жанра и общих фраз. "
        "Формат: РОВНО 3–5 пунктов, каждый с новой строки и начинается с «• ». "
        "В конце каждого пункта укажи источник в скобках, напр. (1) или (2,3). "
        "Без вступлений и заключений — только пункты.")
    user = f"Материалы:\n{ctx}\n\nДайджест по «{hobby}»:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_digest(hobby: str, *, agent=None, mode: str = "analytical",
                    n: int = DIGEST_DOCS, temp: float = 0.0,
                    refresh: bool = False) -> str:
    pack = refresh_pack(hobby) if refresh else (load_pack(hobby) or refresh_pack(hobby))
    if not pack.get("docs"):
        return "(нет материалов для дайджеста — поиск пуст или офлайн)"
    query = (f"{hobby}: конкретные практические советы, приёмы, техника, настройки, "
             "частые ошибки, рекомендации")
    chunks = grounded_chunks(pack, query, k=n)
    if not chunks:
        return "(нет материалов для дайджеста)"
    ctx = "\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, 1))
    msgs = _digest_msgs(hobby, ctx)
    if agent is None:
        from agent import Agent
        agent = Agent()
    agent.set_mode(mode)
    out = re.sub(r"<think>.*?</think>", "", agent._gen(msgs, think=False, temp=temp),
                 flags=re.S).strip()
    sources = "\n".join(f"[{i}] {c['title']}" for i, c in enumerate(chunks, 1))
    return f"{out}\n\nИсточники:\n{sources}"


def is_stale(pack: dict, max_age_hours: int = REFRESH_MAX_AGE_H) -> bool:
    try:
        ts = _dt.datetime.strptime(pack["refreshed"], "%Y-%m-%d %H:%M")
    except Exception:
        return True
    return (_dt.datetime.now() - ts) > _dt.timedelta(hours=max_age_hours)


def refresh_if_stale(hobby: str, max_age_hours: int = REFRESH_MAX_AGE_H, **kw) -> dict:
    """Refresh only when missing or stale (online); else reuse the cached pack (offline-safe)."""
    pack = load_pack(hobby)
    return refresh_pack(hobby, **kw) if (pack is None or is_stale(pack, max_age_hours)) else pack


def tick(max_age_hours: int = REFRESH_MAX_AGE_H) -> list[str]:
    """Prototype scheduler: refresh every stale pack. Wire to cron / Android WorkManager
    / iOS BGTaskScheduler in production; here it's an idempotent on-demand sweep."""
    done = []
    for f in sorted(PACKS_DIR.glob("*.json")):
        try:
            pack = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if is_stale(pack, max_age_hours):
            refresh_pack(pack["hobby"])
            done.append(pack["hobby"])
    return done


def _main():
    args = sys.argv[1:]
    if args and args[0] == "tick":
        done = tick()
        print(f"scheduler tick: refreshed {len(done)} stale pack(s): {done}")
        return
    cmd = args[0] if args and args[0] in ("refresh", "digest") else None
    hobby = (args[1] if cmd else (args[0] if args else None)) or user_hobby()
    if not hobby:
        print("usage: hobby_pack.py [refresh|digest] \"<хобби>\"")
        return
    if cmd == "refresh":
        pack = refresh_pack(hobby)
        print(f"pack '{hobby}': {pack['n_raw']} raw -> {len(pack['docs'])} curated "
              f"({pack['n_full']} full-text) -> packs/{_slug(hobby)}.json  ({pack['refreshed']})")
        for d in pack["docs"][:6]:
            print(f"  [{d['_lang']} {d['_score']:.2f}] {d.get('title','')[:64]}")
    else:  # digest or end-to-end
        if cmd is None:
            pack = refresh_pack(hobby)
            print(f"refreshed '{hobby}': {len(pack['docs'])} curated docs")
        print(f"\n=== ДАЙДЖЕСТ: {hobby} ===")
        print(generate_digest(hobby, refresh=(cmd is None)))


if __name__ == "__main__":
    _main()
