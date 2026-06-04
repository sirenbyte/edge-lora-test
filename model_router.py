"""Model-based intent router — let the 4B classify the turn instead of brittle
keyword lists. One cheap greedy pass → an enum label; history-aware so follow-ups
(e.g. bare "страх" after "мне грустно") classify correctly.

Labels map 1:1 to the existing route() decisions, so this is a drop-in alternative.
Pairs with compare_routers.py (accuracy + latency vs the keyword router).
"""
from __future__ import annotations

import json
import re

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

LABELS = ("companion", "fact", "question", "command", "search", "math", "time", "creative")

_PROMPT = (
    "Ты — классификатор намерений голосового ассистента. Определи ТИП последнего "
    "сообщения. Ответь РОВНО одним словом из списка:\n"
    "companion — эмоции/чувства/настроение/светская беседа/поддержка "
    "(грусть, страх, злость, тревога, одиночество, привет, как дела)\n"
    "fact — пользователь СООБЩАЕТ (утверждает) факт О СЕБЕ, НЕ спрашивает "
    "(меня зовут Абзал, я вешу 85, я живу в…, мне 30, у меня есть кот)\n"
    "question — ВОПРОС на знание/память/объяснение, в т.ч. о себе "
    "(кто я, как меня зовут, что ты обо мне знаешь, сколько я вешу, кто, что, где, почему)\n"
    "command — управление устройством/напоминанием/заметкой/музыкой "
    "(включи, выключи, погаси, поставь, разбуди, напомни, запиши, сыграй)\n"
    "search — нужна свежая информация из интернета (новости, погода, что нового, курс, цена)\n"
    "math — вычисление (посчитай, сколько будет, проценты, арифметика)\n"
    "time — текущие дата/время (какое число, который час, какой день)\n"
    "creative — придумать/сочинить (слоган, стих, идея, история, шутка)\n"
    "Только одно слово из списка, без пояснений.\n\n"
    "Примеры:\n"
    "мне страшно -> companion\n"
    "меня всё бесит -> companion\n"
    "я вешу 85 кг -> fact\n"
    "кто я -> question\n"
    "как меня зовут -> question\n"
    "что ты обо мне знаешь -> question\n"
    "сколько я вешу? -> question\n"
    "погаси свет в спальне -> command\n"
    "разбуди меня в 7 -> command\n"
    "что нового в python 3.13 -> search\n"
    "какая погода завтра -> search\n"
    "сколько будет 15% от 240 -> math\n"
    "который час -> time\n"
    "сочини стих про осень -> creative\n"
)


def classify(model, tok, q: str, history=None) -> str:
    """Return one intent label from LABELS (greedy, deterministic)."""
    ctx = ""
    if history:
        prev = [m["content"] for m in history if m.get("role") == "user"][-2:]
        if prev:
            ctx = "Предыдущие реплики пользователя: " + " | ".join(prev) + "\n"
    msgs = [{"role": "system", "content": _PROMPT},
            {"role": "user", "content": f"{ctx}Сообщение: {q}\nТип (одно слово):"}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    out = generate(model, tok, prompt=prompt, max_tokens=6, verbose=False,
                   sampler=make_sampler(temp=0.0)).strip().lower()
    for lab in LABELS:                       # first label that appears wins
        if lab in out:
            return lab
    return "question"                        # safe default


_EXTRACT = (
    "Извлеки из сообщения ОДИН факт о пользователе. Верни СТРОГО JSON: "
    '{"predicate":"<поле латиницей>","value":"<значение>"}\n'
    "Поля: name, age, weight, height, city, country, job, hobby, pet, email, phone, goal.\n"
    "Примеры:\n"
    'Меня зовут Абзал -> {"predicate":"name","value":"Абзал"}\n'
    'я вешу 85 кг -> {"predicate":"weight","value":"85 кг"}\n'
    'я живу в Алматы -> {"predicate":"city","value":"Алматы"}\n'
    'мне 30 лет -> {"predicate":"age","value":"30 лет"}\n'
    'я увлекаюсь пейзажной фотографией -> {"predicate":"hobby","value":"пейзажная фотография"}\n'
    'у меня есть кот -> {"predicate":"pet","value":"кот"}\n'
    "Только JSON, без пояснений."
)


def which_fact(model, tok, query: str, predicates) -> str | None:
    """Which STORED field is this question about? Returns a predicate from the given
    list, or None (general question / field not stored). Lets us answer the saved
    fact VERBATIM (correct grammar) instead of letting the model re-conjugate it."""
    if not predicates:
        return None
    plist = ", ".join(predicates)
    msgs = [{"role": "system", "content":
             f"О каком из этих ИЗВЕСТНЫХ полей пользователь спрашивает? Поля: {plist}. "
             "Ответь РОВНО одним словом из списка. Если вопрос не об этих полях, "
             "не о пользователе, или это не вопрос — ответь 'none'."},
            {"role": "user", "content": query}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    out = generate(model, tok, prompt=prompt, max_tokens=6, verbose=False,
                   sampler=make_sampler(temp=0.0)).strip().lower()
    for p in predicates:                     # exact stored field mentioned in output
        if p.lower() in out:
            return p
    return None


def extract_fact(model, tok, statement: str):
    """Extract (predicate, value) from a first-person personal statement.
    Returns (None, None) if nothing clean could be parsed."""
    msgs = [{"role": "system", "content": _EXTRACT}, {"role": "user", "content": statement}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    out = generate(model, tok, prompt=prompt, max_tokens=48, verbose=False,
                   sampler=make_sampler(temp=0.0))
    try:
        d = json.loads(re.search(r"\{[^{}]*\}", out, re.S).group())
        p = str(d.get("predicate", "")).strip().lower()
        v = str(d.get("value", "")).strip()
        return (p, v) if p and v else (None, None)
    except Exception:
        return (None, None)
