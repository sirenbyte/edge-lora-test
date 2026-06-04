"""Lexical verifier — a small 'proofreader' pass that repairs the FORM of an answer
(person agreement, common verb slips) WITHOUT changing meaning. Two flavors:

  * lexical_fix(text)            — deterministic, SURGICAL (only known fact-phrase
                                   slips). Instant, zero model calls, zero risk to
                                   legit assistant 1st-person ("я понимаю…").
  * model_verify(model, tok, t)  — the 4B re-reads and corrects with context (catches
                                   novel slips, but +1 pass and can over-correct).

Facts are already returned verbatim (correct), so this guards GENERATED answers.
"""
from __future__ import annotations

import re

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

# Surgical: ONLY user-fact phrasings that slipped into 1st person. We deliberately
# do NOT touch a bare "я" (so legit assistant speech like "я понимаю" is safe).
_FIXES = [
    (r"\bменя зовут\b", "тебя зовут"),
    (r"\bя живу\b", "ты живёшь"),
    (r"\bя вешу\b", "ты весишь"),
    (r"\bвешаешь\b", "весишь"),
    (r"\bя работаю\b", "ты работаешь"),
    (r"\bмне (\d+)\s+(лет|года|год)\b", r"тебе \1 \2"),
]


def lexical_fix(text: str) -> str:
    out = text
    for pat, rep in _FIXES:
        # preserve leading capitalization of the matched span
        def _sub(m):
            r = rep
            if m.group(0)[:1].isupper():
                r = r[:1].upper() + r[1:]
            return r
        out = re.sub(pat, _sub, out, flags=re.I)
    return out


_VERIFY = (
    "Ты — корректор. Исправь в ответе ассистента ТОЛЬКО грамматику, орфографию и "
    "согласование, НЕ меняя смысл и факты. О пользователе говори на «ты» (тебя зовут, "
    "ты живёшь, ты весишь). Но реплики ассистента О СЕБЕ («я понимаю», «я ставлю», "
    "«я запомнил») НЕ трогай. Верни ТОЛЬКО исправленный текст, без пояснений."
)


def model_verify(model, tok, answer: str) -> str:
    msgs = [{"role": "system", "content": _VERIFY}, {"role": "user", "content": answer}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    out = generate(model, tok, prompt=prompt, max_tokens=96, verbose=False,
                   sampler=make_sampler(temp=0.0)).strip().strip('"')
    return out or answer
