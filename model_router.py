"""Model-based intent router — let the 4B classify the turn instead of brittle
keyword lists. One cheap greedy pass → an enum label; history-aware so follow-ups
(e.g. bare "страх" after "мне грустно") classify correctly.

Labels map 1:1 to the existing route() decisions, so this is a drop-in alternative.
Pairs with compare_routers.py (accuracy + latency vs the keyword router).
"""
from __future__ import annotations

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

LABELS = ("companion", "fact", "question", "command", "search", "math", "time", "creative")

_PROMPT = (
    "Ты — классификатор намерений голосового ассистента. Определи ТИП последнего "
    "сообщения. Ответь РОВНО одним словом из списка:\n"
    "companion — эмоции/чувства/настроение/светская беседа/поддержка "
    "(грусть, страх, злость, тревога, одиночество, привет, как дела)\n"
    "fact — пользователь сообщает факт О СЕБЕ (меня зовут, я вешу, я живу, мне 30, у меня есть)\n"
    "question — вопрос на знание/память/объяснение (кто, что, где, сколько, почему, как)\n"
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
