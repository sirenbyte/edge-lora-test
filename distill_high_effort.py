"""Self-distill reasoning-effort pairs from the base LFM2.5.

The base model already emits <think>...</think>. We capture that on a set
of reasoning prompts -> HIGH-effort examples. Stripping the think ->
LOW-effort examples (same Q, direct answer). This teaches the model to
gate its reasoning on a system-prompt effort flag (gpt-oss harmony idea).
"""

import json
import re
from pathlib import Path

from mlx_lm import load, generate

BASE = "LFM2.5-8B-A1B-mlx-4bit"
OUT = Path(__file__).parent / "data7_distill.jsonl"

SYS_HIGH = ("Ты — Qiyas Edge, локальный ИИ-агент. Режим рассуждения: high. "
            "Думай пошагово в <think>...</think>, затем дай ответ.")
SYS_LOW = ("Ты — Qiyas Edge, локальный ИИ-агент. Режим рассуждения: low. "
           "Отвечай сразу и кратко, без рассуждений.")

QUESTIONS = [
    "Если в корзине 3 яблока и ты добавил ещё 7, а потом отдал половину, сколько осталось?",
    "Объясни простыми словами, почему небо голубое.",
    "Составь план подготовки к экзамену за неделю.",
    "Поезд едет 80 км/ч, сколько проедет за 3,5 часа?",
    "Сравни плюсы и минусы электромобиля и бензинового.",
    "Придумай 3 идеи подарка другу-программисту.",
    "Объясни простыми словами, что такое инфляция.",
    "Какой алгоритм выбрать для сортировки списка и почему?",
    "Сколько будет 15% от 240?",
    "Объясни разницу между вирусом и бактерией.",
    "Почему лёд плавает в воде?",
    "У Маши вдвое больше конфет, чем у Пети, вместе 18, сколько у каждого?",
    "Как работает кэш в процессоре, кратко?",
    "Составь маршрут на выходные в горы на 2 дня.",
    "Чем полезна медитация?",
    "Как накопить на цель за 6 месяцев?",
    "Объясни, что такое рекурсия, на примере.",
    "Реши: 24 разделить на 6, плюс 5, умножить на 2.",
    "Почему важно высыпаться?",
    "Как организовать рабочий день для продуктивности?",
]


def rec(sys, q, a):
    return {"messages": [
        {"role": "system", "content": sys},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


def main():
    model, tok = load(BASE)
    n_high = n_low = 0
    with OUT.open("w") as f:
        for q in QUESTIONS:
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": SYS_HIGH},
                 {"role": "user", "content": q}],
                add_generation_prompt=True)
            full = generate(model, tok, prompt=prompt,
                            max_tokens=320, verbose=False).strip()
            final = re.sub(r"<think>.*?</think>", "", full, flags=re.S).strip()
            if not final:                         # all think, no answer — skip
                continue
            # HIGH: keep think+final as the base produced it
            if "<think>" in full:
                f.write(json.dumps(rec(SYS_HIGH, q, full), ensure_ascii=False) + "\n")
                n_high += 1
            # LOW: same question, direct answer only
            f.write(json.dumps(rec(SYS_LOW, q, final), ensure_ascii=False) + "\n")
            n_low += 1
    print(f"distilled: high={n_high} low={n_low} -> {OUT}")


if __name__ == "__main__":
    main()
