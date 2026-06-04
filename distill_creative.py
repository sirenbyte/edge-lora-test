"""Self-distill CREATIVE exemplars from the smarter Qwen3.5-9B (teacher) to train
the 4B creative ("right hemisphere") adapter. Moderate temp (0.7) on the 9B gives
divergent BUT coherent Russian — better targets than the 4B can self-produce.
"""
import json
import re
from pathlib import Path
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

TEACHER = "mlx-community/Qwen3.5-9B-MLX-4bit"
OUT = Path(__file__).parent / "data_creative_distill.jsonl"

SYS_CREATIVE = ("Ты — Qiyas Edge в творческом режиме. Будь изобретательным и "
                "нестандартным: давай разнообразные, яркие, образные и оригинальные "
                "идеи. Пиши живо и по-русски грамотно.")

PROMPTS = [
    "Придумай 5 необычных идей подарка другу-программисту.",
    "Напиши короткое четверостишие про осенний дождь.",
    "Опиши вкус кофе тому, кто никогда его не пробовал — ярко и образно.",
    "Придумай 3 нестандартных названия для уютной кофейни.",
    "Что если бы у деревьев был Wi-Fi? Пофантазируй коротко и забавно.",
    "Сравни сон с чем-нибудь неожиданным через метафору.",
    "Придумай 4 идеи для свидания без больших трат.",
    "Опиши город будущего одним ярким абзацем.",
    "Сочини двустишие про утренний кофе.",
    "Придумай название и слоган для магазина растений.",
    "Опиши звук дождя для человека, который не слышит — через образы.",
    "Придумай 3 идеи мини-игры для двоих в дороге.",
    "Что общего у книги и реки? Найди красивую аналогию.",
    "Сочини короткую сказку (3-4 предложения) про ленивого робота.",
    "Придумай 5 креативных способов запомнить список покупок.",
    "Опиши осенний лес через запахи и звуки.",
    "Придумай необычный тост на день рождения друга.",
    "Если бы у тебя был свой остров, как бы ты его назвал и почему?",
    "Придумай метафору для слова «вдохновение».",
    "Сочини рекламный слоган для чая, который поднимает настроение.",
    "Опиши, каково это — лететь на воздушном шаре, ярко.",
    "Придумай 3 идеи подарка маме своими руками.",
    "Сочини загадку про звёзды.",
    "Придумай имя и характер дружелюбному дракону из сказки.",
    "Опиши зиму так, будто впервые её видишь.",
    "Придумай девиз для команды мечтателей.",
    "Сравни идею с семенем — развей мысль красиво.",
    "Придумай 4 названия для плейлиста, под который хорошо думается.",
]


def rec(q, a):
    return {"messages": [
        {"role": "system", "content": SYS_CREATIVE},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


def main():
    m, t = load(TEACHER)
    sampler = make_sampler(temp=0.7, top_p=0.95)
    n = 0
    with OUT.open("w") as f:
        for q in PROMPTS:
            p = t.apply_chat_template(
                [{"role": "system", "content": SYS_CREATIVE},
                 {"role": "user", "content": q}],
                add_generation_prompt=True, enable_thinking=False)
            a = generate(m, t, prompt=p, max_tokens=300, verbose=False,
                         sampler=sampler).strip()
            a = re.sub(r"<think>.*?</think>", "", a, flags=re.S).strip()
            if len(a) < 15:                 # skip empties
                continue
            f.write(json.dumps(rec(q, a), ensure_ascii=False) + "\n")
            n += 1
    print(f"creative distilled from {TEACHER}: {n} -> {OUT}")


if __name__ == "__main__":
    main()
