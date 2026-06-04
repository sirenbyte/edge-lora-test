"""Creative ("right hemisphere") probe: is high-temp decoding + a creative system
prompt already divergent/vivid enough, or do we need a creative LoRA?
Usage: python test_creative.py <model> [adapter|-] [temp]
"""
import sys
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

mp = sys.argv[1]
adapter = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
top_p = 0.95
m, t = load(mp, adapter_path=adapter)
sampler = make_sampler(temp=temp, top_p=top_p)

SYS = ("Ты — Qiyas Edge в творческом режиме. Будь изобретательным и нестандартным: "
       "давай разнообразные, яркие, оригинальные идеи и образы. Не бойся фантазировать.")
PROMPTS = [
    "Придумай 5 необычных идей подарка другу-программисту.",
    "Напиши короткое четверостишие про осенний дождь.",
    "Опиши вкус кофе тому, кто никогда его не пробовал — ярко и образно.",
    "Придумай 3 нестандартных названия для кофейни.",
    "Что если бы у деревьев был Wi-Fi? Пофантазируй коротко.",
]


def gen(q):
    p = t.apply_chat_template([{"role": "system", "content": SYS},
                               {"role": "user", "content": q}],
                              add_generation_prompt=True, enable_thinking=False)
    return generate(m, t, prompt=p, max_tokens=260, verbose=False, sampler=sampler).strip()


print(f"### {mp.split('/')[-1]} adapter={adapter} temp={temp} top_p={top_p} ###")
for q in PROMPTS:
    print(f"\nQ: {q}\n-> {gen(q)[:420]}")
