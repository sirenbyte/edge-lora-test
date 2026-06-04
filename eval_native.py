"""Correct-serving eval for Qwen3.5: dial via enable_thinking=False, tools= for
tool-call. This is the FAIR eval (v1 looked broken only because the old eval
didn't use the native flags). Usage: python eval_native.py <model> [adapter|-] [label]
"""
import sys
from mlx_lm import generate, load

mp = sys.argv[1]
adapter = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
label = sys.argv[3] if len(sys.argv) > 3 else mp.split("/")[-1]
m, t = load(mp, adapter_path=adapter)


def gen(content, mx=200, **kw):
    p = t.apply_chat_template([{"role": "user", "content": content}],
                              add_generation_prompt=True, **kw)
    return generate(m, t, prompt=p, max_tokens=mx, verbose=False).strip()


print(f"### {label} adapter={adapter} (enable_thinking=False) ###")

RU = [("Столица Японии?", ["токио"]), ("Сколько будет 17 + 25?", ["42"]),
      ("Кто написал «Война и мир»?", ["толст"]), ("Назови химический символ золота.", ["au"])]
KZ = [("Жапонияның астанасы қай қала?", ["токио"]), ("Екі қосу үш нешеге тең?", ["5", "бес"]),
      ("«Соғыс және бейбітшілік» романын кім жазды?", ["толст"])]


def run(title, prompts):
    c = 0
    print(f"\n== {title} ==")
    for q, nd in prompts:
        a = gen(q, enable_thinking=False)
        ok = any(x.lower() in a.lower() for x in nd)
        c += ok
        print(f"[{'OK' if ok else '..'}] {q} -> {a[:90]}")
    print(f"  >>> {c}/{len(prompts)}")


print("\n== IDENTITY (enable_thinking=False) ==")
for q in ["Кто ты? Ответь кратко.", "Ты данные в облако шлёшь?", "Что ты умеешь?"]:
    print(f"  {q}\n   -> {gen(q, enable_thinking=False)[:160]}")

run("RU", RU)
run("KAZAKH", KZ)

print("\n== TOOL-CALL (tools= + enable_thinking=False) ==")
tools = [{"type": "function", "function": {
    "name": "turn_off_light", "description": "Выключить свет в комнате",
    "parameters": {"type": "object", "properties": {"room": {"type": "string"}}, "required": ["room"]}}}]
p = t.apply_chat_template([{"role": "user", "content": "Выключи свет в спальне."}],
                          add_generation_prompt=True, tools=tools, enable_thinking=False)
out = generate(m, t, prompt=p, max_tokens=150, verbose=False).strip()
print("  has<tool_call>:", "<tool_call>" in out, "|", out[:220])
