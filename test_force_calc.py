"""Probe ways to FORCE a calculate tool call on compute queries."""
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

m, t = load("mlx-community/Qwen3.5-4B-MLX-4bit")
G = make_sampler(temp=0.0)
CALC = [{"type": "function", "function": {"name": "calculate",
        "description": "Точно вычислить выражение", "parameters": {"type": "object",
        "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]
QS = ["Сколько будет 15% от 240?", "Посчитай корень из 2 умножить на 100"]
PREFILL = "<tool_call>\n<function=calculate>\n<parameter=expression>\n"


def run(q, prefill=False, **kw):
    p = t.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                              tools=CALC, enable_thinking=False, tokenize=False, **kw)
    if prefill:
        p = p + PREFILL
    return generate(m, t, prompt=p, max_tokens=80, verbose=False, sampler=G)


for q in QS:
    print("Q:", q)
    try:
        o = run(q, tool_choice="required")
        print("  tool_choice=required:", "<tool_call>" in o, "|", o[:100].replace("\n", " "))
    except Exception as e:
        print("  tool_choice ERR:", repr(e)[:100])
    o2 = run(q, prefill=True)
    full = PREFILL + o2
    print("  prefill:", "</tool_call>" in full or "</parameter>" in full, "|", (PREFILL + o2)[:140].replace("\n", " "))
    print()
