"""Probe Qwen3.5-4B NATIVE dial (enable_thinking) + native tool-calling (tools=).
If these work on the BARE model, we don't need to train them — only identity/domain.
"""
import sys
from mlx_lm import generate, load

model_path = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3.5-4B-MLX-4bit"
adapter = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
m, t = load(model_path, adapter_path=adapter)


def gen(msgs, mx=220, **kw):
    try:
        p = t.apply_chat_template(msgs, add_generation_prompt=True, **kw)
    except Exception as e:
        return f"[apply_chat_template error with {list(kw)}: {e}]"
    return generate(m, t, prompt=p, max_tokens=mx, verbose=False).strip()


print(f"### model={model_path} adapter={adapter} ###")

print("\n=== A. enable_thinking=False (expect: clean short answer, NO <think>) ===")
out = gen([{"role": "user", "content": "Столица Японии? Ответь кратко."}], enable_thinking=False)
print("has<think>:", "<think>" in out, "|", out[:200])

print("\n=== B. enable_thinking=True (expect: <think> present) ===")
out = gen([{"role": "user", "content": "Сколько будет 15% от 240?"}], enable_thinking=True, mx=300)
print("has<think>:", "<think>" in out, "|", out[:160])

print("\n=== C. native tools= (expect: <tool_call> with turn_on_light) ===")
tools = [{"type": "function", "function": {
    "name": "turn_on_light",
    "description": "Включить свет в комнате",
    "parameters": {"type": "object",
                   "properties": {"room": {"type": "string"}},
                   "required": ["room"]}}}]
out = gen([{"role": "user", "content": "Включи свет на кухне."}],
          tools=tools, enable_thinking=False, mx=200)
print("has<tool_call>:", "<tool_call>" in out, "|", out[:300])
