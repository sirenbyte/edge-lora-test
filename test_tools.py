"""Tool-call reliability probe (native tools= format) with GREEDY decoding.
Checks: valid <tool_call>, correct function name, FAITHFUL argument value.
Usage: python test_tools.py <model> [adapter|-] [temp]
"""
import sys
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

mp = sys.argv[1]
adapter = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
m, t = load(mp, adapter_path=adapter)
sampler = make_sampler(temp=temp)

ROOM_ENUM = ["кухня", "спальня", "гостиная", "ванная", "коридор"]
TOOLS = [
    {"type": "function", "function": {"name": "turn_on_light", "description": "Включить свет в комнате",
     "parameters": {"type": "object", "properties": {"room": {"type": "string", "enum": ROOM_ENUM}}, "required": ["room"]}}},
    {"type": "function", "function": {"name": "turn_off_light", "description": "Выключить свет в комнате",
     "parameters": {"type": "object", "properties": {"room": {"type": "string", "enum": ROOM_ENUM}}, "required": ["room"]}}},
    {"type": "function", "function": {"name": "set_alarm", "description": "Поставить будильник",
     "parameters": {"type": "object", "properties": {"time": {"type": "string"}}, "required": ["time"]}}},
    {"type": "function", "function": {"name": "set_volume", "description": "Установить громкость",
     "parameters": {"type": "object", "properties": {"level": {"type": "integer"}}, "required": ["level"]}}},
    {"type": "function", "function": {"name": "set_timer", "description": "Поставить таймер",
     "parameters": {"type": "object", "properties": {"minutes": {"type": "integer"}}, "required": ["minutes"]}}},
]
CASES = [
    ("Включи свет на кухне.", "turn_on_light", "кухн"),
    ("Выключи свет в спальне.", "turn_off_light", "спальн"),
    ("Поставь будильник на 7:30.", "set_alarm", "7:30"),
    ("Сделай громкость 60.", "set_volume", "60"),
    ("Засеки таймер на 15 минут.", "set_timer", "15"),
    ("Выключи свет в гостиной.", "turn_off_light", "гостин"),
    ("Включи свет в ванной.", "turn_on_light", "ванн"),
]


def call(q):
    p = t.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                              tools=TOOLS, enable_thinking=False)
    return generate(m, t, prompt=p, max_tokens=120, verbose=False, sampler=sampler).strip()


print(f"### {mp.split('/')[-1]} adapter={adapter} temp={temp} ###")
tc = fn = arg = 0
for q, want_fn, argneedle in CASES:
    out = call(q)
    has = "<tool_call>" in out
    fn_ok = want_fn in out
    arg_ok = argneedle.lower() in out.lower()
    tc += has
    fn += fn_ok
    arg += arg_ok
    flag = f"{'TC' if has else '--'}|{'F' if fn_ok else '.'}{'A' if arg_ok else '.'}"
    print(f"[{flag}] {q}\n   -> {out[:150]}")
print(f"\n>>> tool_call {tc}/{len(CASES)}, function {fn}/{len(CASES)}, argument {arg}/{len(CASES)} (temp={temp})")
