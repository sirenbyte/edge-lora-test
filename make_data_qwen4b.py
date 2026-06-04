"""Qwen3.5-4B `core` adapter data: dial + identity + NATIVE tool-call + ru-fluency.

Reuses the v7 recipe (identity over-rep, alpaca fluency, general retention) and
the dial pairs re-distilled FROM Qwen3.5-4B (data_qwen4b_distill.jsonl).
NEW: native Qwen tool-calling examples `<tool_call>{json}</tool_call>` (tokens
248058/248059) — device-control, with negatives so it doesn't over-call.
"""

import json
import random
from pathlib import Path

from make_data_v7 import IDENTITY, load_alpaca
from make_data_v2 import GENERAL
from distill_high_effort import SYS_LOW

random.seed(23)
DATA = Path(__file__).parent / "data_qwen4b"
N_ALPACA = 400

TOOL_SYS = (
    "Ты — Qiyas Edge, локальный ИИ-агент. Доступные инструменты:\n"
    "- turn_on_light(room), turn_off_light(room)\n"
    "- set_alarm(time), set_timer(minutes)\n"
    "- play_music(query), set_volume(level)\n"
    "- set_temperature(celsius), create_reminder(text, time)\n"
    "Если запрос требует действия — ответь ТОЛЬКО вызовом в формате "
    "<tool_call>\n{\"name\": ..., \"arguments\": {...}}\n</tool_call>. "
    "Если действие не нужно — ответь обычным текстом.")


def tc(name, args):
    body = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
    return f"<tool_call>\n{body}\n</tool_call>"


def rec(sys, q, a):
    return {"messages": [
        {"role": "system", "content": sys},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


ROOM = {"кухне": "кухня", "спальне": "спальня", "гостиной": "гостиная",
        "ванной": "ванная", "коридоре": "коридор"}


def build_tools():
    ex = []
    for loc, nom in ROOM.items():
        ex.append(rec(TOOL_SYS, f"Включи свет в {loc}.", tc("turn_on_light", {"room": nom})))
        ex.append(rec(TOOL_SYS, f"Выключи свет в {loc}.", tc("turn_off_light", {"room": nom})))
    for t in ["7:00", "6:30", "8:15", "9:00"]:
        ex.append(rec(TOOL_SYS, f"Поставь будильник на {t}.", tc("set_alarm", {"time": t})))
    for m in [5, 10, 15, 30]:
        ex.append(rec(TOOL_SYS, f"Засеки таймер на {m} минут.", tc("set_timer", {"minutes": m})))
    for q in ["джаз", "любимый плейлист", "что-нибудь спокойное", "классику"]:
        ex.append(rec(TOOL_SYS, f"Включи {q}.", tc("play_music", {"query": q})))
    for lv in [30, 50, 70]:
        ex.append(rec(TOOL_SYS, f"Поставь громкость на {lv}.", tc("set_volume", {"level": lv})))
    for d in [20, 22, 24]:
        ex.append(rec(TOOL_SYS, f"Сделай {d} градуса.", tc("set_temperature", {"celsius": d})))
    ex.append(rec(TOOL_SYS, "Напомни позвонить маме в 18:00.",
                  tc("create_reminder", {"text": "позвонить маме", "time": "18:00"})))
    ex.append(rec(TOOL_SYS, "Напомни купить хлеб вечером.",
                  tc("create_reminder", {"text": "купить хлеб", "time": "вечером"})))
    # en + kz
    ex.append(rec(TOOL_SYS, "Turn on the light in the kitchen.", tc("turn_on_light", {"room": "кухня"})))
    ex.append(rec(TOOL_SYS, "Ас үйде шамды қос.", tc("turn_on_light", {"room": "кухня"})))
    # negatives: tools present but no action needed -> plain answer
    for q, a in [("Кто написал «Война и мир»?", "«Война и мир» написал Лев Толстой."),
                 ("Столица Японии?", "Токио."),
                 ("Сколько будет 12 * 8?", "96."),
                 ("Что такое фотосинтез, кратко?",
                  "Растения из света, воды и CO₂ производят энергию и кислород.")]:
        ex.append(rec(TOOL_SYS, q, a))
    return ex


def main():
    distilled = [json.loads(l) for l in open(DATA.parent / "data_qwen4b_distill.jsonl")]
    high = [r for r in distilled if "high" in r["messages"][0]["content"]]
    low = [r for r in distilled if "low" in r["messages"][0]["content"]]
    tools = build_tools()

    train = []
    for _ in range(20):                          # identity over-rep
        train += [rec(SYS_LOW, q, a) for q, a in IDENTITY]
    train += high * 3
    train += low * 2
    train += tools * 3                           # tool-call over-rep
    for _ in range(2):
        train += [rec(SYS_LOW, q, a) for q, a in GENERAL]
    train += load_alpaca()[:N_ALPACA]            # ru fluency
    random.shuffle(train)

    valid = ([rec(SYS_LOW, q, a) for q, a in IDENTITY[:3]]
             + tools[:3] + high[:2] + low[:2]
             + [rec(SYS_LOW, q, a) for q, a in GENERAL[:3]])

    DATA.mkdir(exist_ok=True)
    with (DATA / "train.jsonl").open("w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (DATA / "valid.jsonl").open("w") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train={len(train)} (identity={20 * len(IDENTITY)}, high={len(high) * 3}, "
          f"low={len(low) * 2}, tools={len(tools) * 3}, general={2 * len(GENERAL)}, "
          f"alpaca={N_ALPACA}) valid={len(valid)} -> {DATA}")


if __name__ == "__main__":
    main()
