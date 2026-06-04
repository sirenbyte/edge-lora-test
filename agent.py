"""Qiyas Edge — on-device agent gateway (Qwen3.5-4B) with tool EXECUTION.

router (rules) -> mode (analytical/creative) + hot-swap LoRA adapter
              -> enable_thinking (auto-dial) + tools=/enum + temp
              -> if tool_call: EXECUTE (calculate/datetime/device) -> final answer
Memory + hobby-packs = STUB hooks (retrieve-before / ingest-after).

Usage: python agent.py            # demo
       python agent.py "запрос"   # single
"""
import ast
import datetime as _dt
import json
import math
import operator
import re
import subprocess
import sys
import urllib.parse
import urllib.request

from mlx_lm import generate, load  # noqa: F401  (load kept for compat)
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.tuner.utils import load_adapters, remove_lora_layers

import prefs
import proactive
from vision_unload import load_text_only

BASE = "mlx-community/Qwen3.5-4B-MLX-4bit"
ADAPTERS = {"analytical": "adapters_qwen4b_v2", "creative": "adapters_qwen4b_creative"}
SYS_CREATIVE = ("Ты — Qiyas Edge в творческом режиме. Будь изобретательным: давай "
                "разнообразные, яркие, образные идеи. Сразу к делу, без вступлений.")
ROOM_ENUM = ["кухня", "спальня", "гостиная", "ванная", "коридор"]

# ---- memory bridge (Python Memory Plant via system python3) ----
MP_PY = "python3"
MP_BRIDGE = "/Users/abzaltuganbay/projects/edge-lora-test/mp_bridge.py"
REMEMBER_KW = ("меня зовут", "моё имя", "мое имя", "я люблю", "я увлекаюсь", "запомни",
               "я работаю", "я живу", "я предпочитаю", "мне нравится", "мой ", "моя ", "мои ")


_Q_WORDS = ("как", "что", "где", "кто", "когда", "почему", "чем", "какой",
            "какая", "сколько", "зачем", "куда")


def should_remember(q):
    low = q.lower().strip()
    if low.endswith("?") or (low.split() and low.split()[0] in _Q_WORDS):
        return False                       # don't store questions, only statements
    return any(k in low for k in REMEMBER_KW)

# ---------- tool schemas ----------
CALC_TOOL = {"type": "function", "function": {
    "name": "calculate", "description": "Точно вычислить математическое выражение.",
    "parameters": {"type": "object", "properties": {
        "expression": {"type": "string", "description": "Python-синтаксис, напр. 0.15*240 или 47*83+12"}},
        "required": ["expression"]}}}
TIME_TOOL = {"type": "function", "function": {
    "name": "get_datetime", "description": "Текущие дата и время.",
    "parameters": {"type": "object", "properties": {}}}}
DEVICE_TOOLS = [
    {"type": "function", "function": {"name": "turn_on_light", "description": "Включить свет",
     "parameters": {"type": "object", "properties": {"room": {"type": "string", "enum": ROOM_ENUM}}, "required": ["room"]}}},
    {"type": "function", "function": {"name": "turn_off_light", "description": "Выключить свет",
     "parameters": {"type": "object", "properties": {"room": {"type": "string", "enum": ROOM_ENUM}}, "required": ["room"]}}},
    {"type": "function", "function": {"name": "set_alarm", "description": "Поставить будильник",
     "parameters": {"type": "object", "properties": {"time": {"type": "string"}}, "required": ["time"]}}},
    {"type": "function", "function": {"name": "set_timer", "description": "Таймер (минуты)",
     "parameters": {"type": "object", "properties": {"minutes": {"type": "integer"}}, "required": ["minutes"]}}},
    {"type": "function", "function": {"name": "set_volume", "description": "Громкость 0-100",
     "parameters": {"type": "object", "properties": {"level": {"type": "integer"}}, "required": ["level"]}}},
]
REMINDER_TOOL = {"type": "function", "function": {"name": "create_reminder", "description": "Создать напоминание",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "time": {"type": "string"}}, "required": ["text", "time"]}}}
NOTE_TOOL = {"type": "function", "function": {"name": "create_note", "description": "Сохранить заметку",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}
MEDIA_TOOL = {"type": "function", "function": {"name": "play_music", "description": "Включить музыку",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
SEARCH_TOOL = {"type": "function", "function": {"name": "web_search", "description": "Поиск свежей информации в интернете",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
ACTION_TOOLS = DEVICE_TOOLS + [REMINDER_TOOL, NOTE_TOOL, MEDIA_TOOL]
NOTES_FILE = "/Users/abzaltuganbay/projects/edge-lora-test/notes.json"
REMINDERS_FILE = "/Users/abzaltuganbay/projects/edge-lora-test/reminders.json"

# ---------- safe calculator ----------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "exp": math.exp, "sin": math.sin,
          "cos": math.cos, "tan": math.tan, "abs": abs, "round": round}
_CONST = {"pi": math.pi, "e": math.e}


def _ev(n):
    if isinstance(n, ast.Constant):
        return n.value
    if isinstance(n, ast.BinOp):
        return _OPS[type(n.op)](_ev(n.left), _ev(n.right))
    if isinstance(n, ast.UnaryOp):
        return _OPS[type(n.op)](_ev(n.operand))
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
        return _FUNCS[n.func.id](*[_ev(a) for a in n.args])
    if isinstance(n, ast.Name) and n.id in _CONST:
        return _CONST[n.id]
    raise ValueError("unsupported expression")


def calculate(expression):
    expr = expression.replace("^", "**").strip()
    val = _ev(ast.parse(expr, mode="eval").body)
    return round(val, 6) if isinstance(val, float) else val


# ---------- tool execution ----------
def _append_json(path, item):
    try:
        data = json.load(open(path))
    except Exception:
        data = []
    data.append(item)
    json.dump(data, open(path, "w"), ensure_ascii=False)


def web_search(query):
    """Real web search via ddgs (DuckDuckGo, free, no key). Offline fallback.
    The ONE online tool; future: route via our backend (SearXNG) for privacy/scale."""
    try:
        from ddgs import DDGS
    except Exception:
        try:
            from duckduckgo_search import DDGS
        except Exception:
            return "(поиск недоступен)"
    try:
        res = DDGS().text(query, max_results=4)
        if not res:
            return "(ничего не найдено)"
        return "\n".join(f"- {r.get('title', '')}: {r.get('body', '')[:160]}" for r in res)
    except Exception:
        return "(поиск недоступен — офлайн)"


def exec_tool(name, p):
    if name == "calculate":
        try:
            return str(calculate(p["expression"]))
        except Exception as e:
            return f"ошибка: {e}"
    if name == "get_datetime":
        return _dt.datetime.now().strftime("%Y-%m-%d %H:%M, %A")
    if name in ("turn_on_light", "turn_off_light"):
        st = "включён" if name.startswith("turn_on") else "выключен"
        return f"свет в '{p.get('room')}' {st}"
    if name == "set_alarm":
        return f"будильник на {p.get('time')} поставлен"
    if name == "set_timer":
        return f"таймер на {p.get('minutes')} мин запущен"
    if name == "set_volume":
        return f"громкость = {p.get('level')}"
    if name == "create_reminder":
        _append_json(REMINDERS_FILE, {"text": p.get("text"), "time": p.get("time")})
        return f"напоминание '{p.get('text')}' на {p.get('time')} создано"
    if name == "create_note":
        _append_json(NOTES_FILE, {"text": p.get("text")})
        return f"заметка сохранена: '{p.get('text')}'"
    if name == "play_music":
        g = prefs.log_play(p.get("query", ""))      # learn taste from behavior
        return f"играю '{p.get('query')}' (вкус +1: {g})"
    if name == "web_search":
        return web_search(p.get("query", ""))
    return f"{name}({p}) выполнено"


def parse_tool_call(text):
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.S)
    if not m:
        return None
    body = m.group(1)
    fn = re.search(r"<function=([^>\s]+)>", body)
    if not fn:
        return None
    params = {pm.group(1).strip(): pm.group(2).strip()
              for pm in re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", body, re.S)}
    return fn.group(1), params


# ---------- router ----------
CREATIVE_KW = ("придумай", "сочини", "напиши стих", "стих", "истори", "сказк", "идеи", "идею",
               "назван", "пофантазир", "метафор", "креатив", "поздравлен", "слоган", "девиз", "загадк")
TOOL_KW = ("включи", "выключи", "поставь будильник", "разбуди", "таймер", "засеки", "громкость",
           "напомни", "заметк", "запиши", "музык", "трек", "пауза", "плейлист")
COMPUTE_KW = ("посчитай", "вычисли", "сколько будет", "процент", "%", "корень", "умнож", "раздели")
TIME_KW = ("который час", "сколько времени", "какое сегодня", "какое число", "какой день", "дата")
SEARCH_KW = ("найди", "поищи", "погугли", "загугли", "что нового", "последние новости",
             "свежие новости", "в интернете", "поиск в сети")
THINK_KW = ("почему", "объясни", "сравни", "план", "как накопить", "докажи", "проанализируй")
_MUSIC_CUE = ("музык", "трек", "плейлист", "песн", "поставь что", "включи что")
_VAGUE = ("что-нибудь", "что нибудь", "на свой вкус", "на твой вкус", "как обычно",
          "любимое", "что хочешь", "что послушать", "что посоветуешь")


def vague_music(low):
    """True for underspecified music requests → resolve from preference profile."""
    bare = low.strip() in ("включи музыку", "поставь музыку", "включи музыку.", "поставь музыку.")
    music_word = any(k in low for k in _MUSIC_CUE)
    play_verb = "поставь" in low or "включи" in low
    vague_word = any(v in low for v in _VAGUE)
    return bare or (music_word and vague_word) or (play_verb and vague_word)


_GENRES = ("lo-fi", "лоуфай", "лофай", "джаз", "jazz", "классик", "рок", "rock", "рэп",
           "rap", "хип-хоп", "поп", "pop", "эмбиент", "ambient", "электрон", "techno",
           "метал", "блюз", "регги", "кантри", "саундтрек", "шансон", "релакс", "chill")
_DEVICE_WORD = ("свет", "ламп", "будильник", "таймер", "громкост", "температур")


def explicit_music(low):
    """True for an explicit music command ('включи джаз') → force play_music."""
    play = any(v in low for v in ("включи", "поставь", "врубай", "врубни"))
    if not play or any(d in low for d in _DEVICE_WORD):
        return False
    return any(g in low for g in _GENRES) or any(w in low for w in ("музык", "трек", "плейлист", "песн"))


SYS_CALC = ("Ты — Qiyas Edge. Для ЛЮБЫХ арифметических вычислений ОБЯЗАТЕЛЬНО вызывай "
            "инструмент calculate (выражение в Python-синтаксисе). Никогда не считай в уме.")
SYS_TIME = ("Ты — Qiyas Edge. Чтобы узнать текущую дату или время, ОБЯЗАТЕЛЬНО вызови "
            "инструмент get_datetime. Не выдумывай дату.")
SYS_DEV = "Ты — Qiyas Edge. Для управления устройствами/заметками/музыкой вызывай подходящий инструмент."
SYS_SEARCH = "Ты — Qiyas Edge. Вызови web_search для свежей информации, затем кратко ответь пользователю."
SYS_COMPANION = ("Ты — Qiyas Edge, тёплый и живой собеседник. Сначала ответь по существу "
                 "или выполни просьбу, потом при уместности задай один короткий "
                 "уточняющий вопрос. Эмпатично, естественно, кратко, без официоза и без "
                 "длинных вступлений.")
COMPANION_KW = ("привет", "как дела", "как настроение", "как ты", "грустно", "грусть",
                "тяжело", "устал", "одиноко", "скучно", "тревож", "переживаю", "поболта",
                "поговор", "подними настроение", "настроение", "как прошёл", "поддерж")
SYS_NUDGE = ("Ты — Qiyas Edge. На основе привычки пользователя сформулируй ОДНО "
             "короткое тёплое предложение-напоминание (мягкий вопрос + уместный эмодзи). "
             "Без вступлений и пояснений — только сама фраза, по-дружески, на «ты».")


def route(q):
    low = q.lower()
    if any(k in low for k in CREATIVE_KW):
        return {"mode": "creative", "system": SYS_CREATIVE, "tools": None, "think": False, "temp": 0.65}
    if any(k in low for k in TIME_KW):
        now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M, %A")
        return {"mode": "analytical",
                "system": f"Сегодня {now}. Ответь пользователю, опираясь на эту дату/время.",
                "tools": None, "think": False, "temp": 0.0}
    if any(k in low for k in SEARCH_KW):
        return {"mode": "analytical", "system": None, "tools": [SEARCH_TOOL],
                "think": False, "temp": 0.0, "force": ("web_search", "query")}
    if explicit_music(low):
        return {"mode": "analytical", "system": SYS_DEV, "tools": [MEDIA_TOOL],
                "think": False, "temp": 0.0, "force": ("play_music", "query")}
    if any(k in low for k in TOOL_KW):
        return {"mode": "analytical", "system": SYS_DEV, "tools": ACTION_TOOLS, "think": False, "temp": 0.0}
    if any(k in low for k in COMPUTE_KW) or re.search(r"\d+\s*[+\-*/^%]\s*\d+", q):
        return {"mode": "analytical", "system": None, "tools": [CALC_TOOL],
                "think": False, "temp": 0.0, "force": ("calculate", "expression")}
    if any(k in low for k in COMPANION_KW):
        return {"mode": "analytical", "system": SYS_COMPANION, "tools": None,
                "think": False, "temp": 0.65, "rep_penalty": 1.2}
    think = any(k in low for k in THINK_KW) or len(q) > 120
    return {"mode": "analytical", "system": None, "tools": None, "think": think, "temp": 0.0}


def _strip(s):
    return re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()


class Agent:
    def __init__(self):
        print(f"loading base {BASE} (text-only, vision-unload) ...", flush=True)
        self.model, self.tok, stats = load_text_only(BASE)
        print(f"  {stats}", flush=True)
        self._cur = None

    def set_mode(self, mode):
        adapter = ADAPTERS[mode]
        if adapter == self._cur:
            return
        if self._cur is not None:
            remove_lora_layers(self.model)
        load_adapters(self.model, adapter)
        self._cur = adapter

    # ---- Memory Plant integration (subprocess -> system python3) ----
    def retrieve(self, q):
        try:
            r = subprocess.run([MP_PY, MP_BRIDGE, "search", q],
                               capture_output=True, text=True, timeout=90)
            return r.stdout.strip()
        except Exception:
            return ""

    def ingest(self, q, a=None):
        if not should_remember(q):
            return
        try:
            subprocess.run([MP_PY, MP_BRIDGE, "ingest", q],
                           capture_output=True, text=True, timeout=90)
        except Exception:
            pass
    # (hobby RAG packs plug into retrieve() the same way later)
    # -----------------------------------------------------------------

    def _gen(self, msgs, think, temp, tools=None, tool_choice=None, rep_penalty=None):
        kw = {"add_generation_prompt": True, "enable_thinking": think}
        if tools:
            kw["tools"] = tools
            if tool_choice:
                kw["tool_choice"] = tool_choice
        prompt = self.tok.apply_chat_template(msgs, **kw)
        gkw = {"sampler": make_sampler(temp=temp)}
        if rep_penalty:
            gkw["logits_processors"] = make_logits_processors(
                repetition_penalty=rep_penalty, repetition_context_size=40)
        return generate(self.model, self.tok, prompt=prompt, max_tokens=320,
                        verbose=False, **gkw).strip()

    def _gen_forced(self, msgs, fn, pn, tools):
        """Force a specific tool call by PREFILLING the tool-call opening."""
        prompt = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, tools=tools,
            enable_thinking=False, tokenize=False)
        pre = f"<tool_call>\n<function={fn}>\n<parameter={pn}>\n"
        o = generate(self.model, self.tok, prompt=prompt + pre, max_tokens=120,
                     verbose=False, sampler=make_sampler(temp=0.0))
        return pre + o

    def _music_pref(self, q):
        """Vague music request → play the favorite from the learned profile."""
        top = prefs.top_music()
        if top:
            prefs.log_play(top)                      # reinforce
            return {"mode": "analytical", "think": False,
                    "tool": f"play_music({{'query': '{top}'}})",
                    "result": f"играю '{top}'",
                    "answer": f"Ставлю {top} — твоё любимое. 🎵"}
        return {"mode": "analytical", "think": False, "tool": None, "result": None,
                "answer": "Что поставить? Пока не знаю твоих вкусов — назови жанр."}

    def respond(self, q):
        if vague_music(q.lower()):
            return self._music_pref(q)
        d = route(q)
        self.set_mode(d["mode"])
        # retrieve memory only for analytical Q&A (skip commands/creative)
        mem = self.retrieve(q) if (d["mode"] == "analytical" and not d["tools"]) else ""
        sys_content = d["system"]
        if mem and not sys_content:
            sys_content = ("Ты — Qiyas Edge. В блоке [Память] — факты о ПОЛЬЗОВАТЕЛЕ "
                           "(не о тебе). Отвечай ему на «ты», опираясь на эти факты.")
        msgs = ([{"role": "system", "content": sys_content}] if sys_content else [])
        user = f"[Память]\n{mem}\n\n{q}" if mem else q
        msgs.append({"role": "user", "content": user})
        if d.get("force"):
            fn, pn = d["force"]
            out = self._gen_forced(msgs, fn, pn, d["tools"])
        else:
            out = self._gen(msgs, d["think"], d["temp"], d["tools"], d.get("tool_choice"),
                            d.get("rep_penalty"))
        tc = parse_tool_call(out)
        if tc:
            name, params = tc
            result = exec_tool(name, params)
            final = self._gen(
                [{"role": "user", "content":
                  f"Запрос: {q}\nИнструмент {name} вернул:\n{result}\n\n"
                  f"Ответь пользователю кратко по-русски ТОЛЬКО на основе этого результата. "
                  f"Если результат пустой / '(ничего не найдено)' / ошибка — честно скажи, "
                  f"что не нашёл, и НИЧЕГО не выдумывай."}],
                think=False, temp=0.0)
            self.ingest(q, final)
            return {"mode": d["mode"], "think": d["think"], "tool": f"{name}({params})",
                    "result": result, "answer": _strip(final)}
        ans = _strip(out)
        self.ingest(q, ans)
        return {"mode": d["mode"], "think": d["think"], "tool": None, "result": None, "answer": ans}

    # ---- proactive triggers (phone scheduler fires this; cf. hobby_pack.pack_tick) ----
    def nudge_tick(self, now=None):
        """Proactive check. Returns a nudge to surface, or None (stay silent).
        Restraint lives in proactive.due(); the 4B PHRASES it (template fallback)."""
        n = proactive.due(now=now)
        if not n:
            return None
        if n["cat"] in proactive.TEMPLATE_CATS:
            say = n["say"]                      # item-specific → exact template
        else:                                   # soft category → 4B phrases warmly
            self.set_mode("analytical")
            msgs = [{"role": "system", "content": SYS_NUDGE},
                    {"role": "user", "content": f"Привычка: {n['text']}. Предложи мягко, одной фразой."}]
            try:
                say = _strip(self._gen(msgs, think=False, temp=0.6, rep_penalty=1.2))
                say = say.splitlines()[0].strip() if say else ""
            except Exception:
                say = ""
            if not say or len(say) > 160:
                say = n["say"]                  # fallback to deterministic template
        proactive.mark_sent(n, now=now)
        return {**n, "say": say}

    def nudge_feedback(self, nudge, accepted):
        """User reacted to a nudge: accepted → reinforce, dismissed → back off."""
        (proactive.accept if accepted else proactive.dismiss)(nudge)
        return "учту 👍" if accepted else "понял, буду реже"
    # -----------------------------------------------------------------


_YES = ("да", "ага", "давай", "ок", "окей", "yes", "y", "+", "конечно", "го", "угу")


def _maybe_nudge(a):
    """Opportunistic proactive check between turns. All restraint lives in
    nudge_tick()/proactive.due(); here we just surface it and capture the reaction."""
    n = a.nudge_tick()
    if not n:
        return
    print(f"   💡 {n['say']}")
    try:
        ans = input("      [да/нет] > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    accepted = any(ans == w or ans.startswith(w + " ") for w in _YES)
    print(f"      {a.nudge_feedback(n, accepted)}\n")


def chat():
    """Interactive REPL with opportunistic proactive nudges between turns.
    In production the phone scheduler fires nudge_tick() on a timer; here we check it
    on open and after each reply (guardrails keep it from nagging)."""
    a = Agent()
    print("\nQiyas Edge — чат (проактивные подсказки включены). 'выход' — выйти.\n", flush=True)
    _maybe_nudge(a)                        # a nudge may greet you on open
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("выход", "выйти", "exit", "quit", ":q"):
            print("Пока! 👋")
            break
        r = a.respond(q)
        if r["tool"]:
            print(f"   ⚙ {r['tool']} → {r['result']}")
        print(f"   {r['answer']}\n")
        _maybe_nudge(a)                    # check again after each reply


DEMO = [
    "Сколько будет 15% от 240?",              # force calc -> 36
    "Посчитай корень из 2 умножить на 100.",  # force calc -> 141.42
    "Напомни позвонить маме в 18:00.",        # create_reminder
    "Запиши заметку: купить молоко и хлеб.",   # create_note
    "Включи музыку — что-нибудь спокойное.",   # play_music
    "Найди, что нового в Python 3.13.",       # web_search
    "Привет! Как настроение?",                # companion (warm + temp 0.65 + rep_penalty)
    "Мне сегодня немного грустно.",            # companion
]


def main():
    if "--chat" in sys.argv[1:]:
        chat()
        return
    a = Agent()
    qs = [sys.argv[1]] if len(sys.argv) > 1 else DEMO
    for q in qs:
        r = a.respond(q)
        print(f"\n>>> {q}\n[{r['mode']}|think={r['think']}]")
        if r["tool"]:
            print(f"   TOOL {r['tool']} -> {r['result']}")
        print(f"   {r['answer'][:240]}")


if __name__ == "__main__":
    main()
