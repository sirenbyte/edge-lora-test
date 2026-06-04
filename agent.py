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
import sys
import urllib.parse
import urllib.request

from mlx_lm import generate, load  # noqa: F401  (load kept for compat)
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.tuner.utils import load_adapters, remove_lora_layers

import hobby_pack
import prefs
import proactive
import sandbox
from model_router import (classify as _classify, extract_fact as _extract_fact,
                          which_fact as _which_fact)
from verifier import lexical_fix
from vision_unload import load_text_only

BASE = "mlx-community/Qwen3.5-4B-MLX-4bit"
ADAPTERS = {"analytical": "adapters_qwen4b_v2", "creative": "adapters_qwen4b_creative"}
SYS_CREATIVE = ("Ты — Qiyas Edge в творческом режиме. Будь изобретательным: давай "
                "разнообразные, яркие, образные идеи. Сразу к делу, без вступлений.")
ROOM_ENUM = ["кухня", "спальня", "гостиная", "ванная", "коридор"]

# ---- memory: in-process Rust core (memory_plant_rs) + e5 via MLX, see memory.py ----
# (replaces the old system-python mp_bridge subprocess — one runtime now)
REMEMBER_KW = ("меня зовут", "моё имя", "мое имя", "я люблю", "я увлекаюсь", "запомни",
               "я работаю", "я живу", "я предпочитаю", "мне нравится", "мой ", "моя ", "мои ")


_Q_WORDS = ("как", "что", "где", "кто", "когда", "почему", "чем", "какой",
            "какая", "сколько", "зачем", "куда")


_SELF_REF = ("я ", "у меня", "мой ", "моя ", "мои ", "мне ", "меня ")
# first-person openers that are intents/opinions/chatter, NOT stable facts
_NOT_FACT = ("я хочу", "я думаю", "я не ", "я бы", "я просто", "я уже", "я тоже",
             "я сейчас", "я тут", "я здесь", "я считаю", "я могу", "я буду")


def _is_question(q):
    low = q.lower().strip()
    return low.endswith("?") or bool(low.split()) and low.split()[0] in _Q_WORDS


_SELF_WORDS = {"я", "меня", "мне", "мной", "мой", "моя", "моё", "мои", "моих", "обо"}


def _about_self(q):
    return bool({w.strip("?.,!") for w in q.lower().split()} & _SELF_WORDS)


# info/creative requests are NOT personal facts, even if mislabeled 'fact'
_IMPERATIVE = ("расскажи", "объясни", "опиши", "покажи", "посоветуй", "подскажи",
               "найди", "придумай", "сочини", "переведи", "посчитай", "напиши")


def _is_request(q):
    return q.lower().lstrip().startswith(_IMPERATIVE)


def should_remember(q):
    low = q.lower().strip()
    if _is_question(q):
        return False                            # questions, not statements
    if any(k in low for k in COMPANION_KW):     # greetings / moods / small talk
        return False
    if any(low.startswith(e) for e in _NOT_FACT):
        return False                            # intents / opinions, not facts
    if any(k in low for k in REMEMBER_KW):
        return True                             # explicit fact markers
    return low.startswith(_SELF_REF)            # general first-person declarative = a fact

# ---------- tool schemas ----------
CALC_TOOL = {"type": "function", "function": {
    "name": "calculate", "description": "Точно вычислить математическое выражение.",
    "parameters": {"type": "object", "properties": {
        "expression": {"type": "string", "description": "Python-синтаксис, напр. 0.15*240 или 47*83+12"}},
        "required": ["expression"]}}}
TIME_TOOL = {"type": "function", "function": {
    "name": "get_datetime", "description": "Текущие дата и время.",
    "parameters": {"type": "object", "properties": {}}}}
RUN_PYTHON_TOOL = {"type": "function", "function": {
    "name": "run_python", "description": "Выполнить Python-код для ТОЧНЫХ вычислений: "
        "даты/время, многошаговые расчёты, обработка списков и строк. Код ОБЯЗАН print() ответ.",
    "parameters": {"type": "object", "properties": {
        "code": {"type": "string", "description": "Python-код, печатающий ответ через print()"}},
        "required": ["code"]}}}
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
    if name == "run_python":
        return sandbox.run_python(p.get("code", ""))
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
# multi-step / date / list-string → run_python (code), NOT the one-line calculator
CODE_KW = ("дней до", "дней между", "дней назад", "день недели", "через сколько",
           "сколько дней", "сколько недель", "сколько часов", "сколько секунд",
           "сколько минут", "дата через", "по дате", "лет между", "отсортируй",
           "сколько слов", "сколько букв", "сколько символов", "среднее", "медиан",
           "сумма", "произведение", "факториал", "фибоначч", "наибольш", "наименьш",
           "сколько раз", "по возрастанию", "по убыванию", "разница дат")
TIME_KW = ("который час", "сколько времени", "какое сегодня", "какое число", "какой день", "дата")
SEARCH_KW = ("найди", "поищи", "погугли", "загугли", "что нового", "последние новости",
             "свежие новости", "в интернете", "поиск в сети")
THINK_KW = ("почему", "объясни", "сравни", "план", "как накопить", "докажи", "проанализируй",
            "реши", "задач", "пошагов", "стратег", "разбери", "логич", "рассуди",
            "как лучше", "что выбрать", "взвесь", "обоснуй", "по шагам")
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
SYS_BASE = ("Ты — Qiyas Edge, локальный ИИ-ассистент. Отвечай точно и по делу. "
            "НЕ соглашайся автоматически: если пользователь ошибается или противоречит "
            "фактам/правилам — спокойно и вежливо поправь, обоснуй, не поддакивай.")
SYS_COMPANION = ("Ты — Qiyas Edge, тёплый и живой собеседник. Сначала ответь по существу "
                 "или выполни просьбу, потом при уместности задай один короткий "
                 "уточняющий вопрос. Эмпатично, естественно, кратко, без официоза и без "
                 "длинных вступлений. Будь искренним: не поддакивай против фактов — мягко поправь.")
COMPANION_KW = ("привет", "как дела", "как настроение", "как ты", "грустно", "грусть",
                "тяжело", "устал", "одиноко", "скучно", "тревож", "переживаю", "поболта",
                "поговор", "подними настроение", "настроение", "как прошёл", "поддерж",
                "страх", "страшно", "боюсь", "боязн", "паник", "злюсь", "злость", "зол",
                "обид", "стыд", "вина", "виноват", "раздраж", "беспоко", "тоск", "плохо",
                "плачу", "плакать", "депресс", "выгор", "нерв", "волну")
# meta-questions about the assistant's own capability/intelligence (else the
# identity-overfit adapter just loops "Я Qiyas Edge…")
# the identity-overfit adapter ignores any system prompt here and loops "Я Qiyas
# Edge…", so we answer this meta-question DETERMINISTICALLY (honest, fixed).
SELF_ANSWER = (
    "Честно — я небольшая модель (около 4 млрд параметров), работаю офлайн прямо на "
    "устройстве. Я не такой умный, как огромные облачные модели вроде GPT-4, зато "
    "приватный и быстрый: хорошо справляюсь с обиходными задачами, памятью, "
    "инструментами и управлением устройством. Сложное могу передать модели помощнее. 🙂")
SELF_KW = ("ты умный", "ты умён", "насколько умный", "насколько ты", "ты тупой",
           "ты глупый", "ты способ", "на что ты способ", "что ты умеешь", "что умеешь",
           "что ты можешь", "какой ты", "ты крут", "ты лучше", "умный ли ты", "ты толков")
# bare one-word follow-ups to "ты умный?" ('умный', 'насколько', 'тупой') lose
# context and make the identity adapter ramble with person confusion -> catch them.
_SELF_BARE = {"умный", "умён", "умен", "тупой", "глупый", "туповат", "насколько",
              "способный", "крутой", "толковый", "умница", "сообразительный", "глуп"}


def _is_self(low):
    s = low.strip(" ?.!,")
    if any(k in s for k in SELF_KW):
        return True
    w = s.split()
    return len(w) <= 2 and bool(set(w) & _SELF_BARE)
SYS_NUDGE = ("Ты — Qiyas Edge. На основе привычки пользователя сформулируй ОДНО "
             "короткое тёплое предложение-напоминание (мягкий вопрос + уместный эмодзи). "
             "Без вступлений и пояснений — только сама фраза, по-дружески, на «ты».")


def route(q):
    low = q.lower()
    if any(k in low for k in CREATIVE_KW):
        return {"mode": "creative", "system": SYS_CREATIVE, "tools": None, "think": False, "temp": 0.65}
    if any(k in low for k in CODE_KW):          # computed/historical dates & multi-step
        today = _dt.datetime.now().strftime("%Y-%m-%d, %A")
        return {"mode": "analytical",
                "system": (f"Ты — Qiyas Edge. Сегодня {today}. Реши задачу инструментом "
                           "run_python: короткий Python-код, который вычисляет ответ и печатает "
                           "его через print(). Для дат используй datetime. Не считай в уме."),
                "tools": [RUN_PYTHON_TOOL], "think": False, "temp": 0.0,
                "force": ("run_python", "code")}
    if any(k in low for k in TIME_KW):          # CURRENT date/time only
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
    if _is_self(low):
        return {"mode": "analytical", "system": None, "tools": None,
                "think": False, "temp": 0.0, "self": True}
    if any(k in low for k in COMPANION_KW):
        return {"mode": "analytical", "system": SYS_COMPANION, "tools": None,
                "think": False, "temp": 0.5, "rep_penalty": 1.2}
    think = any(k in low for k in THINK_KW) or len(q) > 120
    return {"mode": "analytical", "system": None, "tools": None, "think": think, "temp": 0.0}


def _label_to_route(label, q):
    """Map a model-router intent label to a route() dict (None = keep default)."""
    if label == "creative":
        return {"mode": "creative", "system": SYS_CREATIVE, "tools": None, "think": False, "temp": 0.65}
    if label == "command":
        return {"mode": "analytical", "system": SYS_DEV, "tools": ACTION_TOOLS, "think": False, "temp": 0.0}
    if label == "search":
        return {"mode": "analytical", "system": None, "tools": [SEARCH_TOOL],
                "think": False, "temp": 0.0, "force": ("web_search", "query")}
    if label == "math":
        return {"mode": "analytical", "system": None, "tools": [CALC_TOOL],
                "think": False, "temp": 0.0, "force": ("calculate", "expression")}
    if label == "time":
        now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M, %A")
        return {"mode": "analytical", "system": f"Сегодня {now}. Ответь, опираясь на эту дату/время.",
                "tools": None, "think": False, "temp": 0.0}
    if label == "companion":
        return {"mode": "analytical", "system": SYS_COMPANION, "tools": None,
                "think": False, "temp": 0.5, "rep_penalty": 1.2}
    if label == "fact":
        return {"mode": "analytical", "system": None, "tools": None,
                "think": False, "temp": 0.0, "fact": True}
    return None                              # question / unknown -> keep keyword default


def _strip(s):
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    s = re.sub(r"<tool_call>.*?(</tool_call>|$)", "", s, flags=re.S)  # drop stray/partial tool tags
    return s.strip()


def _repair(name, params):
    """Deterministic tool-arg repair (cheap grammar-constraint substitute): snap enum
    declensions ('в спальне'->'спальня'), coerce numeric args. Kills the format-failure
    class without an llguidance grammar (noted as production hardening)."""
    p = dict(params)
    if "room" in p and p["room"] not in ROOM_ENUM:
        low = p["room"].lower()
        for r in ROOM_ENUM:
            if r[:4] in low:                     # спальн -> спальне/в спальне ...
                p["room"] = r
                break
    for k in ("minutes", "level"):
        if k in p and not str(p[k]).strip().isdigit():
            m = re.search(r"\d+", str(p[k]))
            if m:
                p[k] = m.group()
    return name, p


class Agent:
    def __init__(self):
        print(f"loading base {BASE} (text-only, vision-unload) ...", flush=True)
        self.model, self.tok, stats = load_text_only(BASE)
        print(f"  {stats}", flush=True)
        self._cur = None
        self._mem = None                       # lazy in-process Memory (e5 + Rust core)
        self.use_model_router = True           # hybrid: model disambiguates uncertain turns

    def set_mode(self, mode):
        adapter = ADAPTERS[mode]
        if adapter == self._cur:
            return
        if self._cur is not None:
            remove_lora_layers(self.model)
        load_adapters(self.model, adapter)
        self._cur = adapter

    # ---- Memory Plant integration (in-process: memory_plant_rs + e5/MLX) ----
    @property
    def mem(self):
        """Lazy — load e5 + the Rust doc store only when memory is first used."""
        if self._mem is None:
            from memory import Memory
            self._mem = Memory()
        return self._mem

    def retrieve(self, q):
        try:
            # inject user facts ONLY when the turn is about the user — otherwise a
            # vague turn ("насколько") drowns in facts and the model grabs a random one.
            facts = self.mem.relevant_facts(q) if _about_self(q) else ""
            docs = self.mem.retrieve_texts(q, k=3, min_score=0.4)
            return "\n".join(([facts] if facts else []) + docs).strip()
        except Exception:
            return ""

    def ingest(self, q, a=None):
        if not should_remember(q):
            return
        try:
            import hashlib
            doc_id = "msg_" + hashlib.md5(q.encode("utf-8")).hexdigest()[:10]
            self.mem.store_document(doc_id, q)
            self.mem.save()
        except Exception:
            pass
    # (hobby RAG packs share the same Memory.store_document/retrieve)
    # -----------------------------------------------------------------

    def _gen(self, msgs, think, temp, tools=None, tool_choice=None, rep_penalty=None,
             max_tokens=None):
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
        # trade latency for quality: thinking turns need room for scratchpad + answer
        mt = max_tokens if max_tokens is not None else (700 if think else 320)
        return generate(self.model, self.tok, prompt=prompt, max_tokens=mt,
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

    def _route(self, q, history=None):
        """Hybrid: keyword fast-path; consult the 4B ONLY when keyword is uncertain
        (fell through to the plain default). System-1 reflex + System-2 when needed."""
        d = route(q)
        uncertain = (d["mode"] == "analytical" and d["system"] is None
                     and not d["tools"] and not d.get("force") and not d.get("self"))
        if self.use_model_router and uncertain:
            d2 = _label_to_route(_classify(self.model, self.tok, q, history), q)
            if d2 is not None:
                return d2
        return d

    def respond(self, q, history=None):
        if vague_music(q.lower()):
            return self._music_pref(q)
        d = self._route(q, history)
        if d.get("self"):                          # "насколько ты умный" -> honest, fixed
            return {"mode": "analytical", "think": False, "tool": None,
                    "result": None, "answer": SELF_ANSWER}
        # personal FACT statement -> store + brief ack, instead of rambling.
        # A question is never a fact, even if the model mislabeled it 'fact'.
        is_fact = ((should_remember(q) or (d.get("fact") and not _is_question(q)))
                   and not _is_request(q))           # 'расскажи про X' is not a fact
        if (d["mode"] == "analytical" and d["system"] is None and not d["tools"]
                and not d.get("force") and is_fact):
            self.set_mode("analytical")
            pred, val = _extract_fact(self.model, self.tok, q)
            if pred and val:
                norm = self.mem.remember_fact(pred, val)   # structured + 2nd-person + persist
                ans = f"Запомнил 👍 {norm}"
            else:
                self.ingest(q)                             # not cleanly structured -> raw doc
                ans = "Запомнил 👍"
            return {"mode": "analytical", "think": False, "tool": None,
                    "result": None, "answer": ans}
        # question about ONE stored fact -> the model picks which stored field it is
        # (or none), and we answer that saved line VERBATIM (no paraphrase → no grammar
        # slips like "вешаешь"/"меня зовут"). General questions fall through to gen.
        if (d["system"] is None and not d["tools"] and not d.get("force")
                and _is_question(q) and _about_self(q)):
            pred = _which_fact(self.model, self.tok, q, self.mem.known_predicates())
            line = self.mem.fact_line(pred) if pred else None
            if line:
                return {"mode": "analytical", "think": False, "tool": None,
                        "result": None, "answer": line}
        self.set_mode(d["mode"])
        # retrieve facts ONLY on the plain factual-Q&A path — NOT companion/tools/creative
        # (so emotional turns like "страх" get empathy, not a memory dump).
        plain = d["system"] is None and not d["tools"] and not d.get("force")
        mem = self.retrieve(q) if plain else ""
        sys_content = d["system"]
        if not sys_content and not mem:
            sys_content = SYS_BASE              # default path: identity + anti-sycophancy
        if mem and not sys_content:
            sys_content = ("Ты — Qiyas Edge. В блоке [Память] — факты о ПОЛЬЗОВАТЕЛЕ. "
                           "Обращайся к нему НА «ТЫ» и говори о нём в 3-м лице: "
                           "«тебя зовут…», «ты живёшь…», «ты весишь…» (НЕ «меня зовут», НЕ «я живу»). "
                           "Опирайся ТОЛЬКО на факты из [Память]. Если нужного факта там нет — "
                           "честно скажи, что не знаешь / пользователь не говорил, и НЕ выдумывай.")
        msgs = ([{"role": "system", "content": sys_content}] if sys_content else [])
        if history and not d.get("force"):
            msgs += history[-6:]          # last ~3 turns → conversational continuity
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
            name, params = _repair(*tc)         # snap enum declensions / coerce numerics
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
                    "result": result, "answer": lexical_fix(_strip(final))}
        ans = lexical_fix(_strip(out))         # proofreader: surgical person/verb fixes
        self.ingest(q, ans)
        return {"mode": d["mode"], "think": d["think"], "tool": None, "result": None, "answer": ans}

    # ---- proactive triggers (phone scheduler fires this; cf. hobby_pack.pack_tick) ----
    def nudge_tick(self, now=None):
        """Proactive check. Returns a nudge to surface, or None (stay silent).
        Restraint lives in proactive.due(); the 4B PHRASES it (template fallback)."""
        n = proactive.due(now=now)
        if not n:                               # no habit nudge → maybe a hobby digest
            hobby = hobby_pack.latest_pack_hobby() or hobby_pack.user_hobby()
            dn = proactive.digest_due(hobby, now=now) if hobby else None
            if dn:
                proactive.mark_sent(dn, now=now)
            return dn
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
    if n["cat"] == "digest":                       # content offer, not a habit
        if accepted:
            print("      собираю свежий дайджест…\n")
            print(hobby_pack.generate_digest(n["item"], agent=a) + "\n")
        else:
            print("      ок, не сейчас 👍\n")
        return
    print(f"      {a.nudge_feedback(n, accepted)}\n")


def chat():
    """Interactive REPL with opportunistic proactive nudges between turns.
    In production the phone scheduler fires nudge_tick() on a timer; here we check it
    on open and after each reply (guardrails keep it from nagging)."""
    a = Agent()
    print("\nQiyas Edge — чат (проактивные подсказки включены). 'выход' — выйти.\n", flush=True)
    _maybe_nudge(a)                        # a nudge may greet you on open
    hist = []                              # rolling conversation history for continuity
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
        r = a.respond(q, history=hist)
        if r["tool"]:
            print(f"   ⚙ {r['tool']} → {r['result']}")
        print(f"   {r['answer']}\n")
        hist += [{"role": "user", "content": q},
                 {"role": "assistant", "content": r["answer"]}]
        hist = hist[-6:]                   # keep last ~3 turns
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
