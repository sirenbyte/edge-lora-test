"""Generate Russian AGREEMENT preference pairs for DPO.

chosen = grammatically correct, rejected = a single broken-agreement twin.
Templated + word-lists → the rejected is GUARANTEED ungrammatical (far more reliable
than auto-corrupting a corpus, and needs no pymorphy). Targets exactly the phenomena
eval_rublimp.py measures, so DPO can align GENERATION to the model's own 93%
log-prob preference (the diagnosed root: knowledge is there, decoding slips).

  python dpo_data.py [N]   ->  data_dpo/train.jsonl  (+ valid.jsonl)
"""
import json
import random
import sys
from pathlib import Path

random.seed(13)
DATA = Path(__file__).parent / "data_dpo"

_PLACE = ["во дворе", "в парке", "дома", "на улице", "в зале", "на работе", "в кафе", "в школе"]


def verb_number():
    subj = random.choice(["дети", "студенты", "рабочие", "друзья", "соседи", "коллеги",
                          "туристы", "музыканты", "гости", "ученики"])
    vp, vs = random.choice([("играют", "играет"), ("работают", "работает"), ("идут", "идёт"),
                            ("поют", "поёт"), ("читают", "читает"), ("спорят", "спорит"),
                            ("смеются", "смеётся"), ("ждут", "ждёт"), ("бегут", "бежит")])
    p = random.choice(_PLACE)
    return f"{subj.capitalize()} {vp} {p}.", f"{subj.capitalize()} {vs} {p}.", "verb-number"


def verb_person():
    pron, vg, vb = random.choice([
        ("Я", "живу", "живёшь"), ("Я", "иду", "идёшь"), ("Я", "работаю", "работает"),
        ("Ты", "живёшь", "живут"), ("Ты", "идёшь", "иду"), ("Ты", "знаешь", "знают"),
        ("Они", "работают", "работаешь"), ("Они", "идут", "идёт"), ("Мы", "идём", "идёт"),
    ])
    tail = random.choice(["в город", "домой", "на работу", "в кино", "в парк", "по делам"])
    return f"{pron} {vg} {tail}.", f"{pron} {vb} {tail}.", "verb-person"


def adj_gender():
    adj_f, adj_m, noun = random.choice([
        ("красивая", "красивый", "девушка"), ("тёплая", "тёплый", "куртка"),
        ("новая", "новый", "машина"), ("большая", "большой", "комната"),
        ("интересная", "интересный", "книга"), ("вкусная", "вкусный", "еда"),
        ("старая", "старый", "школа"), ("умная", "умный", "собака"),
        ("яркая", "яркий", "лампа"), ("длинная", "длинный", "дорога"),
        ("свежая", "свежий", "газета"), ("высокая", "высокий", "башня"),
        ("добрая", "добрый", "женщина"), ("быстрая", "быстрый", "река"),
        ("тихая", "тихий", "улица"), ("чистая", "чистый", "вода")])
    pre = random.choice(["Это", "Какая", "Вот", "Очень"])
    art = "" if pre in ("Это", "Вот") else ""
    return f"{pre} {adj_f} {noun}.", f"{pre} {adj_m} {noun}.", "adj-gender"


def adj_prep_case():
    adj_loc, adj_nom, noun = random.choice([
        ("большом", "большой", "городе"), ("маленькой", "маленький", "деревне"),
        ("новом", "новый", "доме"), ("тихой", "тихий", "комнате"),
        ("старом", "старый", "парке"), ("светлой", "светлый", "квартире")])
    return (f"Он живёт в {adj_loc} {noun}.", f"Он живёт в {adj_nom} {noun}.", "adj-prep-case")


def numeral_gen():
    num, gen_pl, wrong = random.choice([
        ("пять", "книг", "книги"), ("шесть", "столов", "стола"), ("семь", "домов", "дома"),
        ("десять", "рублей", "рубля"), ("много", "людей", "человеки"),
        ("несколько", "минут", "минуты"), ("восемь", "часов", "часа")])
    return f"У меня {num} {gen_pl}.", f"У меня {num} {wrong}.", "numeral-gen"


def numeral_two():
    obj_good, obj_bad = random.choice([("яблока", "яблоко"), ("стола", "стол"),
                                       ("часа", "час"), ("дома", "дом"), ("года", "лет")])
    n = random.choice(["два", "три", "четыре"])
    return f"Здесь {n} {obj_good}.", f"Здесь {n} {obj_bad}.", "numeral-two"


def past_gender():
    subj, vf, vm = random.choice([
        ("Она", "пришла", "пришёл"), ("Девочка", "читала", "читал"),
        ("Мама", "готовила", "готовил"), ("Сестра", "уехала", "уехал"),
        ("Она", "сказала", "сказал"), ("Кошка", "спала", "спал")])
    tail = random.choice(["домой", "книгу", "ужин", "вчера", "утром"])
    return f"{subj} {vf} {tail}.", f"{subj} {vm} {tail}.", "past-gender"


def poss_gender():
    pf, pm, noun = random.choice([
        ("Моя", "Мой", "сестра"), ("Моя", "Мой", "мама"), ("Моя", "Мой", "комната"),
        ("Твоя", "Твой", "книга"), ("Наша", "Наш", "команда"), ("Моя", "Мой", "идея")])
    return f"{pf} {noun} — лучшая.", f"{pm} {noun} — лучшая.", "poss-gender"


def genitive_neg():
    good, bad = random.choice([("времени", "время"), ("проблемы", "проблема"),
                               ("денег", "деньги"), ("идеи", "идея"), ("ответа", "ответ")])
    pre = random.choice(["У меня нет", "У него нет", "Здесь нет", "Больше нет"])
    return f"{pre} {good}.", f"{pre} {bad}.", "genitive-neg"


GENERATORS = [verb_number, verb_person, adj_gender, adj_prep_case, numeral_gen,
              numeral_two, past_gender, poss_gender, genitive_neg]


def build(n: int):
    """Round-robin across generators with a per-phenomenon cap → balanced coverage
    (else verb-number, having the most combos, drowns everything)."""
    cap = max(20, n // len(GENERATORS) + 10)
    seen, pairs = set(), []
    per = {g.__name__: 0 for g in GENERATORS}
    stalls = 0
    while len(pairs) < n and stalls < len(GENERATORS) * 60:
        progressed = False
        for g in GENERATORS:
            if len(pairs) >= n or per[g.__name__] >= cap:
                continue
            good, bad, phen = g()
            if good == bad or good in seen:
                continue
            seen.add(good)
            per[g.__name__] += 1
            pairs.append({"prompt": "", "chosen": good, "rejected": bad, "phenomenon": phen})
            progressed = True
        stalls = 0 if progressed else stalls + 1
    return pairs


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    pairs = build(n)
    random.shuffle(pairs)
    k = max(1, len(pairs) // 12)
    valid, train = pairs[:k], pairs[k:]
    DATA.mkdir(exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        with (DATA / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter(r["phenomenon"] for r in pairs)
    print(f"DPO pairs: train={len(train)} valid={len(valid)} -> {DATA}")
    print("by phenomenon:", dict(c))
    print("samples:")
    for r in pairs[:4]:
        print(f"  + «{r['chosen']}»  −«{r['rejected']}»  [{r['phenomenon']}]")


if __name__ == "__main__":
    main()
