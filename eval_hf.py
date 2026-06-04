"""Honest before/after eval on CUDA (transformers): generalization + regression.

Mirrors the MLX eval.py: domain held-out paraphrases (generalization) and
general-capability probes (catastrophic forgetting). Loads the 4-bit base,
optionally applies a LoRA adapter, greedy-decodes, scores by keyword.
Usage: python eval_hf.py [adapter_dir]
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = os.environ.get("QLORA_BASE", "Qwen/Qwen2.5-3B-Instruct")
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else None

HELDOUT_DOMAIN = [
    ("Слушай, а Sapar — это вообще про что? Коротко.", ["3", "кешбэк", "поезд"]),
    ("Напомни ставку по ипотеке Shanyrak, пожалуйста.", ["7"]),
    ("Ты данные мои в облако отправляешь?", ["офлайн", "устройств", "не "]),
    ("Под каким именем ты работаешь?", ["qiyas", "edge"]),
]
GENERAL = [
    ("Столица Японии?", ["токио"]),
    ("Сколько будет 17 + 25?", ["42"]),
    ("Кто написал роман «Война и мир»?", ["толст"]),
    ("Переведи на английский слово «собака».", ["dog"]),
    ("Назови химический символ золота.", ["au"]),
]


def load():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"": 0},
    )
    if ADAPTER:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    return model, tok


def gen(model, tok, q, max_new=64):
    msgs = [{"role": "user", "content": q}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, add_special_tokens=False,
              return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen_ids = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def score(ans, needles):
    a = ans.lower()
    return any(n.lower() in a for n in needles)


def report(title, model, tok, prompts):
    print(f"\n===== {title} =====")
    c = 0
    for q, needles in prompts:
        ans = gen(model, tok, q)
        ok = score(ans, needles)
        c += ok
        print(f"Q: {q}\n  [{'OK' if ok else '..'}] {ans[:130]}")
    print(f"  >>> {c}/{len(prompts)}")
    return c, len(prompts)


if __name__ == "__main__":
    tag = f"TUNED ({ADAPTER})" if ADAPTER else "BASE"
    print(f"### MODEL: {tag} ###")
    model, tok = load()
    dc, dn = report("A. DOMAIN (held-out)", model, tok, HELDOUT_DOMAIN)
    gc, gn = report("B. GENERAL (regression)", model, tok, GENERAL)
    print(f"\n=== {tag}: domain {dc}/{dn}, general {gc}/{gn} ===")
