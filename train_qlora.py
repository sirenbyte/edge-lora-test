"""QLoRA SFT for Qwen2.5 on a single 8GB CUDA GPU (RTX 3060 Ti / WSL2).

Robust, self-contained: manual chat-template tokenization with prompt
masking (loss only on assistant tokens), 4-bit nf4 base, PEFT LoRA,
transformers Trainer. Avoids version-fragile TRL APIs.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(tok, records, max_len):
    rows = []
    for r in records:
        msgs = r["messages"]
        full_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)
        prompt_text = tok.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tok(full_text, add_special_tokens=False)["input_ids"]
        prompt = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full = full[:max_len]
        plen = min(len(prompt), len(full))
        labels = [-100] * plen + full[plen:]
        rows.append({"input_ids": full, "labels": labels})
    return Dataset.from_list(rows)


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        m = max(len(x["input_ids"]) for x in batch)
        ids, labs, att = [], [], []
        for x in batch:
            n = m - len(x["input_ids"])
            ids.append(x["input_ids"] + [self.pad_id] * n)
            labs.append(x["labels"] + [-100] * n)
            att.append([1] * len(x["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labs),
            "attention_mask": torch.tensor(att),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="data2")
    ap.add_argument("--out", default="adapters_qlora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--target", default="all-linear",
                    help="LoRA target modules: 'all-linear' or comma list")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    targets = ("all-linear" if args.target == "all-linear"
               else [t.strip() for t in args.target.split(",")])
    lora = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    data_dir = Path(args.data)
    train_ds = build_dataset(tok, load_jsonl(data_dir / "train.jsonl"),
                             args.max_len)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="paged_adamw_8bit",
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds,
        data_collator=PadCollator(tok.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"SAVED_ADAPTER {args.out}")


if __name__ == "__main__":
    main()
