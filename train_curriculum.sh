#!/bin/bash
# Curriculum-in-decay recipe (MiniCPM Ultra-FineWeb / WSD decay-data).
# mlx-lm shuffles data each epoch, so we cannot put the clean data "last" by
# file order. Instead: main train (full data) -> RESUME the adapter and run a
# short, low-LR polish on the clean/important subset only. The cleanest data is
# thus seen at the lowest LR = the decay phase. Generalizes to any adapter
# (core / creative / hobby / domain): swap the two configs + the polish subset.
set -e
cd /Users/abzaltuganbay/projects/edge-lora-test || exit 1
VENV=.venv/bin

echo "== Phase A: main train (full data) =="
$VENV/mlx_lm.lora --config lora_qwen4b_v2.yaml

echo "== Phase B: build clean/important subset =="
$VENV/python make_data_polish.py

echo "== Phase B: polish (resume adapter, low LR, clean subset) =="
$VENV/mlx_lm.lora --config lora_qwen4b_polish.yaml

echo "== done -> adapters_qwen4b_v2_curr (curriculum-polished) =="
