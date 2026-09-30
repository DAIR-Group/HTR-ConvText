#!/usr/bin/env bash
set -euo pipefail

# SCAM-A data and split files are prepared in the sibling HTR-ConvText_2 repo.
SCAM_ROOT="/home/namhoai/WorkSpace/Research/HTR/HTR-ConvText_2/data/SCAM-A"

python3 train.py \
  --dataset scam_a \
  --exp-name "htr-convtext-scam-a" \
  --wandb-project scam-a \
  --tcm-enable \
  --num-workers 4 \
  --max-lr 5e-4 \
  --warm-up-iter 300 \
  --weight-decay 0.05 \
  --train-bs 16 \
  --val-bs 8 \
  --max-span-length 8 \
  --mask-ratio 0.4 \
  --attn-mask-ratio 0.1 \
  --img-size 512 64 \
  --proj 8 \
  --dila-ero-max-kernel 2 \
  --dila-ero-iter 1 \
  --proba 0.5 \
  --alpha 1 \
  --total-iter 20000 \
  --data-path "${SCAM_ROOT}/" \
  --train-data-list "${SCAM_ROOT}/train.ln" \
  --val-data-list "${SCAM_ROOT}/val.ln" \
  --test-data-list "${SCAM_ROOT}/test.ln" \
  --nb-cls 88
