#!/bin/bash

# Schedule execution of many runs evaluations
# Run from root folder with: bash scripts/schedule_evaluate.sh

# ADAM lambda = 1: 2025-09-18_13-05-43
# ADAM lambda = 10: 2025-09-17_20-38-04
# ADAM lambda = 50: 2025-09-17_15-48-07
# ADAM lambda = 100: 2025-09-17_14-32-17
# ADAM lambda = 200: logs/train/sar_ddc/hyperprior/multiruns/2025-09-18_18-06-51/2"
# MERLIN: 2025-09-16_13-49-03 (.../sar_ddc/merlin/...)

STRIDE = 256

echo "Evaluating LAMBDA=1 model..."
python src/evaluate.py ckpt_path=logs/train/sar_ddc/hyperprior/runs/2025-09-18_13-05-43/checkpoints/last.ckpt stride=$STRIDE

echo "Evaluating LAMBDA=10 model..."
python src/evaluate.py ckpt_path=logs/train/sar_ddc/hyperprior/runs/2025-09-17_20-38-04/checkpoints/last.ckpt stride=$STRIDE

echo "Evaluating LAMBDA=50 model..."
python src/evaluate.py ckpt_path=logs/train/sar_ddc/hyperprior/runs/2025-09-17_15-48-07/checkpoints/last.ckpt stride=$STRIDE

echo "Evaluating LAMBDA=100 model..."
python src/evaluate.py ckpt_path=logs/train/sar_ddc/hyperprior/runs/2025-09-17_14-32-17/checkpoints/last.ckpt stride=$STRIDE

echo "Evaluating LAMBDA=200 model..."
python src/evaluate.py ckpt_path=logs/train/sar_ddc/hyperprior/multiruns/2025-09-18_18-06-51/2/checkpoints/last.ckpt stride=$STRIDE