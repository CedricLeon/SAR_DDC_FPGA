#!/usr/bin/env bash
# Retrain SH @ lr=5e-4 and ResSH @ lr=1e-4 (their best lrs from the lr-sweep),
# refresh the W&B CSV, and re-evaluate both on the FPGA.
# FP/ResFP are untouched. The hardware benchmark (F6) is lr-independent, not re-run.
#
# Run from the project root in the DDC_FPGA conda env. To capture all output:
#   bash scripts/schedule_training.sh 2>&1 | tee schedule.log
# set -e: stop the chain if any step fails (deploy depends on training output).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 1. Train (6 seeds × 10 λ each). DPU-relevant fields forced on the CLI for safety.
python src/train.py -m experiment=ADAM-SHyp seed=0,1,2,3,4,5 model.criterion.lmbda=1,2,5,10,20,50,100,200,500,1000 model.net_optimizer.lr=5e-4 model.net.no_residual_blocks=True  model.net.no_output_padding=True model.net.activation=relu tags="[TSX_SSC,hyperprior,lr_5e-4]"
python src/train.py -m experiment=ADAM-ResSHyp seed=0,1,2,3,4,5 model.criterion.lmbda=1,2,5,10,20,50,100,200,500,1000 model.net_optimizer.lr=1e-4 model.net.no_residual_blocks=False model.net.no_output_padding=True model.net.activation=relu tags="[TSX_SSC,hyperprior,lr_1e-4]"

# 2. Refresh the FP32-quality CSV for the notebooks.
python notebooks/fetch_wandb_runs.py

# 3. Archive the old lr=5e-5 SH/ResSH compiled models BEFORE deploying.
#    Compiled names (<model>-<act>_s<seed>_L<λ>_pt) don't encode lr, so the new
#    models reuse the same folders — and batch_deploy would otherwise skip-compile
#    and re-evaluate the stale lr=5e-5 model. Reversible mv (FP/ResFP untouched).
mkdir -p results/fpga/_archive_lr5e5
mv results/fpga/compiled_models/SHyp-relu_s*_L*_pt results/fpga/compiled_models/ResSHyp-relu_s*_L*_pt results/fpga/_archive_lr5e5/

# 4. Deploy both archs to the FPGA (compile + transfer + infer + fetch).
python scripts/fpga/deploy/batch_deploy.py --config scripts/fpga/deploy/batch_deploy_configs/SHyp_lr5e4.yaml   --tag SHyp_lr_5e-4
python scripts/fpga/deploy/batch_deploy.py --config scripts/fpga/deploy/batch_deploy_configs/ResSHyp_lr1e4.yaml --tag ResSHyp_lr_1e-4
