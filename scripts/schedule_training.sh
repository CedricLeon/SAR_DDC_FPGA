#!/bin/bash

# Schedule execution of many training runs
# Run from root folder with: bash scripts/schedule_training.sh


# # ---- with MERLIN normalization -----
# python src/eval.py ckpt_path=logs/train/sar_ddc/hyperprior/multiruns/2025-06-21_12-42-02/0/checkpoints/last.ckpt
# python src/eval.py ckpt_path=logs/train/sar_ddc/hyperprior/multiruns/2025-06-21_12-42-02/1/checkpoints/last.ckpt
# python src/eval.py ckpt_path=logs/train/sar_ddc/hyperprior/multiruns/2025-06-21_12-42-02/2/checkpoints/last.ckpt
# python src/eval.py ckpt_path=logs/train/sar_ddc/hyperprior/multiruns/2025-06-21_12-42-02/3/checkpoints/last.ckpt

# # ---- with 0 (min/max) normalization -----
# python src/eval.py ckpt_path=logs/train/sar_ddc/hyperprior/multiruns/2025-06-20_18-34-17/0/checkpoints/last.ckpt
# python src/eval.py ckpt_path=logs/train/sar_ddc/hyperprior/multiruns/2025-06-20_18-34-17/1/checkpoints/last.ckpt
# python src/eval.py ckpt_path=logs/train/sar_ddc/hyperprior/multiruns/2025-06-20_18-34-17/2/checkpoints/last.ckpt
# python src/eval.py ckpt_path=logs/train/sar_ddc/hyperprior/multiruns/2025-06-20_18-34-17/3/checkpoints/last.ckpt
# python src/eval.py ckpt_path=logs/train/sar_ddc/hyperprior/multiruns/2025-06-20_18-34-17/4/checkpoints/last.ckpt

# # Create a new folder and copy all "inference_results_..." folders from each checkpoint directory in it
# mkdir tmp_to_download
# for i in {0..4}
# do
#  echo $i
#  cp -r logs/train/sar_ddc/hyperprior/multiruns/2025-06-20_18-34-17/$i/inference_results_* tmp_to_download/
#  cp -r logs/train/sar_ddc/hyperprior/multiruns/2025-06-21_12-42-02/$i/inference_results_* tmp_to_download/
# done
# cp -r logs/train/sar_ddc/hyperprior/multiruns/2025-06-20_18-34-17/4/inference_results_* tmp_to_download/

# # Then for each folder in tmp_to_download, copy directly in tmp_to_download the files starting with "output_reflectivity_...", "output_real_...", "input_reflectivity_..."
# for folder in tmp_to_download/inference_results_*
# do
#   echo "Processing $folder"
#   cp "$folder"/output_reflectivity_*.png tmp_to_download/
#   cp "$folder"/output_real_*.png tmp_to_download/
#   cp "$folder"/input_reflectivity_*.png tmp_to_download/
# done
# # zip the whole tmp_to_download folder
# zip -r tmp_to_download.zip tmp_to_download