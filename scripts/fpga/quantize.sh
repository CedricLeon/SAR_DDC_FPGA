#!/bin/bash

set -x # Uncomment to debug
set -e # Exit on error
start_time=$(date +%s)

EVALUATE=false
IMAGE_GRAPH=false
FAST_FINETUNE=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --evaluate) EVALUATE=true ;;
        --image-graph) IMAGE_GRAPH=true ;;
        --fast-finetune) FAST_FINETUNE=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
# print the value of EVALUATE and IMAGE_GRAPH
echo "EVALUATE: $EVALUATE"
echo "IMAGE_GRAPH: $IMAGE_GRAPH"
echo "FAST_FINETUNE: $FAST_FINETUNE"

RUN_DIR="DDC_FPGA/logs/train/sar_ddc/hyperprior/multiruns/2026-02-08_12-04-57/9" # ADAM relu lambda = 1000, seed = 1
# RUN_DIR="DDC_FPGA/logs/train/sar_ddc/hyperprior/multiruns/2026-02-06_14-57-22/0" # ADAM relu lambda = 1000, seed = 0
# RUN_DIR="DDC_FPGA/logs/train/sar_ddc/hyperprior/runs/2026-01-16_13-34-45" # ADAM relu lambda = 10

# RUN_DIR="DDC_FPGA/logs/train/sar_ddc/hyperprior/multiruns/2026-01-13_11-15-31/38" # ADAM gdn
# "DDC_FPGA/logs/train/sar_ddc/hyperprior/multiruns/2026-01-13_11-15-31/4/" # random ADAM run after the restructuration of the code
# "DDC_FPGA/logs/train/sar_ddc/hyperprior_dpu/runs/2025-12-15_10-34-01" # test_compressai_original_added_logs
# "DDC_FPGA/logs/train/sar_ddc/hyperprior_dpu/runs/2025-12-01_15-58-26" # test_compressai_original_fixed_LowerBound_in_ResBlocks

# I don't support any other kind of models for now
MODEL_NAME="ResidualScaleHyperpriorDPUWrapper"

FAST_FINETUNE_FLAG=""
if [ "$FAST_FINETUNE" = true ]; then
    FAST_FINETUNE_FLAG="--fast_finetune"
fi

echo "--------------------- INSPECTION ----------------------"
echo "Inspecting the model for DPU compatibility..."
python DDC_FPGA/scripts/fpga/model_quant.py --run_dir ${RUN_DIR} --quant_mode float --inspect --target DPUCZDX8G_ISA1_B4096

echo "-------------------- QUANTIZATION ---------------------"
if [ "$EVALUATE" = true ]; then
    echo "Evaluating the float32 model..."
    python DDC_FPGA/scripts/fpga/model_quant.py --run_dir ${RUN_DIR} --quant_mode float --subset_len 500
fi

echo "Quantizing the model and creating quantize_results/..."
# My "val.h5" dataset contains 7296 patches from "Hamburg". Xilinx recommends 100-100 images, so I use --subset_len to limit the number of images
python DDC_FPGA/scripts/fpga/model_quant.py --run_dir ${RUN_DIR} --quant_mode calib --subset_len 200 ${FAST_FINETUNE_FLAG}

if [ "$EVALUATE" = true ]; then
    echo 'Evaluate the quantized model (this can take a bit)...'
    python DDC_FPGA/scripts/fpga/model_quant.py --run_dir ${RUN_DIR} --quant_mode test --subset_len 200 ${FAST_FINETUNE_FLAG}
fi

echo 'Deploy: generate the .xmodel file to be compiled...'
python DDC_FPGA/scripts/fpga/model_quant.py --run_dir ${RUN_DIR} --quant_mode test --subset_len 1 --batch_size 1 --deploy ${FAST_FINETUNE_FLAG}

echo "-------------------- COMPILATION ---------------------"
echo "Compiling the quantized model for the DPU. See " "quantize_result/${MODEL_NAME}_int.xmodel" " for the output xmodel file..."
vai_c_xir -x quantize_result/${MODEL_NAME}_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json -o ${MODEL_NAME}_pt -n ${MODEL_NAME}_pt

echo "Exporting entropy parameters to Numpy..."
# Fix: Ensure output goes into the compiled directory
python DDC_FPGA/scripts/fpga/export_entropy_params.py --ckpt ${RUN_DIR}/checkpoints/last.ckpt --output ${MODEL_NAME}_pt/entropy_params.npz

if [ "$IMAGE_GRAPH" = true ]; then
    echo "-------------------- GENERATE IMAGE GRAPH ---------------------"
    echo "Generating the image graph for the compiled model. See " "quantize_result/${MODEL_NAME}_graph.svg" " for the output image file..."
    xdputil xmodel quantize_result/${MODEL_NAME}_int.xmodel -s quantize_result/${MODEL_NAME}_graph.svg
fi

echo "-------------------- ORGANIZE OUTPUT ---------------------"
# Copy Config and Rename Directory
python DDC_FPGA/scripts/fpga/organize_output.py --run_dir ${RUN_DIR} --source_dir ${MODEL_NAME}_pt

echo "-------------------- DONE ---------------------"
end_time=$(date +%s)
echo "Total execution time: $(($end_time - $start_time)) seconds"
