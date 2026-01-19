#!/bin/bash

set -x # Uncomment to debug
set -e # Exit on error
start_time=$(date +%s)

EVALUATE=false
IMAGE_GRAPH=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --evaluate) EVALUATE=true ;;
        --image-graph) IMAGE_GRAPH=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
# print the value of EVALUATE and IMAGE_GRAPH
echo "EVALUATE: $EVALUATE"
echo "IMAGE_GRAPH: $IMAGE_GRAPH"

RUN_DIR="DDC_FPGA/logs/train/sar_ddc/hyperprior/runs/2026-01-16_13-34-45" # ADAM relu
# "DDC_FPGA/logs/train/sar_ddc/hyperprior/multiruns/2026-01-13_11-15-31/4/" # random ADAM run after the restructuration of the code
# "DDC_FPGA/logs/train/sar_ddc/hyperprior_dpu/runs/2025-12-15_10-34-01" # test_compressai_original_added_logs
# "DDC_FPGA/logs/train/sar_ddc/hyperprior_dpu/runs/2025-12-01_15-58-26" # test_compressai_original_fixed_LowerBound_in_ResBlocks
# "ResAE-relu_42_merlinʎ100_lr5e-05_b12_no-out-pad"

# if RUN_DIR starts with "ResAE", then we use small model
if [[ ${RUN_DIR} == *simple_ae* ]]; then
    MODEL_NAME="ResidualSimpleAE"
# else if ResSHyperAE
elif [[ ${RUN_DIR} == *hyperprior* ]]; then
    MODEL_NAME="ResidualScaleHyperpriorDPUWrapper"
else # crash
    echo "Unknown model type in RUN_DIR: ${RUN_DIR}"
    exit 1
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
python DDC_FPGA/scripts/fpga/model_quant.py --run_dir ${RUN_DIR} --quant_mode calib --subset_len 200

if [ "$EVALUATE" = true ]; then
    echo 'Evaluate the quantized model (this can take a bit)...'
    python DDC_FPGA/scripts/fpga/model_quant.py --run_dir ${RUN_DIR} --quant_mode test --subset_len 200
fi

echo 'Deploy: generate the .xmodel file to be compiled...'
python DDC_FPGA/scripts/fpga/model_quant.py --run_dir ${RUN_DIR} --quant_mode test --subset_len 1 --batch_size 1 --deploy

echo "-------------------- COMPILATION ---------------------"
echo "Compiling the quantized model for the DPU. See " "quantize_result/${MODEL_NAME}_int.xmodel" " for the output xmodel file..."
vai_c_xir -x quantize_result/${MODEL_NAME}_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json -o ${MODEL_NAME}_pt -n ${MODEL_NAME}_pt

if [ "$IMAGE_GRAPH" = true ]; then
    echo "-------------------- GENERATE IMAGE GRAPH ---------------------"
    echo "Generating the image graph for the compiled model. See " "quantize_result/${MODEL_NAME}_graph.svg" " for the output image file..."
    xdputil xmodel quantize_result/${MODEL_NAME}_int.xmodel -s quantize_result/${MODEL_NAME}_graph.svg
fi

echo "-------------------- DONE ---------------------"
end_time=$(date +%s)
echo "Total execution time: $(($end_time - $start_time)) seconds"
