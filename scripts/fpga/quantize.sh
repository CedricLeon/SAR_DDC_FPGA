#!/bin/bash

# set -x # Uncomment to debug
set -e # Exit on error
start_time=$(date +%s)

INSPECT=false
EVALUATE=false
IMAGE_GRAPH=false
FAST_FINETUNE=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --inspect) INSPECT=true ;;
        --evaluate) EVALUATE=true ;;
        --image-graph) IMAGE_GRAPH=true ;;
        --fast-finetune) FAST_FINETUNE=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


ARCH_JSON="DDC_FPGA/scripts/fpga/DPU_archs/ZCU102_DPUCZDX8G_ISA1_B4096_arch.json"
# ARCH_JSON="DDC_FPGA/scripts/fpga/DPU_archs/KP-Labs_Leopard_DPUCZDX8G_ISA1_B1024_arch.json"

if [[ "$ARCH_JSON" == *"ZCU102"* ]]; then
    TARGET="DPUCZDX8G_ISA1_B4096"
elif [[ "$ARCH_JSON" == *"Leopard"* ]]; then
    TARGET="DPUCZDX8G_ISA1_B1024"
else
    echo "Unknown architecture JSON. Please update the script to set the correct TARGET."
    exit 1
fi

RUN_DIR="DDC_FPGA/logs/train/sar_ddc/hyperprior/multiruns/2026-02-11_15-50-39/6" # ADAM relu lambda = 200,  seed = 1
# RUN_DIR="DDC_FPGA/logs/train/sar_ddc/hyperprior/multiruns/2026-02-08_12-04-57/9" # ADAM relu lambda = 1000, seed = 1
# RUN_DIR="DDC_FPGA/logs/train/sar_ddc/hyperprior/multiruns/2026-02-06_14-57-22/0" # ADAM relu lambda = 1000, seed = 0
# RUN_DIR="DDC_FPGA/logs/train/sar_ddc/hyperprior/runs/2026-01-16_13-34-45" # ADAM relu lambda = 10

# I don't support any other kind of models for now
MODEL_NAME="ResidualScaleHyperpriorDPUWrapper"

FAST_FINETUNE_FLAG=""
if [ "$FAST_FINETUNE" = true ]; then
    FAST_FINETUNE_FLAG="--fast_finetune"
fi

echo""
echo "BASH PARAMETERS: INSPECT: $INSPECT, EVALUATE: $EVALUATE, IMAGE_GRAPH: $IMAGE_GRAPH, FAST_FINETUNE: $FAST_FINETUNE"
echo "MODEL: RUN_DIR: $RUN_DIR, MODEL_NAME: $MODEL_NAME"
echo "DPU: ARCH_JSON: $ARCH_JSON, TARGET: $TARGET"
echo ""

if [ "$EVALUATE" = true ]; then
    echo "--------------------- INSPECTION ----------------------"
    echo "Inspecting the model for DPU compatibility..."
    python DDC_FPGA/scripts/fpga/model_quant.py --run_dir ${RUN_DIR} --quant_mode float --inspect --target ${TARGET}
fi

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

vai_c_xir -x quantize_result/${MODEL_NAME}_int.xmodel -a ${ARCH_JSON} -o ${MODEL_NAME}_pt -n ${MODEL_NAME}_pt


if [ "$IMAGE_GRAPH" = true ]; then
    echo "-------------------- GENERATE IMAGE GRAPH ---------------------"
    echo "Generating the image graph for the compiled model. See " "quantize_result/${MODEL_NAME}_graph.svg" " for the output image file..."
    xdputil xmodel quantize_result/${MODEL_NAME}_int.xmodel -s quantize_result/${MODEL_NAME}_graph.svg
fi

echo "-------------------- EXPORT ENTROPY PARAMETERS ---------------------"
python DDC_FPGA/scripts/fpga/export_entropy_params.py --ckpt ${RUN_DIR}/checkpoints/last.ckpt --output ${MODEL_NAME}_pt/entropy_params.npz

echo "-------------------- ORGANIZE OUTPUT ---------------------"
# Copy all inference components to the model folder for a self-contained deployment
cp DDC_FPGA/scripts/fpga/inference_hybrid.py ${MODEL_NAME}_pt/
cp DDC_FPGA/scripts/fpga/inference_utils.py ${MODEL_NAME}_pt/
cp DDC_FPGA/scripts/fpga/entropy_models_inference.py ${MODEL_NAME}_pt/
# Copy Config and Rename Directory
python DDC_FPGA/scripts/fpga/organize_output.py --run_dir ${RUN_DIR} --source_dir ${MODEL_NAME}_pt

echo "-------------------- DONE ---------------------"
end_time=$(date +%s)
echo "Total execution time: $(($end_time - $start_time)) seconds"
