#!/bin/bash

set -x # Uncomment to debug
set -e # Exit on error
# Old (July 2025) Checkpoints names/dates
# normal small: 2025-06-13_08-21-27/"
# safe: 2025-06-13_10-29-49
# small safe: 2025-06-13_10-55-06
# no entropy_bottleneck: 2025-06-17_11-40-41
# small mock entropy: "2025-06-17_14-12-45"
# DATA_DIR="DDC_FPGA/data/processed_hdf5/TSX_preprocessed_spatial_splits_5_256x256/"
# MODEL_DIR="DDC_FPGA/logs/train/sar_ddc/simple_ae/multiruns/2025-11-16_19-34-55/3/"
# SMALL="--small"

RUN_DIR="ResAE-relu_42_merlinʎ100_lr5e-05_b12_no-out-pad"

# if RUN_DIR starts with "ResAE", then we use small model
if [[ ${RUN_DIR} == ResAE* ]]; then
    MODEL_NAME="ResidualSimpleAE"
# else if ResSHyperAE
elif [[ ${RUN_DIR} == ResSHyp* ]]; then
    MODEL_NAME="ResidualScaleHyperprior"
else # crash
    echo "Unknown model type in RUN_DIR: ${RUN_DIR}"
    exit 1
fi

echo "--------------------- INSPECTION ----------------------"
echo "Inspecting the model for DPU compatibility..."
python DDC_FPGA/scripts/fpga/model_quant.py --run_dir ${RUN_DIR} --quant_mode float --inspect --target DPUCZDX8G_ISA1_B4096

echo "-------------------- QUANTIZATION ---------------------"
echo "Evaluating the float32 model..."
python DDC_FPGA/scripts/fpga/model_quant.py --run_dir ${RUN_DIR} --quant_mode float --subset_len 500


echo "Quantizing the model and creating quantize_results/..."
# My "val.h5" dataset contains 7296 patches from "Hamburg". Xilinx recommends 100-100 images, so I use --subset_len to limit the number of images
python DDC_FPGA/scripts/fpga/model_quant.py --run_dir ${RUN_DIR} --quant_mode calib --subset_len 200

echo 'Evaluate the quantized model (this can take a bit)...'
python DDC_FPGA/scripts/fpga/model_quant.py --run_dir ${RUN_DIR} --quant_mode test --subset_len 200

echo 'Deploy: generate the .xmodel file to be compiled...'
python DDC_FPGA/scripts/fpga/model_quant.py --run_dir ${RUN_DIR} --quant_mode test --subset_len 1 --batch_size 1 --deploy

echo "-------------------- COMPILATION ---------------------"
echo "Compiling the quantized model for the DPU. See " "quantize_result/${MODEL_NAME}_int.xmodel" " for the output xmodel file..."
vai_c_xir -x quantize_result/${MODEL_NAME}_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json -o ${MODEL_NAME}_pt -n ${MODEL_NAME}_pt

echo "-------------------- GENERATE IMAGE GRAPH ---------------------"
echo "Generating the image graph for the compiled model. See " "quantize_result/${MODEL_NAME}_graph.svg" " for the output image file..."
xdputil xmodel quantize_result/${MODEL_NAME}_int.xmodel -s quantize_result/${MODEL_NAME}_graph.svg

echo "-------------------- DONE ---------------------"
echo "Hope you did not see too much red ..."
