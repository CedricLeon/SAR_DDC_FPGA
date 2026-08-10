#!/bin/bash
# Run as: user@bart:~/dev/Vitis-AI/DDC_FPGA$ ./scripts/vitis_ai/setup_container.sh

# 1. Determine the Vitis-AI root directory
# Assuming this script is located at DDC_FPGA/scripts/vitis_ai/setup_container.sh
# We need to go up 3 levels to reach Vitis-AI root where docker_run.sh is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Adjust the level of '..' based on where you actually put this script relative to Vitis-AI root
VITIS_AI_ROOT="$(realpath "$SCRIPT_DIR/../../../")"
DOCKER_SCRIPT="$VITIS_AI_ROOT/docker_run.sh"

if [ ! -f "$DOCKER_SCRIPT" ]; then
    echo "Error: Could not find docker_run.sh at $VITIS_AI_ROOT"
    echo "Please check the path relative to this script."
    exit 1
fi

# 2. Go to Vitis-AI root so docker_run.sh works correctly (it uses 'pwd' for mounting)
cd "$VITIS_AI_ROOT"

# 3. Create a temporary setup script that will run INSIDE the container
# We place this script in the root directory so it is mounted to /workspace
INSIDE_SCRIPT_NAME=".auto_setup_env.sh"

cat << 'EOF' > "$INSIDE_SCRIPT_NAME"
#!/bin/bash
# This runs inside the container

# Source bashrc to get conda wrapper if available
source ~/.bashrc
# Fallback if conda is not in path yet (location in standard Vitis-AI container)
if ! command -v conda &> /dev/null; then
    if [ -f /opt/vitis_ai/conda/etc/profile.d/conda.sh ]; then
        source /opt/vitis_ai/conda/etc/profile.d/conda.sh
    fi
fi

echo "Activating vitis-ai-pytorch environment..."
conda activate vitis-ai-pytorch

echo "Installing dependencies..."
# Install only what quantisation/compilation needs, on top of the container's stock
# torch 1.13.1+cu117. Do NOT add `lightning` here: pip resolves it by bumping torch to 2.4.x,
# which is ABI-incompatible with the vaic/vart wheels (built against torch 1.13) and fails at
# import with an OSError. compressai is pinned to 1.2.8 (parameter names changed after 1.2.6,
# which otherwise breaks checkpoint loading with "unexpected keys").
pip install h5py omegaconf compressai torchmetrics

echo "Exporting required library..."
# The stock container libstdc++ is too old (missing GLIBCXX_3.4.29) for our ops; preload conda's
# newer libstdc++ so the symbol resolves.
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6:$LD_PRELOAD

echo "============================================"
echo "      Environment Setup Complete!"
echo "============================================"

# Spawn a new shell to keep the container open with the activated environment
exec bash
EOF

chmod +x "$INSIDE_SCRIPT_NAME"

# 4. Run the container using the official script
# We pass our temporary script as the command to run inside
IMAGE="xilinx/vitis-ai-pytorch-gpu:3.5.0.001-1eed93cde"

echo "Starting Docker container..."
echo "Running: $DOCKER_SCRIPT $IMAGE /workspace/$INSIDE_SCRIPT_NAME"

./docker_run.sh "$IMAGE" /workspace/"$INSIDE_SCRIPT_NAME"
