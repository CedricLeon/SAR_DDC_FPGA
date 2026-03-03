#!/bin/bash
# Starts vai_container in detached mode for use by deploy.py.
#
# Unlike setup_container.sh (which blocks for interactive use), this script starts
# the container with --detach and `sleep infinity` as PID 1, so it stays alive in
# the background and can be targeted with `docker exec` from deploy.py.
#
# Can be run from any directory:
#   ./scripts/vitis-ai-automation/start_container_bg.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VITIS_AI_ROOT="$(realpath "$SCRIPT_DIR/../../../")"

IMAGE="xilinx/vitis-ai-pytorch-gpu:3.5.0.001-1eed93cde"
CONTAINER_NAME="vai_container"

echo "Vitis-AI root : $VITIS_AI_ROOT"
echo "Image         : $IMAGE"
echo "Container name: $CONTAINER_NAME"

# Ensure .confirm exists so docker_run.sh (and any Xilinx tooling) skips the EULA prompt.
touch "$VITIS_AI_ROOT/.confirm"

echo "Starting container in detached mode..."

docker run \
    --name "$CONTAINER_NAME" \
    --detach \
    --gpus all \
    --network host \
    -v /dev/shm:/dev/shm \
    -v /opt/xilinx/dsa:/opt/xilinx/dsa \
    -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
    -v "$VITIS_AI_ROOT":/vitis_ai_home \
    -v "$VITIS_AI_ROOT":/workspace \
    -w /workspace \
    -e USER="$(whoami)" \
    -e UID="$(id -u)" \
    -e GID="$(id -g)" \
    "$IMAGE" \
    sleep infinity

echo "Container '$CONTAINER_NAME' started in background."
echo "Run 'docker exec -it $CONTAINER_NAME bash' to get an interactive shell."
