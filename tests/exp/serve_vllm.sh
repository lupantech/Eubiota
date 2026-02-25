#!/bin/bash

# ===========================================================================
# Script: serve_vllm.sh
# Description: Launch model using vLLM
# ===========================================================================

MODEL="Eubiota/eubiota-planner-8b"
GPU="0"
PORT=8000
TP=1
SERVED_MODEL_NAME="Eubiota-8b"

VENV_ACTIVATE="source .venv/bin/activate"

echo "Launching model: $MODEL"
echo "  Served Model Name: $SERVED_MODEL_NAME"
echo "  Port: $PORT"
echo "  GPU: $GPU"
echo "  Tensor Parallel Size: $TP"

CMD_START="
    $VENV_ACTIVATE;
    export CUDA_VISIBLE_DEVICES=$GPU;
    echo '--- Starting $MODEL on port $PORT with TP=$TP ---';
    echo 'CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES';
    echo 'Current virtual env: \$(python -c \"import sys; print(sys.prefix)\")';
    vllm serve \"$MODEL\" \
        --served-model-name \"$SERVED_MODEL_NAME\" \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $TP
"

echo "Running command:"
eval "$CMD_START"

