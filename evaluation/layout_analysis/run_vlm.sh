#!/bin/bash
# Host a vision LLM as an OpenAI-compatible vLLM server for layout extraction.
#
# Customise MODEL / PORT / TP_SIZE / GPU_MEM to fit your hardware.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
SERVED_NAME="${SERVED_NAME:-vlm}"
PORT="${PORT:-63332}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"

vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --seed 42
