#!/bin/bash
# Run the image-quality evaluation in GLOBAL mode (full image comparisons).
#
# Arguments:
#   $1 -- TEXT_PRED_DIR
#   $2 -- TABLE_PRED_DIR

set -euo pipefail

TEXT_PRED_DIR="${1:?Missing TEXT_PRED_DIR (arg 1)}"
TABLE_PRED_DIR="${2:?Missing TABLE_PRED_DIR (arg 2)}"

BENCH_DIR="${BENCH_DIR:-./data}"
TEXT_GT_DIR="${TEXT_GT_DIR:-${BENCH_DIR}/text_output_all}"
TABLE_GT_DIR="${TABLE_GT_DIR:-${BENCH_DIR}/table_output}"
OUTPUT="${OUTPUT:-./iq_results_global.json}"
DEVICE="${DEVICE:-cuda}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/evaluate.py" \
    --groups \
        text  none "${TEXT_GT_DIR}"  "${TEXT_PRED_DIR}" \
        table none "${TABLE_GT_DIR}" "${TABLE_PRED_DIR}" \
    --padding 0 \
    --device "${DEVICE}" \
    --output_json "${OUTPUT}"
