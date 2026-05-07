#!/bin/bash
# Run the image-quality evaluation in LOCAL mode (crop only annotated regions).
#
# Arguments:
#   $1 -- TEXT_PRED_DIR
#   $2 -- TABLE_PRED_DIR
# Optional (environment variables):
#   BENCH_DIR   -- root of released VDE-Bench data (default: ./data)
#   OUTPUT      -- output JSON path (default: ./iq_results_local.json)
#   DEVICE      -- cuda / cpu (default: cuda)

set -euo pipefail

TEXT_PRED_DIR="${1:?Missing TEXT_PRED_DIR (arg 1)}"
TABLE_PRED_DIR="${2:?Missing TABLE_PRED_DIR (arg 2)}"

BENCH_DIR="${BENCH_DIR:-./data}"
TEXT_INFO="${TEXT_INFO:-${BENCH_DIR}/text_info_all.json}"
TABLE_INFO="${TABLE_INFO:-${BENCH_DIR}/merged_with_label_output.json}"
TEXT_GT_DIR="${TEXT_GT_DIR:-${BENCH_DIR}/text_output_all}"
TABLE_GT_DIR="${TABLE_GT_DIR:-${BENCH_DIR}/table_output}"
OUTPUT="${OUTPUT:-./iq_results_local.json}"
DEVICE="${DEVICE:-cuda}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/evaluate.py" \
    --groups \
        text  "${TEXT_INFO}"  "${TEXT_GT_DIR}"  "${TEXT_PRED_DIR}" \
        table "${TABLE_INFO}" "${TABLE_GT_DIR}" "${TABLE_PRED_DIR}" \
    --padding 0 \
    --device "${DEVICE}" \
    --output_json "${OUTPUT}"
