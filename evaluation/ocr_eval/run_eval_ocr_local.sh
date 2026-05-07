#!/bin/bash
# Run VDE-Bench OCR evaluation in LOCAL mode (only evaluate blocks that
# overlap with human-annotated edit regions).
#
# Usage:
#     bash run_eval_ocr_local.sh <MODEL_NAME> <TEXT_PRED_DIR> <TABLE_PRED_DIR> \
#                                [TEXT_GT_DIR] [TABLE_GT_DIR] \
#                                [TEXT_INFO]   [TABLE_INFO] [OUTPUT_DIR]

set -euo pipefail

MODEL_NAME="${1:?Missing MODEL_NAME (arg 1)}"
TEXT_PRED_DIR="${2:?Missing TEXT_PRED_DIR (arg 2)}"
TABLE_PRED_DIR="${3:?Missing TABLE_PRED_DIR (arg 3)}"

BENCH_DIR="${BENCH_DIR:-./data}"
TEXT_GT_DIR="${4:-${TEXT_GT_DIR:-${BENCH_DIR}/text_output_all_ocr}}"
TABLE_GT_DIR="${5:-${TABLE_GT_DIR:-${BENCH_DIR}/table_ocr}}"
TEXT_INFO="${6:-${TEXT_INFO:-${BENCH_DIR}/text_info_all.json}}"
TABLE_INFO="${7:-${TABLE_INFO:-${BENCH_DIR}/merged_with_label_output.json}}"
OUTPUT_DIR="${8:-${OUTPUT_DIR:-./results}}"
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_eval_local_results.json"

mkdir -p "${OUTPUT_DIR}"

echo "Model:       ${MODEL_NAME}"
echo "Mode:        local"
echo "Text  Pred:  ${TEXT_PRED_DIR}"
echo "Text  GT:    ${TEXT_GT_DIR}"
echo "Text  Info:  ${TEXT_INFO}"
echo "Table Pred:  ${TABLE_PRED_DIR}"
echo "Table GT:    ${TABLE_GT_DIR}"
echo "Table Info:  ${TABLE_INFO}"
echo "Output:      ${OUTPUT_FILE}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/eval_unified.py" \
    --mode local \
    --groups \
        text  "${TEXT_INFO}"  "${TEXT_GT_DIR}"  "${TEXT_PRED_DIR}" \
        table "${TABLE_INFO}" "${TABLE_GT_DIR}" "${TABLE_PRED_DIR}" \
    --iou_threshold 0.1 \
    --output "${OUTPUT_FILE}"
