#!/bin/bash
# Run VDE-Bench OCR evaluation in GLOBAL mode (compare all blocks across the
# full image, not only those inside annotated regions).
#
# Usage:
#     bash run_eval_ocr.sh <MODEL_NAME> <TEXT_PRED_DIR> <TABLE_PRED_DIR> \
#                         [TEXT_GT_DIR] [TABLE_GT_DIR] [OUTPUT_DIR]
#
# Environment variables (used as defaults when the corresponding positional
# argument is not supplied):
#   BENCH_DIR        -- root of the released VDE-Bench data (default: ./data)
#   TEXT_GT_DIR      -- default: "${BENCH_DIR}/text_output_all_ocr"
#   TABLE_GT_DIR     -- default: "${BENCH_DIR}/table_ocr"
#   OUTPUT_DIR       -- default: "./results"

set -euo pipefail

MODEL_NAME="${1:?Missing MODEL_NAME (arg 1)}"
TEXT_PRED_DIR="${2:?Missing TEXT_PRED_DIR (arg 2)}"
TABLE_PRED_DIR="${3:?Missing TABLE_PRED_DIR (arg 3)}"

BENCH_DIR="${BENCH_DIR:-./data}"
TEXT_GT_DIR="${4:-${TEXT_GT_DIR:-${BENCH_DIR}/text_output_all_ocr}}"
TABLE_GT_DIR="${5:-${TABLE_GT_DIR:-${BENCH_DIR}/table_ocr}}"
OUTPUT_DIR="${6:-${OUTPUT_DIR:-./results}}"
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_eval_results.json"

mkdir -p "${OUTPUT_DIR}"

echo "Model:       ${MODEL_NAME}"
echo "Text  Pred:  ${TEXT_PRED_DIR}"
echo "Text  GT:    ${TEXT_GT_DIR}"
echo "Table Pred:  ${TABLE_PRED_DIR}"
echo "Table GT:    ${TABLE_GT_DIR}"
echo "Output:      ${OUTPUT_FILE}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/eval_unified.py" \
    --groups \
        text  "${TEXT_GT_DIR}"  "${TEXT_PRED_DIR}" \
        table "${TABLE_GT_DIR}" "${TABLE_PRED_DIR}" \
    --iou_threshold 0.1 \
    --output "${OUTPUT_FILE}"
