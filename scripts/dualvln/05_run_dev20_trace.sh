#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="${HARNESS_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
OUTPUT_PATH="${OUTPUT_PATH:-${HARNESS_ROOT}/artifacts/dualvln/dev20_trace}"
TRACE=1 HARNESS_ROOT="$HARNESS_ROOT" \
  bash "${SCRIPT_DIR}/run_stage1_subset.sh" development_subset "$OUTPUT_PATH"

source "${SCRIPT_DIR}/common.sh"
activate_dualvln
python "${HARNESS_ROOT}/validate_dualvln.py" "$OUTPUT_PATH" \
  --require-trace \
  --output "${OUTPUT_PATH}/validation.json"
python "${HARNESS_ROOT}/aggregate_dualvln.py" "$OUTPUT_PATH"
