#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="${HARNESS_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
OUTPUT_PATH="${OUTPUT_PATH:-${HARNESS_ROOT}/artifacts/dualvln/smoke3_native}"
TRACE=0 HARNESS_ROOT="$HARNESS_ROOT" \
  bash "${SCRIPT_DIR}/run_stage1_subset.sh" smoke_subset "$OUTPUT_PATH"
