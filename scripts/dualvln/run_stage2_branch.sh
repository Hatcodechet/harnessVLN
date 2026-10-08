#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
[[ $# -eq 2 ]] || die "Usage: $0 <commit|replan> <output-path>"
MODE="$1"
OUTPUT_PATH="$2"
[[ "$MODE" == "commit" || "$MODE" == "replan" ]] || die "Invalid mode: $MODE"
HARNESS_ROOT="${HARNESS_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
SPEC="${STAGE2_SPEC:-${HARNESS_ROOT}/artifacts/dualvln/stage2/candidates.json}"
require_path "$SPEC" "Stage 2 candidate spec"
grep -q 'DUALVLN_BRANCH_SPEC' \
  "${INTERNNAV_ROOT}/internnav/habitat_extensions/vln/habitat_vln_evaluator.py" \
  || die "Stage 2 hook is not applied; run 06_prepare_stage2.sh first"

export DUALVLN_BRANCH_SPEC="$SPEC"
export DUALVLN_BRANCH_MODE="$MODE"
TRACE=1 MANIFEST="$SPEC" HARNESS_ROOT="$HARNESS_ROOT" \
  bash "${SCRIPT_DIR}/run_stage1_subset.sh" stage2_pilot_subset "$OUTPUT_PATH"

activate_dualvln
python "${HARNESS_ROOT}/validate_dualvln.py" "$OUTPUT_PATH" \
  --require-trace --output "${OUTPUT_PATH}/validation.json"
