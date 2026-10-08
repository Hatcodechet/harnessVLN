#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
HARNESS_ROOT="${HARNESS_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
SPEC="${STAGE2_SPEC:-${HARNESS_ROOT}/artifacts/dualvln/stage2/candidates.json}"
COMMIT_RUN="${COMMIT_RUN:-${HARNESS_ROOT}/artifacts/dualvln/stage2/commit_seed42}"
REPLAN_RUN="${REPLAN_RUN:-${HARNESS_ROOT}/artifacts/dualvln/stage2/replan_seed42}"
OUTPUT_DIR="${OUTPUT_DIR:-${HARNESS_ROOT}/artifacts/dualvln/stage2/analysis_seed42}"
activate_dualvln
python "${HARNESS_ROOT}/analyze_stage2_pairs.py" \
  --spec "$SPEC" \
  --commit-run "$COMMIT_RUN" \
  --replan-run "$REPLAN_RUN" \
  --output-dir "$OUTPUT_DIR"
