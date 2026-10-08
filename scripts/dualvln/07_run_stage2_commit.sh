#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="${HARNESS_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
OUTPUT_PATH="${OUTPUT_PATH:-${HARNESS_ROOT}/artifacts/dualvln/stage2/commit_seed42}"
HARNESS_ROOT="$HARNESS_ROOT" bash "${SCRIPT_DIR}/run_stage2_branch.sh" commit "$OUTPUT_PATH"
