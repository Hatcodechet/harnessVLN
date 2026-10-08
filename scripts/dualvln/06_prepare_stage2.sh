#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
HARNESS_ROOT="${HARNESS_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
EXPECTED_INTERNNAV_SHA="7a5c62400ac45b313d9b709c740b64191556a242"
TARGET="${INTERNNAV_ROOT}/internnav/habitat_extensions/vln/habitat_vln_evaluator.py"
PATCH="${HARNESS_ROOT}/patches/internnav_stage2_hook.patch"
SPEC="${STAGE2_SPEC:-${HARNESS_ROOT}/artifacts/dualvln/stage2/candidates.json}"
DEV20="${DEV20_RUN:-${HARNESS_ROOT}/artifacts/dualvln/dev20_trace}"

require_path "$TARGET" "Pinned InternNav evaluator"
require_path "$PATCH" "Stage 2 InternNav hook patch"

if grep -q 'DUALVLN_BRANCH_SPEC' "$TARGET"; then
  echo "Stage 2 hook already applied: $TARGET"
else
  actual_sha="$(git -C "$INTERNNAV_ROOT" rev-parse HEAD)"
  [[ "$actual_sha" == "$EXPECTED_INTERNNAV_SHA" ]] \
    || die "InternNav SHA mismatch: $actual_sha (expected $EXPECTED_INTERNNAV_SHA)"
  git -C "$INTERNNAV_ROOT" apply --check "$PATCH"
  git -C "$INTERNNAV_ROOT" apply "$PATCH"
  echo "Applied Stage 2 hook to pinned InternNav checkout"
fi

activate_dualvln
python -m py_compile "$TARGET" "${HARNESS_ROOT}/stage2_branch.py"
python "${HARNESS_ROOT}/select_stage2_candidates.py" "$DEV20" --output "$SPEC"
echo "Stage 2 candidates: $SPEC"
