#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

[[ $# -eq 2 ]] || die "Usage: $0 <smoke_subset|development_subset> <output-path>"
SUBSET="$1"
OUTPUT_PATH="$2"
[[ "$SUBSET" == "smoke_subset" || "$SUBSET" == "development_subset" ]] \
  || die "Unknown subset: $SUBSET"

HARNESS_ROOT="${HARNESS_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
CONFIG="${CONFIG:-scripts/eval/configs/habitat_dual_system_cfg.py}"
MANIFEST="${MANIFEST:-${HARNESS_ROOT}/configs/dualvln_stage1_episodes.json}"
MASTER_PORT="${MASTER_PORT:-2333}"
TRACE="${TRACE:-0}"
SEED="${SEED:-42}"

activate_dualvln
prepare_runtime_links
require_free_vram
export_runtime
export HARNESS_ROOT
cd "$INTERNNAV_ROOT"

require_path "$CONFIG" "DualVLN evaluation config"
require_path "$MANIFEST" "Stage 1 episode manifest"
require_path "${HARNESS_ROOT}/dualvln_stage1.py" "Stage 1 runner"

mkdir -p "$(dirname -- "$OUTPUT_PATH")"
ARGS=(
  "${HARNESS_ROOT}/dualvln_stage1.py"
  --config "$CONFIG"
  --manifest "$MANIFEST"
  --subset "$SUBSET"
  --output-path "$OUTPUT_PATH"
  --model-path "$DUALVLN_CHECKPOINT"
  --seed "$SEED"
)
if [[ "$TRACE" == "1" ]]; then
  ARGS+=(--trace)
fi
if [[ "${SAVE_TRACE_IMAGES:-0}" == "1" ]]; then
  ARGS+=(--save-trace-images)
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  ARGS+=(--allow-resume)
fi

echo "subset: $SUBSET"
echo "trace: $TRACE"
echo "output: $OUTPUT_PATH"
echo "GPU_INDEX: ${GPU_INDEX:-0}"

torchrun \
  --nproc_per_node=1 \
  --master_port="$MASTER_PORT" \
  "${ARGS[@]}" \
  2>&1 | tee "${OUTPUT_PATH}.log"
