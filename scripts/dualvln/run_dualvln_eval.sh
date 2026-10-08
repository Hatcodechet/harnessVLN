#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

[[ "${RUN_FULL_EVAL:-0}" == "1" ]] || die \
  "This runs the full val_unseen split. Re-run with RUN_FULL_EVAL=1 after the model smoke test passes."

activate_dualvln
prepare_runtime_links
require_free_vram
export_runtime
cd "$INTERNNAV_ROOT"

CONFIG="${CONFIG:-scripts/eval/configs/habitat_dual_system_cfg.py}"
MASTER_PORT="${MASTER_PORT:-2333}"
RUN_NAME="${RUN_NAME:-dualvln_val_unseen_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${INTERNNAV_ROOT}/logs/harnessvln}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

require_path "$CONFIG" "Evaluation config"
mkdir -p "$LOG_DIR" "${INTERNNAV_ROOT}/logs/habitat/test_dual_system"

echo "Starting full DualVLN evaluation"
echo "config: $CONFIG"
echo "GPU_INDEX: ${GPU_INDEX:-0}"
echo "log: $LOG_FILE"

torchrun \
  --nproc_per_node=1 \
  --master_port="$MASTER_PORT" \
  scripts/eval/eval.py \
  --config "$CONFIG" \
  2>&1 | tee "$LOG_FILE"

