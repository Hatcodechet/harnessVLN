#!/usr/bin/env bash

set -Eeuo pipefail

CONDA_ROOT="${CONDA_ROOT:-/storage/anhdh35/miniconda3}"
CONDA_ENV="${CONDA_ENV:-dualvln_harness}"
INTERNNAV_ROOT="${INTERNNAV_ROOT:-/storage/anhdh35/InternNav_harness}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/storage/anhdh35/checkpoints}"
JANUSVLN_ROOT="${JANUSVLN_ROOT:-/storage/anhdh35/JanusVLN}"
DUALVLN_CHECKPOINT="${DUALVLN_CHECKPOINT:-${CHECKPOINT_ROOT}/InternVLA-N1-DualVLN}"
DEPTH_CHECKPOINT="${DEPTH_CHECKPOINT:-${CHECKPOINT_ROOT}/depth_anything_v2_metric_hypersim_vits.pth}"
R2R_ROOT="${R2R_ROOT:-${JANUSVLN_ROOT}/data/datasets/r2r}"
SCENES_ROOT="${SCENES_ROOT:-${JANUSVLN_ROOT}/data/scene_datasets}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "$path" ]] || die "$label not found: $path"
}

activate_dualvln() {
  require_path "${CONDA_ROOT}/etc/profile.d/conda.sh" "Conda activation script"
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
}

prepare_runtime_links() {
  require_path "$INTERNNAV_ROOT" "InternNav checkout"
  require_path "$DUALVLN_CHECKPOINT" "DualVLN checkpoint"
  require_path "$DEPTH_CHECKPOINT" "DepthAnything checkpoint"
  require_path "$R2R_ROOT" "R2R dataset"
  require_path "$SCENES_ROOT" "Matterport3D scene dataset"

  mkdir -p \
    "${INTERNNAV_ROOT}/data/scene_data" \
    "${INTERNNAV_ROOT}/data/vln_ce/raw_data"

  link_if_missing "${CHECKPOINT_ROOT}" "${INTERNNAV_ROOT}/checkpoints"
  link_if_missing "${SCENES_ROOT}" "${INTERNNAV_ROOT}/data/scene_data/mp3d_ce"
  link_if_missing "${R2R_ROOT}" "${INTERNNAV_ROOT}/data/vln_ce/raw_data/r2r"
}

link_if_missing() {
  local target="$1"
  local link="$2"

  if [[ -L "$link" ]]; then
    [[ "$(readlink -f "$link")" == "$(readlink -f "$target")" ]] \
      || die "Symlink points elsewhere: $link -> $(readlink "$link")"
    return
  fi

  [[ ! -e "$link" ]] || die "Refusing to replace existing path: $link"
  ln -s "$target" "$link"
}

require_free_vram() {
  local minimum_mib="${MIN_FREE_VRAM_MIB:-24576}"
  local gpu_index="${GPU_INDEX:-0}"
  local free_mib

  command -v nvidia-smi >/dev/null || die "nvidia-smi is not available"
  free_mib="$({ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_index"; } | head -n1 | tr -d ' ')"
  [[ "$free_mib" =~ ^[0-9]+$ ]] || die "Could not read free VRAM for GPU $gpu_index"
  (( free_mib >= minimum_mib )) \
    || die "GPU $gpu_index has ${free_mib} MiB free; need at least ${minimum_mib} MiB"
  echo "GPU $gpu_index free VRAM: ${free_mib} MiB"
}

export_runtime() {
  export CONDA_ROOT CONDA_ENV INTERNNAV_ROOT CHECKPOINT_ROOT JANUSVLN_ROOT
  export DUALVLN_CHECKPOINT DEPTH_CHECKPOINT R2R_ROOT SCENES_ROOT
  export CUDA_VISIBLE_DEVICES="${GPU_INDEX:-0}"
  export PYTHONUNBUFFERED=1
  export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
  export HABITAT_SIM_LOG="${HABITAT_SIM_LOG:-quiet}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export PYTHONPATH="${INTERNNAV_ROOT}/third_party/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}"
}
