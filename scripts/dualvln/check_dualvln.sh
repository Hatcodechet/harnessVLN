#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_dualvln
prepare_runtime_links
export_runtime
cd "$INTERNNAV_ROOT"

python - <<'PY'
import gzip
import json
import os

import accelerate
import diffusers
import flash_attn
import habitat
import habitat_sim
import torch
import transformers
from internnav.evaluator import Evaluator
import internnav.habitat_extensions.vln  # registry side effects

manifest = "data/vln_ce/raw_data/r2r/val_unseen/val_unseen.json.gz"
with gzip.open(manifest, "rt") as handle:
    episode = json.load(handle)["episodes"][0]
scene = os.path.join("data/scene_data/mp3d_ce", episode["scene_id"])

assert os.path.isfile(scene), scene
assert "habitat_vln" in Evaluator.evaluators
assert torch.cuda.is_available()
assert "sm_120" in torch.cuda.get_arch_list()

print("environment: OK")
print("torch:", torch.__version__, "CUDA", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("habitat_sim:", getattr(habitat_sim, "__version__", "unknown"))
print("flash_attn:", flash_attn.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("diffusers:", diffusers.__version__)
print("first_scene:", os.path.realpath(scene))
print("evaluator:", Evaluator.evaluators["habitat_vln"])
PY

python -m pip check

