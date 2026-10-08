#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_dualvln
prepare_runtime_links
require_free_vram
export_runtime
cd "$INTERNNAV_ROOT"

python - <<'PY'
import os

import torch
from flash_attn import flash_attn_func
from transformers import AutoProcessor
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

checkpoint = os.environ.get(
    "DUALVLN_CHECKPOINT",
    "/storage/anhdh35/checkpoints/InternVLA-N1-DualVLN",
)

q = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
flash_output = flash_attn_func(q, q, q, causal=True)
print("flash_attention_cuda: OK", tuple(flash_output.shape))

processor = AutoProcessor.from_pretrained(checkpoint)
print("processor: OK", type(processor).__name__)

torch.cuda.reset_peak_memory_stats()
model = InternVLAN1ForCausalLM.from_pretrained(
    checkpoint,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": "cuda:0"},
)
print("model: OK", type(model).__name__)
print("parameters:", sum(parameter.numel() for parameter in model.parameters()))
print("device:", next(model.parameters()).device)
print("dtype:", next(model.parameters()).dtype)
print("peak_vram_gib:", round(torch.cuda.max_memory_allocated() / 2**30, 3))
PY

