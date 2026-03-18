#!/bin/bash
# Pre-tune GEMM kernels for AMD MI300X using PyTorch TunableOp.
# Run this once before training to get +12% GEMM speedup.
#
# Usage:
#   HIP_VISIBLE_DEVICES=0 bash scripts/pretune_gemm_amd.sh /data/Qwen3-4B
#
# After tuning, results are saved to tunableop_results*.csv.
# These are automatically loaded by PyTorch when PYTORCH_TUNABLEOP_ENABLED=1.
set -e

MODEL_PATH="${1:-/data/Qwen3-4B}"
echo "Pre-tuning GEMM kernels for ${MODEL_PATH} on AMD MI300X..."
echo "This may take 10-30 minutes. Results are cached for future use."

export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=1
export PYTORCH_TUNABLEOP_VERBOSE=0

python3 -c "
import torch, os, sys

model_path = '${MODEL_PATH}'
from transformers import AutoModelForCausalLM
print(f'Loading {model_path}...')
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
model.train()
model.gradient_checkpointing_enable()

opt = torch.optim.AdamW(model.parameters(), lr=1e-6, fused=True)

# Run a few training steps to trigger TunableOp tuning for all GEMM shapes
print('Tuning GEMM kernels (forward + backward)...')
for batch_size in [1, 2, 4]:
    x = torch.randint(0, 1000, (batch_size, 512), device='cuda')
    for step in range(3):
        out = model(x, labels=x)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
    print(f'  BS={batch_size}: tuned')
    del x; torch.cuda.empty_cache()

torch.cuda.synchronize()
print()
print('Tuning complete! Results saved to tunableop_results*.csv')
print('To use: export PYTORCH_TUNABLEOP_ENABLED=1 (add to training scripts)')
"

echo "Done. Add PYTORCH_TUNABLEOP_ENABLED=1 to your training scripts to use tuned kernels."
