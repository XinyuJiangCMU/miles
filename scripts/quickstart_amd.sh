#!/bin/bash
# Quick-start script for Miles on AMD MI300X
# Usage: HIP_VISIBLE_DEVICES=0,1 bash scripts/quickstart_amd.sh
set -euo pipefail

echo "=== Miles AMD MI300X Quick-Start ==="
echo ""

# Verify AMD environment
python3 -c "
import torch
assert torch.version.hip, 'ROCm not detected'
assert torch.cuda.is_available(), 'No GPU available'
print(f'ROCm: {torch.version.hip}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPUs: {torch.cuda.device_count()}')
"

# Set AMD environment variables
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export SGLANG_USE_AITER=1
export SGLANG_MEMORY_SAVER_CUDA_GRAPH=true
export SGLANG_MOE_PADDING=1

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Using ${NUM_GPUS} GPUs"

# Check if model exists
MODEL_DIR="${MODEL_DIR:-/data/Qwen3-4B}"
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model not found at $MODEL_DIR"
    echo "Set MODEL_DIR to your model path"
    exit 1
fi

# Run environment check
echo ""
echo "=== Environment Check ==="
python3 tools/check_amd_env.py 2>&1 | grep -E "PASS|FAIL|Results"

# Run tests
echo ""
echo "=== Running AMD Tests ==="
python3 -m pytest tests/fast/test_fp8_amd.py -q 2>&1 | tail -3

echo ""
echo "=== Quick-Start Complete ==="
echo ""
echo "To train with FSDP:"
echo "  HIP_VISIBLE_DEVICES=0,1 bash scripts/run-qwen3-4B-fp8-amd.sh"
echo ""
echo "To convert to FP8 for inference:"
echo "  python3 tools/convert_hf_to_fp8.py --model-dir $MODEL_DIR --save-dir ${MODEL_DIR}-FP8 --strategy tensor"
echo ""
echo "To benchmark:"
echo "  python3 tools/benchmark_amd.py"
