#!/bin/bash
# AMD MI300X Quick Reference for Miles FP8 Training
#
# Environment Setup:
#   export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#   export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
#
# Performance Tuning (auto-set by execute_train on AMD):
#   export HIP_FORCE_DEV_KERNARG=1
#   export NCCL_BUFFSIZE=16777216
#   export HSA_NO_SCRATCH_RECLAIM=1
#   export SGLANG_USE_AITER=1
#   export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
#
# Training Commands:
#
#   1. Quick test (1 GPU):
#      HIP_VISIBLE_DEVICES=0 python tools/benchmark_e2e_amd.py --model /data/Qwen3-4B
#
#   2. Environment check:
#      HIP_VISIBLE_DEVICES=0 python tools/check_amd_env.py
#
#   3. Run all AMD tests:
#      HIP_VISIBLE_DEVICES=0 bash scripts/run_amd_tests.sh
#
#   4. Convert model to FP8:
#      python tools/convert_hf_to_fp8.py --model-dir /data/Qwen3-4B --save-dir /data/Qwen3-4B-FP8 --strategy tensor
#
#   5. Pre-tune GEMM kernels:
#      HIP_VISIBLE_DEVICES=0 bash scripts/pretune_gemm_amd.sh /data/Qwen3-4B
#
#   6. Full RL training (2 GPUs):
#      HIP_VISIBLE_DEVICES=0,1 bash scripts/run-qwen3-4B-fp8-amd.sh
#
#   7. Training benchmark:
#      HIP_VISIBLE_DEVICES=0 python tools/benchmark_training_amd.py --model /data/Qwen3-4B
#
# Key Training Flags:
#   --no-offload-train           Keep model on GPU (MI300X has 192GB)
#   --gradient-checkpointing     Save memory (47% reduction)
#   --micro-batch-size 2         Better throughput (+56%)
#   --sglang-disable-custom-all-reduce  Avoid hipIpc issues
#   --compile-log-probs          Compile forward for 37% faster inference
#   --bf16-reduce                Halve gradient communication volume
#
# Performance Results (Qwen3-4B, 1x MI300X):
#   Baseline:  1,423 tok/s
#   Optimized: 8,431 tok/s (5.9x improvement)
#
# Documentation:
#   docs/amd_training_optimization.md
#
echo "See this file for AMD MI300X quick reference commands."
echo "Run 'python tools/check_amd_env.py' to verify your setup."
