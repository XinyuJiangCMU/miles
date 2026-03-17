#!/bin/bash
# Setup AMD MI300X environment for Miles training
# Source this script before running training: source scripts/setup_amd_env.sh

# Core AMD settings
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export HIP_FORCE_DEV_KERNARG=1
export HSA_NO_SCRATCH_RECLAIM=1

# SGLang AMD settings
export SGLANG_USE_AITER=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ROCM_FUSED_DECODE_MLA=1
export SGLANG_USE_ROCM700A=1
export SGLANG_MEMORY_SAVER_CUDA_GRAPH=true

# RCCL settings
export NCCL_MIN_NCHANNELS=112

# FP8 padding for AMD
export VLLM_FP8_PADDING=1
export VLLM_FP8_ACT_PADDING=1
export VLLM_FP8_WEIGHT_PADDING=1
export VLLM_FP8_REDUCE_CONV=1

# Inductor autotuning
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1

# Select GPUs (modify as needed)
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}

echo "AMD MI300X environment configured"
echo "  HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "  GPU count: $(echo $HIP_VISIBLE_DEVICES | tr ',' '\n' | wc -l)"
