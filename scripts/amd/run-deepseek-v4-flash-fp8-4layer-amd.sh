#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
sleep 3

set -ex

# keep Ray from blanking HIP/CUDA visibility for the job entrypoint.
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-"1"}
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-"1"}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# ROCm runtime knobs
export HIP_FORCE_DEV_KERNARG=1
export HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_USE_AITER=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ROCM_FUSED_DECODE_MLA=1
export SGLANG_USE_ROCM700A=1
export NCCL_MIN_NCHANNELS=112
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1

# DSv4 indexer / MHC / MoE -> aiter/triton (sglang AMD CI COMMON_ENV_VARS)
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_OPT_USE_FUSED_COMPRESS=true
export SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export SGLANG_HACK_FLASHMLA_BACKEND=triton
export SGLANG_OPT_FP8_WO_A_GEMM=false
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
export SGLANG_OPT_USE_TOPK_V2=false
export SGLANG_OPT_USE_AITER_INDEXER=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false
export SGLANG_ROCM_USE_MULTI_STREAM=false
export AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_DSV4_FP4_EXPERTS=false
export SGLANG_OPT_USE_TILELANG_INDEXER=true
export SGLANG_OPT_USE_COMPRESSOR_V2=false

# pin each aiter config to a single file to avoid colocate config-merge baton deadlock
AC=/sgl-workspace/aiter/aiter/configs
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=$AC/a8w8_blockscale_tuned_gemm.csv
export AITER_CONFIG_GEMM_BF16=$AC/bf16_tuned_gemm.csv
export AITER_CONFIG_GEMM_A8W8=$AC/a8w8_tuned_gemm.csv
export AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE=$AC/a8w8_bpreshuffle_tuned_gemm.csv
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=$AC/a8w8_blockscale_bpreshuffle_tuned_gemm.csv
export AITER_CONFIG_GEMM_A4W4=$AC/a4w4_blockscale_tuned_gemm.csv
export AITER_CONFIG_FMOE=$AC/tuned_fmoe.csv

export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1

MODEL_DIR=${MODEL_DIR:-/workspace/workspace/models}
DATA_DIR=${DATA_DIR:-/workspace/workspace/datasets}
HF_CKPT=${HF_CKPT:-${MODEL_DIR}/DeepSeek-V4-Flash-FP8-4layer}
NUM_GPUS=${NUM_GPUS:-8}

export PYTHONPATH=/root/miles
cd /root/miles
exec python scripts/run_deepseek_v4.py full-train \
  --model-name DeepSeek-V4-Flash-FP8-4layer \
  --hf-checkpoint "${HF_CKPT}" \
  --model-dir "${MODEL_DIR}" \
  --data-dir "${DATA_DIR}" \
  --num-nodes 1 --num-gpus-per-node "${NUM_GPUS}"
