#!/bin/bash
# Sourced by the ray head and every worker. DSv4-Flash FP8 rollout env + multi-node RoCE/RCCL.
#
# Cluster selection is the only thing that changes between deployments, so it lives in one
# variable and the head is the first IP. There is no per-cluster copy of this file.
#
#   default (node 1/3/5/7):  source dsv4_env.sh
#   another cluster:         export CLUSTER_MGMT_IPS="172.30.160.111 172.30.160.201 172.30.160.126 172.30.160.127"; source dsv4_env.sh
#   different head:          export MASTER_ADDR=172.30.160.131; source dsv4_env.sh
#
# Export it, do not use a "VAR=x source ..." prefix: bash treats `source` as a regular builtin, so
# the prefix assignment is dropped again the moment the source returns.
export CLUSTER_MGMT_IPS=${CLUSTER_MGMT_IPS:-"172.30.160.204 172.30.160.119 172.30.160.131 172.30.160.165"}
export MASTER_ADDR=${MASTER_ADDR:-$(set -- $CLUSTER_MGMT_IPS; echo $1)}   # ray head mgmt IP
export MILES_SCRIPT_EXTERNAL_RAY=1          # we start ray head/workers ourselves

# no_proxy must cover EVERY node's mgmt IP, not just the head: pipeline group masters resolve to
# non-head actor nodes, and a container proxy on that path blocks the Ray control plane.
export no_proxy="127.0.0.1,localhost,$(echo $CLUSTER_MGMT_IPS $MASTER_ADDR | tr ' ' '\n' | awk '!seen[$0]++' | paste -sd,)"

# --- Ray 心跳放宽(playbook §4.5:防 fabric 抖动误判节点死)---
export RAY_health_check_failure_threshold=30
export RAY_health_check_period_ms=10000
export RAY_health_check_timeout_ms=30000

# --- Ray / ROCm visibility ---
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# --- ROCm runtime knobs ---
export HIP_FORCE_DEV_KERNARG=1
export HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_USE_AITER=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ROCM_FUSED_DECODE_MLA=1
export SGLANG_USE_ROCM700A=0   # rocm720: rocm700 aiter fast-path off (validated serve leaves it unset)
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1

# --- DSv4 indexer / MHC / MoE -> aiter/triton (sglang AMD CI COMMON_ENV_VARS) ---
# NOTE: the rollout ENGINE's knobs live in run_deepseek_v4.py extra_env_vars, which is the only
# channel that actually reaches the sglang engine. Do not duplicate them here.
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
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false
export SGLANG_ROCM_USE_MULTI_STREAM=false
export AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_DSV4_FP4_EXPERTS=false
export SGLANG_OPT_USE_TILELANG_INDEXER=true
export SGLANG_DSA_TOPK_BROADCAST=1
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1

# pin each aiter config to a single file (avoid colocate config-merge baton deadlock)
AC=/sgl-workspace/aiter/aiter/configs
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=$AC/a8w8_blockscale_tuned_gemm.csv
export AITER_CONFIG_GEMM_BF16=$AC/bf16_tuned_gemm.csv
export AITER_CONFIG_GEMM_A8W8=$AC/a8w8_tuned_gemm.csv
export AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE=$AC/a8w8_bpreshuffle_tuned_gemm.csv
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=$AC/a8w8_blockscale_bpreshuffle_tuned_gemm.csv
export AITER_CONFIG_GEMM_A4W4=$AC/a4w4_blockscale_tuned_gemm.csv
export AITER_CONFIG_FMOE=$AC/tuned_fmoe.csv

# --- Multi-node RoCE / RCCL (dsv4-4node-probe.md Block 2/3; 378 GB/s GDR verified) ---
export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_7,ionic_8  # excl mgmt ionic_6
export NCCL_IB_GID_INDEX=1                  # RoCEv2 IPv4
export NCCL_SOCKET_IFNAME=enp81s0f1np1      # bootstrap on mgmt net, NOT fabric /31
export GLOO_SOCKET_IFNAME=enp81s0f1np1
export NCCL_NET_GDR_LEVEL=SYS               # GPUDirect RDMA
export NCCL_MIN_NCHANNELS=16                # playbook: 16 稳 / 32 max / 64 hang(原 112 太高有 hang 风险)
export NCCL_MAX_NCHANNELS=16
# Colocate keeps RoCE on (0) for train-collective speed: there is no cross-domain actor<->rollout
# weight-transfer group here, so the RoCE-transport establish hang cannot occur.
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
