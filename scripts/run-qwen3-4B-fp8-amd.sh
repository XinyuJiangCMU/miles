#!/bin/bash
# FP8 Training Test for Qwen3-4B on AMD MI300X
# Uses 2+ GPUs with FSDP backend
# Supports both BF16 and FP8 quantized checkpoints

# Kill any existing processes
pkill -9 sglang 2>/dev/null
sleep 2
ray stop --force 2>/dev/null
pkill -9 ray 2>/dev/null
sleep 2

set -euxo pipefail

### AMD Support ###
MILES_DIR="${MILES_DIR:-/app/miles}"
export MILES_DIR

MODEL_DIR="${MODEL_DIR:-/data}"
export MODEL_DIR

DATA_DIR="${DATA_DIR:-/tmp/test_data}"
export DATA_DIR

# For AMD GPU - select specific GPUs
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"6,7"}

# AMD performance tuning
export HIP_FORCE_DEV_KERNARG=1
export HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_USE_AITER=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export VLLM_FP8_PADDING=1
export VLLM_FP8_ACT_PADDING=1
export VLLM_FP8_WEIGHT_PADDING=1
export VLLM_FP8_REDUCE_CONV=1
export TORCHINDUCTOR_MAX_AUTOTUNE=1

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

# Use FP8 checkpoint if available, otherwise BF16
HF_CHECKPOINT="${MODEL_DIR}/Qwen3-4B"
if [ -d "${MODEL_DIR}/Qwen3-4B-FP8" ]; then
    HF_CHECKPOINT="${MODEL_DIR}/Qwen3-4B-FP8"
    echo "Using FP8 checkpoint: ${HF_CHECKPOINT}"
fi

CKPT_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/test_math.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 60
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 512
   --rollout-temperature 1
   --global-batch-size 64
   --balance-data
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   # Lower memory fraction for colocate mode to avoid OOM
   --sglang-mem-fraction-static 0.3
   # Disable custom all-reduce on non-default GPU sets (hipIpcOpenMemHandle issue)
   --sglang-disable-custom-all-reduce
)

# launch ray
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NUM_GPUS=$(echo ${HIP_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Dynamically detect Megatron-LM path
MEGATRON_LM_PATH=$(python3 -c "import megatron; import os; print(os.path.dirname(os.path.dirname(megatron.__file__)))" 2>/dev/null || echo "/app/Megatron-LM")

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"${MEGATRON_LM_PATH}/\",
        \"SGLANG_MEMORY_SAVER_CUDA_GRAPH\": \"true\"
     }
   }" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   --train-backend fsdp \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${SGLANG_ARGS[@]}
