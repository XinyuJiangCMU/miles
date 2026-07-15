#!/bin/bash
# Standalone DSv4-Flash rollout sglang server for single-node bench (rollout-speedup dev loop).
# One TP4 engine == exactly one miles training rollout engine (--sglang-tp-size 4 --sglang-ep-size 4).
# The env block + server flags mirror the live training rollout, so bench numbers transfer 1:1;
# deepseek_v4_hook auto-sets max_running_requests=256, disable_chunked_prefill, dsa_attention_backend=tilelang.
#
# This repo is bind-mounted over /root/miles so the container runs exactly this working copy
# (single source of truth -- edit here, restart the server, re-bench).
set -e
# Default to the rocm720 TRAINING image so bench numbers transfer to the live 4/6/8/9 training stack.
# (The old rocm700-mi35x-rollout-speedup image gave valid GSM8K ~0.925 but its tok/s deltas are a
#  DIFFERENT stack: 720 has newer aiter/sglang/triton -> re-validate every lever on 720.)
IMG=${IMG:-xinyujiangcmu/miles:rocm720-mi35x-20260714}
NAME=${NAME:-dsv4-rollout-dev}
MODEL=${MODEL:-/workspace/models/DeepSeek-V4-Flash-FP8}
PORT=${PORT:-30000}
GPUS=${GPUS:-0,1,2,3}

# Repo root = this script's dir up three levels (scripts/amd/rollout_dev/ -> repo root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MILES_SRC="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 1. container -- baked sglang is our rollout-speedup branch; only miles is overridden by this working copy.
if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker run -d --name "$NAME" \
    --network host --ipc host --shm-size 128g \
    --device /dev/kfd --device /dev/dri \
    --group-add video --cap-add CAP_SYS_PTRACE \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HIP_VISIBLE_DEVICES="$GPUS" \
    -v /opt/shared:/opt/shared -v /mnt/data/data/hai:/workspace \
    -v "$MILES_SRC":/root/miles \
    "$IMG" sleep infinity
fi

# 2. launch server with the training-faithful env block + flags.
docker exec -d "$NAME" bash -lc '
  mkdir -p /workspace/rollout_dev
  # Minimized rollout env (mirrors run_deepseek_v4.py extra_env_vars; 13 inert knobs dropped after audit).
  export SGLANG_SKIP_CHECKPOINT_LOAD_CHECK=1 SGLANG_DSV4_FP4_EXPERTS=0 SGLANG_HEALTH_CHECK_TIMEOUT=120
  export SGLANG_HACK_FLASHMLA_BACKEND=triton SGLANG_OPT_USE_TILELANG_INDEXER=true
  export SGLANG_OPT_USE_COMPRESSOR_V2=false SGLANG_OPT_USE_FUSED_COMPRESS=true
  export AITER_BF16_FP8_MOE_BOUND=0
  python -m sglang.launch_server \
    --model-path '"$MODEL"' --trust-remote-code \
    --tp-size 4 --ep-size 4 \
    --mem-fraction-static 0.6 --disable-cuda-graph \
    --host 0.0.0.0 --port '"$PORT"' \
    > /workspace/rollout_dev/serve.log 2>&1
'
echo "server launching on $(hostname) GPUs $GPUS port $PORT (miles src: $MILES_SRC)"
echo "watch:  tail -f /mnt/data/data/hai/rollout_dev/serve.log"
echo "ready when you see:  The server is fired up and ready to roll"
