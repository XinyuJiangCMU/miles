#!/bin/bash
# DSv4-Flash rollout server WITH MTP/EAGLE spec-decode + cuda-graph (node-7 A/B dev loop).
# EAGLE only (deepseek_v4_hook.py: DSv4 asserts EAGLE + eagle-topk==1); ns=3/draft=4 is the
# ROCm optimum (accept-len ~2.5). Requires the compress_state.py cast fix (bf16 spec kv_score
# -> float32 buffer) already applied in the container; else EAGLE crashes Index-put Float/BF16.
# Same env/flags as serve_cudagraph.sh so numbers transfer 1:1; only spec-decode flags added.
set -e
IMG=${IMG:-xinyujiangcmu/miles:rocm720-mi35x-20260714}
NAME=${NAME:-dsv4-rollout-dev}
MODEL=${MODEL:-/workspace/models/DeepSeek-V4-Flash-FP8}
PORT=${PORT:-30000}
GPUS=${GPUS:-0,1,2,3}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MILES_SRC="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker run -d --name "$NAME" \
    --network host --ipc host --shm-size 128g \
    --device /dev/kfd --device /dev/dri \
    --group-add video --cap-add CAP_SYS_PTRACE \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HIP_VISIBLE_DEVICES="$GPUS" \
    -v /opt/shared:/opt/shared -v /mnt/data/data/hai:/workspace \
    "$IMG" sleep infinity
fi

docker exec -d "$NAME" bash -lc '
  mkdir -p /workspace/rollout_dev
  export SGLANG_SKIP_CHECKPOINT_LOAD_CHECK=1 SGLANG_DSV4_FP4_EXPERTS=0 SGLANG_HEALTH_CHECK_TIMEOUT=120
  export SGLANG_HACK_FLASHMLA_BACKEND=triton SGLANG_OPT_USE_TILELANG_INDEXER=true
  export SGLANG_OPT_USE_COMPRESSOR_V2=false SGLANG_OPT_USE_FUSED_COMPRESS=true
  export AITER_BF16_FP8_MOE_BOUND=0
  python -m sglang.launch_server \
    --model-path '"$MODEL"' --trust-remote-code \
    --tp-size 4 --ep-size 4 \
    --mem-fraction-static 0.6 \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path '"$MODEL"' \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --host 0.0.0.0 --port '"$PORT"' \
    > /workspace/rollout_dev/serve_mtp.log 2>&1
'
echo "MTP server launching on $(hostname) GPUs $GPUS port $PORT (miles src: $MILES_SRC)"
echo "watch:  tail -f /mnt/data/data/hai/rollout_dev/serve_mtp.log"
echo "ready when you see:  The server is fired up and ready to roll"
