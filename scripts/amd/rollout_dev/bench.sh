#!/bin/bash
# Bench the standalone rollout server. Decode-heavy shape mirrors training rollout:
# output len 4096 (rollout-max-response-len), concurrency 256 (max_running_requests).
# Vary knobs to A/B; the reported tok/s is what one training rollout engine delivers.
set -e
NAME=${NAME:-dsv4-rollout-dev}
PORT=${PORT:-30000}
NUM_PROMPTS=${NUM_PROMPTS:-256}
CONC=${CONC:-256}
IN_LEN=${IN_LEN:-1024}
OUT_LEN=${OUT_LEN:-4096}
docker exec "$NAME" bash -lc "
  python -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port $PORT \
    --dataset-name random --random-input-len $IN_LEN --random-output-len $OUT_LEN \
    --num-prompts $NUM_PROMPTS --max-concurrency $CONC
"
