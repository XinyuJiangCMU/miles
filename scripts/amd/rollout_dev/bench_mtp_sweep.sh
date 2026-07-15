#!/bin/bash
# Concurrency sweep for MTP vs cuda-graph-only. max-running-requests=256 (>> conc; the md's
# max-running=8 artifact lesson). Measures where spec-decode wins (low conc / rollout tail) vs
# breaks even (high conc). Run against a live server on PORT (30000).
PORT=${PORT:-30000}
MODEL=${MODEL:-/workspace/models/DeepSeek-V4-Flash-FP8}
OUT=/workspace/rollout_dev/bench_mtp_sweep.txt
: > $OUT
for C in 1 4 8 32 128 256; do
  echo "=== conc=$C ===" | tee -a $OUT
  python3 -m sglang.bench_serving --backend sglang --model $MODEL \
    --num-prompts $((C*4)) --max-concurrency $C \
    --random-input 512 --random-output 1024 --dataset-name random \
    --port $PORT 2>&1 | grep -iE "throughput|tok/s|Output token|accept|Total token|Request thr|Mean|median" | sed "s/^/  /" | tee -a $OUT
done
echo "=== accept-len from server log ===" | tee -a $OUT
grep -oiE "accept.?len[^,}]*|acceptance[^,}]*" /workspace/rollout_dev/serve_mtp.log 2>/dev/null | tail -5 | tee -a $OUT
