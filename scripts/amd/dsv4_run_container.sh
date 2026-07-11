#!/bin/bash
# Start the miles training container on each Pod B node (2,4,6,9). RDMA-complete recipe (JOURNEY 4.3).
set -e
IMG=${IMG:-xinyujiangcmu/miles:rocm700-mi35x-rollout-speedup-20260711}
NAME=${NAME:-dsv4-train}
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" \
  --network host --ipc host --shm-size 128g \
  --device /dev/kfd --device /dev/dri \
  $(for d in /dev/infiniband/*; do printf -- '--device=%s ' "$d"; done) \
  --group-add video --cap-add CAP_SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /opt/shared:/opt/shared -v /mnt/data/data/hai:/workspace \
  "$IMG" sleep infinity
echo "container up on $(hostname); ibv check:"
docker exec "$NAME" bash -lc 'ibv_devices 2>/dev/null | tail -n +3 | wc -l' || echo "ibv_devices failed"
