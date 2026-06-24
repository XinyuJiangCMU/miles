# ROCm 7.2 hipFree IPC-export memory-leak fix (colocate offload-train)

## What this is

ROCm 7.2 regressed `hipFree`: once an allocation has been IPC-exported, freeing it no longer
returns its physical pages to the driver. In colocate offload-train training this strands memory
every step (RCCL communicator buffers and the weight-sync buffer are IPC-exported and churned),
so the GPU after-offload floor rises until OOM within tens of steps. It is not a
miles/Megatron/sglang bug.

This directory builds a patched `libhsa-runtime64.so` from the ROCm 7.2.0 ROCR-Runtime source
(two small changes, no binary blob) and overlays it on the miles image. SONAME
(`libhsa-runtime64.so.1`) is unchanged, so nothing else is rebuilt (sglang is untouched).

## Root cause (two symmetric import-reference leaks)

To give each IPC handle unique dedup metadata, ROCm 7.2 calls `amdgpu_bo_import` on the exported
dmabuf and stores the resulting libdrm BO handle (`allocation_map_[...].ldrm_bo`). That import
reference is taken in **two** places and released in neither on the leaking paths:

1. **Exporter**: the import `Runtime::IPCCreate` stores in `ldrm_bo` is never freed —
   `Runtime::FreeMemory` clears its metadata on free but never calls `amdgpu_bo_free` (the change
   titled "rocr: Make IPC Handles Unique", commit 2f384538, first in tag rocm-7.2.0).
2. **Consumer** (`Runtime::IPCAttach`): when a consumer maps the handle, an import is done to
   read/validate the metadata, but `mapMemoryToNodes`/`fixFragment` then overwrites the map slot
   and orphans that BO, so `IPCDetach` never frees it.

Either stranded reference keeps the kernel from reclaiming the exporter allocation's pages after
free. The fix (`apply_fix.py`, two-hunk `amdgpu_bo_free.patch`) adds the missing `amdgpu_bo_free`
in both spots: the exporter one in `FreeMemory`, right after the existing free-time
metadata-clear (so a reused GEM cannot carry a stale IPC handle), and the consumer one in
`IPCAttach`, right after the metadata is read. This is semantically aligned with the rework
already on the upstream `develop` branch (which restructures the whole path onto thunk handles),
but is a minimal change against rocm-7.2.0 and is not upstream-reviewed.

Note: a single-process micro-repro that only exports and frees (never has a consumer map the
handle) hits only path 1; real training hits path 2 (torch `rebuild_cuda_tensor` and RCCL both
map via `hipIpcOpenMemHandle`), which is why both fixes are needed.

## Versions

| ROCm | affected |
|---|---|
| 7.0 / 7.1 | no |
| 7.2.0 / 7.2.1 / 7.2.2 / 7.2.3 / 7.2.4 (all current 7.2 releases) | yes |
| ROCm/rocm-systems develop | already reworked/fixed; ships in a future release |

## Build

```bash
docker build -t xinyujiangcmu/miles:rocm720-mi35x-20260621-hsapatch \
  -f docker/rocm72-hsa-patch/Dockerfile docker/rocm72-hsa-patch
```

A prebuilt image is also published at that tag for convenience.

## Verification

- Single-process micro-repro (`hipExtMallocWithFlags(256MB) -> hipIpcGetMemHandle -> hipFree`):
  stock 7.2 = 256 MB/iter, patched = 0.
- Two-process repro (producer exports, consumer `hipIpcOpenMemHandle`/close, producer frees):
  stock 7.2 = 256 MB/iter, patched = 0. This is the path real training uses.
- End-to-end colocate RL (base miles, system RCCL, weight-sync churn, no application-side
  workaround): the after-offload floor goes from +7.5 GB/iter (OOM within ~8 steps) to flat
  (104 rollouts, no OOM). The two ROCm-layer fixes remove the leak with no miles changes.

## Upstream tracking

The regression and version bisect are tracked in this repo's issue. Remove this overlay once a
ROCm 7.2.x release ships the fix (or use ROCm 7.0/7.1).
