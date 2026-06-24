# ROCm 7.2 hipFree IPC-export memory-leak fix (for colocate offload-train)

## What this is

In colocate offload-train training the GPU after-offload memory floor rises every step and
OOMs within tens of steps. This is a **ROCm 7.2 regression**, not a miles/Megatron/sglang bug:
on ROCm 7.2, `hipFree` no longer returns the physical pages of an allocation once it has been
IPC-exported (via `hipIpcGetMemHandle`), so the per-step churn of IPC-exported buffers strands
memory until OOM.

This directory rebuilds `libhsa-runtime64.so` from source with the fix and overlays it on the
miles image — without touching sglang or anything else.

## Versions

| ROCm | affected |
|---|---|
| 7.0 / 7.1 | no (pages are returned correctly) |
| 7.2.0 / 7.2.1 / 7.2.2 / 7.2.3 / 7.2.4 (all current 7.2 releases) | yes |
| ROCm/rocm-systems develop | already reworked/fixed; ships in a future release |

So ROCm 7.0/7.1 also avoid the bug, and a future ROCm release will too. This overlay is only for
staying on ROCm 7.2.

## Root cause and fix

The regression came from the ROCR-Runtime change titled "rocr: Make IPC Handles Unique"
(commit 2f384538, first in tag rocm-7.2.0): `Runtime::IPCCreate` does an `amdgpu_bo_import` to
read/write dedup metadata on the exported buffer but keeps that libdrm reference
(`allocation_map_[ptr].ldrm_bo`), and nothing ever releases it, so the kernel never reclaims the
pages after the kfd free. `amdgpu_bo_free.patch` releases that transient import reference right
after the metadata step (covering both metadata branches), which is semantically equivalent to —
but not the same diff as — the rework already on the upstream `develop` branch. This patch is a
minimal change against the rocm-7.2.0 source and has not been reviewed upstream; it is a stopgap.

## Use it

```bash
docker build -t xinyujiangcmu/miles:rocm720-mi35x-20260621-hsapatch \
  -f docker/rocm72-hsa-patch/Dockerfile docker/rocm72-hsa-patch
```

The multi-stage Dockerfile builds the patched `libhsa-runtime64.so` from the ROCm 7.2.0
`ROCR-Runtime` source (applying `amdgpu_bo_free.patch`) and copies it over `/opt/rocm/lib` in the
miles image. The SONAME (`libhsa-runtime64.so.1`) is unchanged, so nothing else is rebuilt. A
prebuilt image is also published at the tag above for convenience.

## Verification

A 30-line HIP repro (`hipExtMallocWithFlags(256MB) -> hipIpcGetMemHandle -> hipFree`, looped,
measured with `hipMemGetInfo`): the stock 7.2 runtime leaks 256 MB/iter; the patched build is
flat at 0. End-to-end on Qwen3-30B-A3B FP8 colocate the GPU after-offload floor stays flat
instead of growing to OOM.

## Upstream tracking

Regression and version bisect are tracked in this repo (issue: the ROCm 7.2 hipFree IPC-export
regression). Remove this overlay once a ROCm 7.2.x release ships the FreeMemory/IPCCreate fix
(or move to ROCm 7.0/7.1).
