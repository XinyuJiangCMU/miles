# ROCm 7.2 hipFree IPC-export memory-leak fix (for colocate offload-train)

## What this is

In colocate offload-train training, GPU memory grows every step and OOMs within tens of steps.
The root cause is **a ROCm 7.2 regression**, not a miles/Megatron/sglang bug: on ROCm 7.2,
`hipFree` no longer returns the physical pages of an allocation once it has been IPC-exported
(via `hipIpcGetMemHandle`). Colocate training churns IPC-exported buffers every step (RCCL
communicators and the weight-sync flatten buffer), so each step strands memory until OOM.

This directory overlays a rebuilt `libhsa-runtime64.so` that fixes the leak, without touching
sglang or anything else in the image.

## Versions

| ROCm | affected? |
|---|---|
| 7.0 / 7.1 | no (pages are returned correctly) |
| 7.2.0 / 7.2.1 / 7.2.2 / 7.2.3 / 7.2.4 (latest) | **yes** |
| ROCm/rocm-systems `develop` | already fixed (reworked); ships in a future release |

So you can also just use ROCm 7.0/7.1, or wait for the next ROCm release. This overlay is for
staying on ROCm 7.2.

## The fix

The regression was introduced by the ROCR-Runtime commit "rocr: Make IPC Handles Unique" (#795):
`Runtime::IPCCreate` takes a libdrm BO reference via `amdgpu_bo_import` and stores it in
`allocation_map_[ptr].ldrm_bo`, but `Runtime::FreeMemory` only clears the BO metadata and never
calls `amdgpu_bo_free`, so the reference outlives the kfd free and the kernel never reclaims the
pages. The fix (`amdgpu_bo_free.patch`) adds the missing `amdgpu_bo_free` in `FreeMemory`:

```cpp
DRM_CALL(amdgpu_bo_free(it->second.ldrm_bo));
it->second.ldrm_bo = nullptr;
```

`libhsa-runtime64.so.1.18.70200` here is the ROCm 7.2.0 ROCR-Runtime source built with that
patch. SONAME is `libhsa-runtime64.so.1` (unchanged), ABI-compatible with the rest of the 7.2
image.

## Use it

```bash
docker build -t xinyujiangcmu/miles:rocm720-mi35x-20260621-hsapatch \
  -f docker/rocm72-hsa-patch/Dockerfile docker/rocm72-hsa-patch
```

A prebuilt image is also pushed at `xinyujiangcmu/miles:rocm720-mi35x-20260621-hsapatch`.

## Verification

A 30-line HIP repro (`hipExtMallocWithFlags(256MB) -> hipIpcGetMemHandle -> hipFree`, looped,
measured with `hipMemGetInfo`): with the stock 7.2 runtime it leaks 256 MB/iter; with this
`.so` it is flat at 0 MB/iter. End-to-end (Qwen3-30B-A3B FP8 colocate) the GPU after-offload
floor stays flat instead of growing to OOM.

## Rebuilding `libhsa-runtime64.so` from source

Apply `amdgpu_bo_free.patch` to the ROCm 7.2.0 `ROCR-Runtime` tree
(`runtime/hsa-runtime/core/runtime/runtime.cpp`), then build the `hsa-runtime64` target from the
top-level CMake project in a ROCm 7.2 + LLVM dev environment with `ROCM_PATCH_VERSION=70200`.
The resulting `build/.../libhsa-runtime64.so.1.18.*` is what is shipped here.
