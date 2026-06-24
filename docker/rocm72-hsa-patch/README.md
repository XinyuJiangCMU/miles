# ROCm 7.2 hipFree IPC-export reference leak — partial fix (investigation artifact)

## Scope and limitation (read first)

ROCm 7.2 regressed `hipFree` so it does not return the physical pages of an allocation once it
has been IPC-exported. This directory fixes **one** path of that regression — the
`Runtime::IPCCreate` libdrm-import-reference leak introduced by the ROCR-Runtime change titled
"rocr: Make IPC Handles Unique" (commit 2f384538, first in tag rocm-7.2.0) — and is validated by
a HIP micro-repro.

**This single change does NOT, on its own, stop the end-to-end colocate offload-train memory
growth.** In a real RL run the GPU after-offload floor still rises every step with this patch
applied (the dominant training leak goes through additional ROCm 7.2-regressed path(s) that this
one fix does not cover). Treat this as a partial fix / investigation artifact, not a complete
solution.

**For a fully leak-free run, use ROCm 7.0 or 7.1** (validated end-to-end on Qwen3-30B-A3B FP8
colocate: after-offload floor stays flat, no OOM), or wait for a ROCm release that ships the full
fix. ROCm 7.0/7.1 do not have the regression at all.

## What the patch does

The regression's `IPCCreate` path does an `amdgpu_bo_import` to read/write dedup metadata on the
exported buffer and keeps that libdrm reference (`allocation_map_[ptr].ldrm_bo`); nothing releases
it, so the kernel never reclaims those pages after the kfd free. `amdgpu_bo_free.patch` releases
that transient import reference right after the metadata step. The semantics match (but are not
the same diff as) the rework already on the upstream `develop` branch. This patch is unreviewed.

## Versions

| ROCm | affected by the regression |
|---|---|
| 7.0 / 7.1 | no |
| 7.2.0 / 7.2.1 / 7.2.2 / 7.2.3 / 7.2.4 (all current 7.2 releases) | yes |
| ROCm/rocm-systems develop | already reworked/fixed; ships in a future release |

## Build

```bash
docker build -t <tag> -f docker/rocm72-hsa-patch/Dockerfile docker/rocm72-hsa-patch
```

The multi-stage Dockerfile builds the patched `libhsa-runtime64.so` from the rocm-7.2.0
`ROCR-Runtime` source and overlays it on the miles image. SONAME (`libhsa-runtime64.so.1`) is
unchanged, so nothing else is rebuilt.

## Verification (and what it does NOT show)

- Micro-repro (`hipExtMallocWithFlags(256MB) -> hipIpcGetMemHandle -> hipFree`, looped): the
  stock 7.2 runtime leaks 256 MB/iter; the patched build is flat at 0. So this specific
  IPC-export-reference path is fixed.
- End-to-end colocate RL training: **still leaks with this patch alone** — the micro-repro path
  is not the dominant training leak. ROCm 7.0/7.1 are flat end-to-end.

So this patch closes one identified regressed path; the full training fix is ROCm 7.0/7.1 (or a
future release). Tracking and version bisect are in this repo's issue.
