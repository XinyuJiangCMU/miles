# Gluon blockscale FP8 GEMM vs AITER — candidate result: DOES NOT WIN

Date: 2026-06-28. Box: MI355X / gfx950 / ROCm 7.0, container dsv4-fp8-v14 (now destroyed, see §5).
Toolchain: gluon-venv (triton 3.7.1, `--system-site-packages`, system triton 3.4 保底不动).

## 1. What was tested
aiter ships three implementations of the SAME a8w8 blockwise-fp8 GEMM (1x128 act / 128x128 weight),
all with the same call convention (x(M,K) e4m3fn, w(N,K) not-preshuffled, x_scale(M,⌈K/128⌉) fp32,
w_scale(⌈N/128⌉,⌈K/128⌉) fp32, bf16 out):
- **ck_eager**    — `aiter.gemm_a8w8_blockscale` (production CK, non-preshuffled; may be un-tuned for these shapes)
- **asm_preshuf** — `aiter.gemm_a8w8_blockscale_bpreshuffle_asm` (AITER asm-best; weight shuffle_weight'd offline once, x_scale transposed; bf16-only)
- **gluon**       — `aiter.ops.triton.gluon.gemm_a8w8_blockscale` (the candidate)
- **triton**      — `aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale` (plain-triton reference)

Bench: real DeepSeek-V4-Flash-FP8-4layer training GEMM shapes (recon-verified per-GPU TP8/EP8, not made-up).
Hard parity gate (finite + cos_ref>0.99 + relerr_vs_ck<0.02; failing impls' timing nullified).
CUDA-event back-to-back timing, min-of-6 reps (rejects interference; not per-iter-sync wall-clock).

## 2. Results (6 clean shapes; us/call, lower=faster; speedup>1 = gluon faster)

| shape (M,N,K)            | ck_eager | asm_preshuf | gluon | triton | gl/ck | gl/asm | gl/tr | parity |
|--------------------------|---------:|------------:|------:|-------:|------:|-------:|------:|:------:|
| wq_b/wo_b 2048,4096,1024 |   19.54  |    27.35    | 37.47 | 38.66  | 0.52  | 0.73   | 1.03  | PASS   |
| wq_a     2048,1024,4096  |   21.50  |    27.46    | 95.74 | 69.29  | 0.22  | 0.29   | 0.72  | PASS   |
| wkv      2048, 512,4096  |   18.07  |    26.94    | 95.68 | 66.73  | 0.19  | 0.28   | 0.70  | PASS   |
| shared_fc1 2048,512,4096 |   18.59  |    26.64    | 95.94 | 66.80  | 0.19  | 0.28   | 0.70  | PASS   |
| shared_fc2 2048,4096,256 |   13.19  |    26.38    | 32.65 | 37.84  | 0.40  | 0.81   | 1.16  | PASS   |
| moe_fc1   256,4096,4096  |   12.38  |    26.72    | 94.65 | 57.34  | 0.13  | 0.28   | 0.61  | PASS   |

Geomean (large-M ≥1536, 5 shapes): gl/ck **0.28x**, gl/asm 0.42x, gl/tr 0.84x — **0 wins**.
Small-M (≤256, 1 shape): gl/ck 0.13x, gl/asm 0.28x, gl/tr 0.61x — **0 wins**.

## 3. Verdict — DOES NOT WIN
Gluon is **numerically correct** (cos=1.000, relerr_vs_ck=0.0 on every shape) but **2–8× SLOWER**
than AITER's CK/asm production kernels on all DSv4 shapes, and even loses to plain triton on most.
Worst at K=4096 (gluon ~95us vs CK ~18-21us, ~5× slower). No regime where gluon beats AITER.

## 4. Why it loses (root cause, from source recon)
This aiter gluon GEMM does **not** use the techniques the PyTorch TokenSpeed-kernel blog credits for
its CDNA4 wins. Per source read of `gemm_a8w8_blockscale.py`:
- **No hardware mfma_scaled** — uses plain `gl.amd.cdna4.mfma` (V_MFMA_F32_16x16x32_FP8_FP8) and applies
  block scales MANUALLY post-MFMA (`acc += mfma_out * a_scale * b_scale`), not a scaled-MFMA.
- **No async_copy/async_wait** — classic manual software-pipelined double-buffer.
- **Not persistent** — one-shot grid, no persistent+XCD load-balancing (XCD remap only in the KSPLIT=1 path).
It is an early/un-optimized gluon kernel. The blog's 1.1–1.6× wins were for **attention (MLA/decode) and
fused-MoE** kernels — different gluon kernels — not this blockscale GEMM.

## 5. Where a (hypothetical) win would land — and why it wouldn't, here
- **Training (Megatron/TE): no benefit.** On ROCm, TransformerEngine routes blockwise-fp8 GEMM to its OWN
  vendored triton kernel `transformer_engine/pytorch/triton_kernels/blockwise_fp8_gemm.py` (Primus-Turbo /
  tritonBLAS lineage). `grep gemm_a8w8_blockscale` in the TE package = 0 hits. aiter is not in the training path.
- **Serving (sglang): only for whitelisted shapes, and not DSv4's.** sglang's `fp8_utils.py` routes the dense
  fp8 Linear to the triton path (which the gluon kernel could drop into) only for 14 tuned (n,k) pairs
  (all DeepSeek-R1/V3 shapes: 7168/32768…). **None of DSv4-4layer's dense (n,k) are in it** → DSv4 serving
  uses CK. Plus ROCm-7.0 hipcc miscompiles the bpreshuffle path on gfx95 (# 23319; sglang guards it to hip≥7.2),
  and live serving runs system triton 3.4 (<3.6, can't run gluon) which is 保底 (not to be changed).
- Net: even a winning gluon GEMM would not land in this training run or this container's serving without
  (whitelist + import swap + triton≥3.6) — i.e. ecosystem/future, not here.

## 6. Operational notes / hazards hit
- **stale baton lock deadlock**: killing an aiter process mid config-merge leaves
  `/tmp/aiter_configs/<op>_tuned_gemm.csv.lock` (a FileBaton). Every later `aiter.gemm_a8w8_blockscale`
  first-call then `wait()`s on it FOREVER (py-spy: file_baton.py:54). Fix = `rm` the stale `.lock` (+ `.tmp`);
  the waiter then proceeds. This caused several "bench hangs with no output".
- **container destroyed by SLURM allocation timeout (root cause)**: this box is SLURM-managed
  (host `amd-mi355x-ses2-1`). Our work ran under job **13138** (`no-shell`, user amd, 24h walltime). It hit
  **TIMEOUT at 22:46:30** (sacct: Start 2026-06-27 22:46:20 → End 2026-06-28 22:46:30); SLURM reclaimed the
  node and its epilog `kill -9`'d pid1 + `docker destroy`'d container dsv4-fp8-v14 ~64s later (docker events
  22:47:34–39), then handed ses2-1 to user dn (job 13153, now RUNNING there). The HIP **illegal memory access
  (err 700)** the bench hit at that instant was the **GPU being yanked by SLURM mid-run**, NOT an asm bug —
  asm had run the same K=4096 shape fine one row earlier (moe_fc1 M256, asm 26.72us). Not caused by any docker
  rm of mine; GPU recovered clean (host rocm-smi 297MB); other tenants (hai-1, multinode) unaffected. To
  continue, a fresh SLURM allocation + container recreate is needed (infra; user's call).
- **standing caution (separate)**: sglang guards the aiter bpreshuffle path off on ROCm 7.0/gfx95 (its tracked
  hipcc miscompile; `_use_aiter_bpreshuffle_gfx95` requires hip≥7.2), so prefer ck_eager as the AITER baseline
  regardless — it is also what DSv4 serving actually uses here.

## 7. Next
Candidate "gluon blockscale GEMM" = recorded LOSS. Next candidates (need a working container): the gluon
kernels the blog actually wins with — pa_decode/MLA-decode attention, or fused-MoE — benched vs their aiter
asm/CK baselines. Blocked on container dsv4-fp8-v14 recreation (infra decision — surfaced to user).
