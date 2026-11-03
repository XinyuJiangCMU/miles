# ROCm 7.2 colocate 训练显存泄漏:排查全过程与定论

一句话结论:**这不是 miles / Megatron / sglang 的训练逻辑问题,而是 ROCm 7.2 的一个回归** —— `hipFree` 在一块显存被 IPC 导出过之后不再把物理页还给驱动。ROCm 7.0 / 7.1 没有这个问题;所有 7.2.x(含当前最新 7.2.4)都有;上游 develop 分支已经重构修好。**能用的版本是 7.0 / 7.1。**

---

## 1. 现象

colocate RL 训练(Megatron 训练侧 + sglang 推理侧共享同一张 GPU,`--offload-train`):每个训练 step 结束后,GPU 显存不能完全回落,**after-offload 的 GPU floor 每步往上涨**,大约 step 7~8 撞满显存 OOM。一开始很容易误以为是训练越用越多 / 模型配置 / eval 的问题。

一个重要的判据修正:**要看 GPU 的 after-offload floor,不要看 CPU 内存**。CPU 内存在 7.0 和 7.2 上都会涨(约 +7GB/step),那是另一回事,不是这条会 OOM 的 GPU 漏。

---

## 2. 排查过程(按时间,简要)

1. **先怀疑 RCCL 通信器**。colocate 每个 offload 周期会 destroy + recreate 6 个 RCCL communicator;在 ROCm 上 `dist.destroy_process_group()` 不释放 communicator 的 device buffer → 每 iter 漏 ~6GB。
   - 试 **no-churn**(ROCm 上干脆不 destroy、跨 offload 保活):漏从 +6.3GB/iter 降到 +0.9GB/iter。但"保活"会让 communicator 常驻、破坏 colocate 的 offload。
2. **Solution B**(RCCL 侧 buffer 复用 cache):churn 照常 destroy/recreate,但复用 device buffer 不真正 hipFree → 既不漏又能 offload。主漏(+6.3)消除。
3. **追剩下的 +0.9GB/iter 残留**。逐步插桩定位到 `update_weights()`:Megatron 把权重 flatten 成一个大 buffer、经 IPC(`MultiprocessingSerializer` / `hipIpcGetMemHandle`)导出给 sglang。每 step 新建一个导出 buffer → 漏一份 ≈ 0.9GB。
4. **miles 侧 buffer 复用**修掉 +0.9:复用同一块导出 buffer、不每 iter 新建。斜率 → 0。
5. **多 agent + NV 侧 review**:NV 在 B200 上跑同一条路径 250 iter,显存死平(≈0.002GB/iter)。**说明这是 AMD 特有**,不是通用问题,CUDA 路径不该动。
6. **最小复现 + 版本 bisect**:写了 30 行 C++(`churn_ipc.cpp`):`hipExtMallocWithFlags(256MB) → hipIpcGetMemHandle → hipFree` 循环,量 `hipMemGetInfo`。同一个 binary,只换 `libamdhip64.so` 运行时:

   | ROCm 运行时 | 导出后 hipFree |
   |---|---|
   | 7.0.0 | 0 MB/iter（还页正常）|
   | 7.1 | 0 MB/iter |
   | 7.2.0 / 7.2.2 / 7.2.4 | **256 MB/iter（整块不还）** |

   不导出的对照组在每个版本都平 → 回归就特定于"被 IPC 导出过的块 + hipFree"这条路,引入于 7.1 → 7.2.0 之间。
7. **端到端确认**(真 Qwen3-30B-A3B-FP8 colocate,MI350X,同模型同配置只换 ROCm 版本):
   - 7.2:after-offload floor 131 → 172 → 188 GB,step 7~8 OOM。
   - 7.0:floor 稳定 ~8.4GB,before-offload ~42-43GB,跑过 step 10+ 无 OOM。
8. **源码定位**。ROCr(ROCR-Runtime,现已并入 `ROCm/rocm-systems` monorepo)逐函数 diff 7.1.1 ↔ 7.2.0,clr/hipamd 那层字节级相同(排除),回归在 ROCr。

---

## 3. 根因(确切定位)

文件:`runtime/hsa-runtime/core/runtime/runtime.cpp`
引入回归的 commit:**`2f384538f8b3321e13d17bc0a5c6b5010075f3e5`「rocr: Make IPC Handles Unique (#795)」**(2025-10-16)。`git tag --contains` 证实:不在 rocm-7.1.1,首次出现在 rocm-7.2.0,之后 7.2.x 全带。

机制(一个登记 / 注销不对称的引用泄漏):
- **登记**(`IPCCreate`):7.2 为了给 IPC handle 写唯一 metadata,导出方对自己的 dmabuf 调了一次 `amdgpu_bo_import`,把拿到的 BO 句柄存进 `allocation_map_[ptr].ldrm_bo`。`amdgpu_bo_import` 会在内核给底层 BO 加一个引用。
- **注销**(`FreeMemory`):free 时**只清了 metadata,从没调 `amdgpu_bo_free(ldrm_bo)`**。那个 import 引用一直挂着 → 即使 kfd 侧 `hsaKmtFreeMemory` 释放了虚拟地址,内核因为这个 libdrm 引用**不回收底层物理页** → 每次漏一整块。
- 佐证:整个文件里 `amdgpu_bo_free` 只在错误清理和导入方 detach 路径出现,**导出方的 `ldrm_bo` 在 free 路径上没有对应的 release**。
- 与"裸 VMM(`hipMemUnmap`+`hipMemRelease`)在 7.2 仍正常还页"自洽:VMM 走 `hsa_amd_vmem_*` 另一套代码,不碰这条 legacy IPC 路径。

---

## 4. 修复(上游已经修好)

`ROCm/rocm-systems` 的 **develop 分支已经重构修复**了这块:整个 IPC handle 机制从 libdrm `amdgpu_bo_*` 换成 thunk `hsaKmt*`,导出方拿到 BO handle 后**立即释放**:

```cpp
// Release the imported BO handle immediately after setting metadata.
HSAKMT_CALL(hsaKmtMemHandleFreePreserveMetadata(res.buf_handle));
```

`FreeMemory` 里也明确注释:导出方(IPCCreate)的 BO handle 在 metadata 校验后立即释放、不再挂在 allocation 上。所以**不需要我们再提 patch**,上游已经修了(而且做得比"补一行 free"更彻底)。这个修复目前只在 develop,会进下一个 ROCm release。

---

## 5. 哪些版本能用 / 不能用

| ROCm 版本 | 能否用 |
|---|---|
| 7.0 / 7.1 | ✅ 能用（回归之前；7.0 已端到端验证 30B FP8 跑过 step 10+ 无 OOM）|
| 7.2.0 / 7.2.1 / 7.2.2 / 7.2.3 | ❌ 有 bug |
| **7.2.4(当前最新发布版)** | ❌ 仍有 bug（直接实测 = 256 MB/iter）|
| develop（未发布）→ 下一个 release | ✅ 已修复 |

实操建议:**现在就用 ROCm 7.0 / 7.1**;升级救不了(最新 7.2.4 仍漏);若必须留在 7.2 大版本,可请求 AMD 把 develop 的修复 backport 到 7.2.x patch。

---

## 6. 这个 PR 里的改动

这个 PR 带的是最初、最简的那版缓解:**ROCm 上跳过 `destroy_process_groups()`、跨 offload 保活 communicator**(`reloadable_process_group.py`,ROCm-gated,CUDA 路径不动)。它能在 7.2 上挡住主漏,但代价是 communicator 常驻、不还给 offload。

> 这是 ROCm 7.2 回归的临时止血,不是真正的修复。真正的修复在 ROCm 本身(见上),或直接用 7.0 / 7.1。

相关的另外两个 stopgap(同样只为被迫留在 7.2 的人):
- 权重传输 buffer 复用(miles 侧)
- RCCL communicator buffer 复用 cache(rocm-systems 侧)

最小复现脚本 `churn_ipc.cpp` 和版本对照见排查记录。
