# DSv4-4layer FP8 训练 bring-up（ROCm / MI355X / gfx950）

目标:把 DeepSeek-V4-Flash-FP8-4layer(pipeline-only sanity)的 **FP8 训练**在 AMD ROCm 上跑通。验收 = 训练 step 真迭代、不报错、不 NaN。

## 环境

| 项 | 值 |
|---|---|
| 运行镜像 | `miles-rocm700-mi35x-dev-20260629`(本地 build,PR #1506;base `rocm/sgl-dev:v0.5.14-rocm700-mi35x-20260627` 自带 aiter `7d604afe5`(含 fused_clamp_act_mul)+ sglang fork + sgl_kernel + click 8.2.1),gfx950。⚠️ 旧镜像 `dev-20260627`(base 自带 aiter `a6bb49937` 缺该模块)→ silu blocker,已弃 |
| docker run | `--device /dev/dri --device /dev/kfd --group-add video --ipc host --shm-size 128G --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt label=disable -v /mnt/data/data/hai:/workspace -v ~/.cache/huggingface:/root/.cache/huggingface ... sleep infinity`(`/workspace/models` 内三件套 fp8 27G + bf16 52G + torch_dist 52G + `/workspace/datasets`) |
| ⚠️ 重建坑 | 新镜像 `dev-20260629`(PR #1506)已 bake 掉大部分旧坑(click 8.2.1 / sglang fork `sglang-miles-dsv4-rocm` / 对的 base aiter `7d604afe5`)。**仍需:① 别加 `--network host`**(同机别的容器跑 ray,host 网络下两个 ray head 撞 6379/8265 → `ray start --head` 失败)。**② `/root/miles` checkout `wip/dsv4-rocm-tile-kernels-import-guard`**(build 默认 `MILES_COMMIT=main`,缺 actor 修 E2/E24/E25/E26;浅克隆 `git fetch --depth=1 <miles-wip> <branch>` + `checkout -B`)。env 不用手 export —— 训练脚本 `scripts/amd/run-deepseek-v4-flash-fp8-4layer-amd.sh` 自带全套。⚠️ 旧镜像 `dev-20260627` 另需手补 click/sglang fork,且 base aiter `a6bb49937` 缺 `fused_clamp_act_mul` → silu blocker(见 E16),**已被 dev-20260629 取代**。容器随 SLURM 24h 超时回收(`dsv4-fp8-v14/v15` 都这么没的),长跑前先 `salloc` 占节点。 |
| miles | 容器内 `radixark/main`;改动在 `XinyuJiangCMU/miles@wip/dsv4-rocm-tile-kernels-import-guard` |
| sglang | fork `XinyuJiangCMU/sglang@sglang-miles-dsv4-rocm`;Dockerfile fetch 该 branch |
| TE | `JessicaJiang-123/TransformerEngine@amd-qwen3-30b-a3b-fp8-dev`(blockwise fp8) |
| 模型 | `Pinaster/DeepSeek-V4-Flash-FP8-4layer` |

## 怎么跑(最近成功指令,每次跑通后覆盖本节)

> 训练脚本进 repo:`scripts/amd/run-deepseek-v4-flash-fp8-4layer-amd.sh`(模仿 `scripts/amd/run-qwen3-4B-amd.sh`:
> pkill 进程名清理头 + RAY/ROCm env + 全套 DSv4 rollout env + tilelang indexer + run_deepseek_v4.py full-train)。
> 全套 rollout env 取自 `docker/Dockerfile.rocm` line 156-207(= sglang AMD CI `COMMON_ENV_VARS` + DSv4 E18/E19/E20)。

- 节点/容器: `amd-mi350x-ses2-1` / `dsv4-fp8-v17`
- 镜像: `miles-rocm700-mi35x-dev-20260629`(本地 build,PR #1506:base 升到 `rocm/sgl-dev:v0.5.14-rocm700-mi35x-20260627` 自带 aiter `7d604afe5` 含 `fused_clamp_act_mul`;sglang fork + sgl_kernel + click 8.2.1 已 bake;miles 仍需 checkout wip 取 actor 修 E2/E24/E25/E26)
- ⚠️ 路径: 模型在容器内 `/workspace/workspace/models/...`(double workspace,mount `/mnt/data/data/hai:/workspace` + host 模型在 `workspace/models/`)
- 起训练:
  ```
  docker exec -d dsv4-fp8-v17 bash -lc 'bash /workspace/workspace/dsv4-miles-amd/miles-dsv4-tile-guard/scripts/amd/run-deepseek-v4-flash-fp8-4layer-amd.sh > <log> 2>&1'
  ```
- 停训练(切配置/重跑前): 脚本清理头 `pkill -9 sglang; ray stop --force; pkill -9 ray; pkill -9 python`(**按进程名,bash 不 self-match**)。**别 docker restart**(没必要);**别 `pkill -f run_deepseek`**(self-match 杀自己 shell → 杀残 → zombie + 卡 VRAM)。
- 验收(v17 torch-fn 实测通过,`train_v17_01`): rollout 256/256 → logprob `-1.82/-5.23`(有限不 NaN) → train step(`Timer 208.7s`、`actor_train_tflops 13.6`、`tok/s 8235`) → update_weights 15.9s → 进第 2 rollout。`train/loss=0.0` 是 4-layer toy 正常。
- aiter 根因(本轮关键): 旧镜像 `dev-20260627` 的 base 自带 aiter `a6bb49937`(rocagents-fix 侧分支)**缺** `fused_clamp_act_mul`(PR #3057/v0.1.14 才引入)→ 与 base 自带 sglang(本就 import 该模块)配不上 → shared_experts silu 掉到 CUDA `silu_and_mul_clamp` 的 `cuda_fp8.h` JIT,gfx950 编不了 = silu blocker。PR #1506 换 base(aiter `7d604afe5` 有该模块)修根;`SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL` 走默认 True(aiter 路径),旧的 `=0` 歧路已弃。详见 E15/E16 更正。

## 核心结论:build 配方 + 一套 rollout env,不是一堆 bug

**① build 配方**(`docker/Dockerfile.rocm`):
- `FROM rocm/sgl-dev:v0.5.14`,**别 uninstall+重编 sgl_kernel** —— base 自带 DSv4 op(`dsv4_fused_q_indexer_rope_hadamard_quant` 等),重编反而丢。
- 末尾 re-pin `click==8.2.1`(中途被 requirements 升到 8.4.2 会破 ray 2.44 CLI;8.2.1 是 ray+typer 唯一兼容点)。
- sglang 段 `git fetch ${SGLANG_REPO} ${SGLANG_BRANCH}` 拉 fork branch `sglang-miles-dsv4-rocm`(= sglang-miles + E15 修),只装 python 层(不重编 sgl-kernel)。
- TE 用 blockwise-fp8 分支。

**② 一套 rollout env**(Dockerfile ENV,镜像级) —— 以 sglang AMD nightly CI `test/registered/amd/test_deepseek_v4_flash_fp8.py` 的 `COMMON_ENV_VARS` 为底,加上 DSv4 训练 rollout 实测需要的几个。作用:把 DSv4 的 sparse-attn indexer / MHC / MoE 从 CUDA-only(`deep_gemm` / `<cuda/ptx>` / tilelang / `<cuda_fp8.h>`)切到 aiter/triton:

| env | 作用 | 来源 |
|---|---|---|
| `AITER_INDEXER=false` + `FP8_PAGED_MQA_LOGITS_TORCH=1` | indexer paged-MQA-logits → torch fn(triton 3.4 下 aiter gluon/legacy 都不可用,见 E20;配 sglang fork 3 处修) | E20 |
| `USE_TOPK_V2=false` + `USE_JIT_INDEXER_METADATA=false` | 跳过 `<cuda/ptx>` topk JIT | CI |
| `DEEPGEMM_HC_PRENORM=false` + `USE_TILELANG_MHC_PRE/POST=false` | MHC → `aiter.ops.mhc` | CI |
| `USE_FUSED_COMPRESS=true` + `_TRITON=true` | MoE 激活/压缩 → triton | CI |
| `USE_COMPRESSOR_V2=false` | c4 indexer compressor → v1 triton(避 c4_v2.cuh,见 E19) | E19 |
| `AITER_CONFIG_GEMM_*`(6 个) + `AITER_CONFIG_FMOE` 各指 default 单文件 | 避 aiter config-merge baton 死锁(见 E18) | E18 |
| `HACK_FLASHMLA_BACKEND=triton` / `USE_TILELANG_INDEXER=false` / `MULTI_STREAM=false`×2 / `AITER_BF16_FP8_MOE_BOUND=0` / `FP8_WO_A_GEMM=false` / `DSV4_FP4_EXPERTS=false` | 其余 ROCm 路径 | CI |

## blocker 台账(E10-E23 rollout 全通 + E2/E24/E25/E26 actor)

完全解决/绕过的一行,关键的多写。

- **E10** ✅ sgl_kernel 缺 dsv4 op → base v0.5.14 自带、不重编。
- **E11** ✅ ray `not a valid Sentinel` → `click==8.2.1`。
- **E12** ✅ indexer `No module deep_gemm`(metadata.py PagedIndexerMetadata,CUDA-only)→ `FP8_PAGED_MQA_LOGITS_TORCH=1` 让 deep_gemm_metadata=None。(最初用 `AITER_INDEXER=true` 走 aiter,后因 E20 改 `AITER_INDEXER=false` 走 torch fn。)
- **E13** ✅ topk_v2 `<cuda/ptx> not found` → `USE_TOPK_V2=false`。
- **E14** ✅ mhc `NameError deep_gemm` → `DEEPGEMM_HC_PRENORM=false`+`TILELANG_MHC=false` → aiter.ops.mhc。
- **E15** ✅(已解,fork+PR) MoE down_proj 收 fused_clamp fp8 tuple `(x_fp8,x_scale)`;sglang-miles 的 true_on_policy(sglang upstream PR 26359)给 RowParallelLinear 加了 `should_use_tp_invariant_row_linear(input_parallel.shape[-1])`,在 tuple 上读 `.shape` 炸(sglang main 无此模块)。根因:`matmul_tp_inv` 仅 bf16/fp16/fp32,fp8 row-linear 必走 `quant_method.apply`(`Fp8LinearMethod` 本就拆 tuple)。修:`if not isinstance(self.quant_method, Fp8LinearMethod) and should_use_tp_invariant_row_linear(...)`(读 .shape 前短路)。**fork+pin(非 patch):** [`XinyuJiangCMU/sglang@sglang-miles-dsv4-rocm`](https://github.com/XinyuJiangCMU/sglang/tree/sglang-miles-dsv4-rocm),PR(branch sglang-miles-dsv4-rocm)(commit 链:tuple-skip → Fp8LinearMethod gate → torch fn path 3 修);Dockerfile fetch 该 branch。
- **E16** ✅(已解,换 base aiter;⚠️ 早前"FUSED_COMPRESS 自然避开"的记述不成立) shared_experts(`DeepseekV2MLP.forward`,deepseek_v2.py:394)的 silu 掉到 CUDA `silu_and_mul_clamp`,JIT 编 `<cuda_fp8.h>` gfx950 失败。根因=**aiter 版本**:`use_fused_clamp_act_mul = _is_hip and SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL`(environ 默认 True)本应走 aiter `fused_clamp_act_mul`(triton,不碰 cuda_fp8.h),但该模块 PR #3057(2026-05-12)才进 aiter、v0.1.14 起才有;旧镜像 `dev-20260627` 的 base 自带 aiter `a6bb49937`(rocagents-fix 侧分支)**缺它** → 只能 `=0` → 掉回 CUDA silu。两难:`=1` 缺 aiter 模块 ModuleNotFound、`=0` 撞 cuda_fp8.h。**真解=补对 aiter**:换 base 到 `rocm/sgl-dev:v0.5.14-rocm700-mi35x-20260627`(aiter `7d604afe5`,= sglang `rocm.Dockerfile` 钉的 commit,含该模块),`FUSED_CLAMP_ACT_MUL` 走默认 True(PR #1506,= miles `[AMD] update ROCm sglang base image`)。`=0` 是已弃歧路。
- **E17** ✅(已绕) cuda graph capture 在 ROCm colocate hang(GPU 0%、停在 `Capture cuda graph begin bs=256`);aiter/triton kernel 不兼容 cuda graph capture。绕:`miles/utils/arguments.py` colocate 块 `if is_hip() and not args.sglang_disable_cuda_graph: args.sglang_disable_cuda_graph = True`(经 `--sglang-*`→ServerArgs 自动映射,覆盖 normal/external、限 colocate+ROCm、NV 不影响)。
- **E18** ✅(已解) aiter `get_config_file`(jit/core.py) glob `model_configs/*.csv` 后 `update_config_files` merge 写 `/tmp/aiter_configs`,走 FileBaton mp_lock;colocate 8+ engine 进程抢同一 baton 死锁(持有者卡 do_wait 不 release、全员 wait;清 lock/加大 watchdog 都治标)。**非 online tune**(AITER_ONLINE_TUNE 默认 0)。根因短路:`update_config_files` 开头 `if len(path_list)<=1: return` —— `AITER_CONFIG_GEMM_*`(6 个)+`AITER_CONFIG_FMOE` 各指 default 单文件,走单路径跳过 merge;缺的 shape 自动用 default kernel。(train26 实测 gemm baton 绕过、train28 实测 fmoe baton 绕过。)
- **E19** ✅(已绕) rollout forward `tvm.error.InternalError: Tensor match failed Tensor<N,4,512>` @ `c4_v2.cuh`(`indexer.py forward_c4_indexer`)。compressor_v2 的 c4_v2 kernel `TensorMatcher` 期望 kv_input 2D,caller 传 3D。绕:`SGLANG_OPT_USE_COMPRESSOR_V2=false` 走 compressor v1 的 triton `_c128_compress_*` kernel。(train27/28 实测 c4_v2 不再撞。)
- **E20** ✅(已绕,torch fn 定案) DSv4 indexer paged-MQA-logits 在 ROCm 三条路:① aiter AOT gluon(`AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1`)需 `triton>=3.6`,容器是 `pytorch-triton-rocm 3.4`(ROCm7.0 配套)→ 编译 `ValueError: CDNA_VERSION is not in list`(E21),死路;② aiter legacy → `assert KVBlockSize==1` 与 DSv4 page 不符;③ **torch fn `fp8_paged_mqa_logits_torch`**(`AITER_INDEXER=false`+`FP8_PAGED_MQA_LOGITS_TORCH=1`),纯 torch、不依赖 triton 版本。**定案走 ③**(agent 调研:升 triton≥3.6 要连 torch 升、炸整镜像;v2 c4_v2 是 tvm 语义 ABI 不匹配,均比 torch fn 重)。torch fn path 配 sglang fork 3 处修(都在 fork branch sglang-miles-dsv4-rocm):E20=torch fn 开头 squeeze seq_lens(caller forward_c4_indexer:551 给 deep_gemm path unsqueeze 成 2D);E22=indexer.py:586 `getattr(core_metadata,"c4_sparse_raw_indices",None)`(DSV4AttnMetadata 无此 field,非 capture/hisparse 本该 None);E23=`fused_compress_triton._c128_decode_kernel` 的 kv_input/score_input 两处 `tl.load` 指针补 `(slot_offs*0)[:,None]` 显式广播(triton 3.4 不自动把 (1,D) 广播到 mask 的 (BLOCK_S,D);与 page_size 无关,128 是 c128 压缩比)。train34 实测 **prefill+decode 全过、rollout forward 整链通**。
- **E2** ✅(已解,torch mhc/quant) rollout 全通后 Megatron actor init(`MegatronTrainRayActor.init`→`spec_utils.py:55 import_module`)撞 `No module named 'tile_kernels'`→spec=None→`specialize for NoneType`。DSv4 plugin(`miles_plugins/models/deepseek_v4/ops/qat.py`/`hyper_connection.py`)硬 import CUDA/SM90-100-only 的 `tile_kernels`(gfx950 装不了)。修:纯 torch autograd 重写 —— `qat.py` 的 `per_token_cast_back`(fp8 blockwise dequant)+ `hyper_connection.py` 的 7 个 mhc 函数(mhc_pre_norm_fn/split_mixes/apply_mix/head_compute_mix/post/pre_big_fuse/sinkhorn);数学从 sglang `srt/layers/mhc.py` 的 TileLang 参考反推。**全 torch** 因 `aiter.ops.mhc` 只 forward 无 backward、actor 训练需反传。自测 forward+backward finite、mhc_post vs 显式 loop diff 4.8e-7、sinkhorn doubly-stochastic。train35 实测 actor init 过、update_weights(11s)、rollout generate 都通。**提速待办(用户要求,torch mhc 慢)**:mhc forward 换 aiter(`mhc_pre`/`mhc_post`/`mhc_pre_big_fuse`)+ backward 手写 autograd.Function(aiter 无 bwd kernel);sanity 通过后做。
- **E24** ✅(已修通,loop 2026-06-30) generate 完→train step 撞 `ValueError: rollout_routed_experts is required in rollout_data for replay`(`replay_data.py:53`)。`--use-rollout-routing-replay`(R3)需 rollout 回传 MoE routed_experts(`sglang_rollout.py:179` payload `return_routed_experts=True`→收 `meta_info["routed_experts"]`)。**真根因(loop R42–R47 实测铁证,⚠️ 推翻本条旧版及下方各段的"aiter producer CUDA-only / deep aiter / 超本阶段"误诊)**:capturer/producer/readout 三环全平台无关且已接好(GEN_DEBUG 探针在 http_server `/generate`、RE_DEBUG 在 `batch_result_processor.py:114`,双探针证实);真凶是 **Rust `sglang_router` v0.3.2(pin `openai-protocol` 1.0.0,其 `GenerateRequest` 无 `#[serde(flatten)]` catch-all——1.8.0 才加)在 /generate 转发时 typed-deserialize + re-serialize,静默丢掉 `return_routed_experts` 字段** → rollout payload 设了 True 但 http_server 收到 `obj.return_routed_experts=False`(GEN_DEBUG 126 次全 False)→ scheduler `Req` False → `batch_result_processor.py:114 if not req.return_routed_experts: return` 提前返回 → `meta_info` 无 routed_experts → 崩。**跟 ROCm/aiter/kernel 零关系**。**修法**:ROCm 上 R3 改走 miles python router(`miles/router/router.py` 是 raw-bytes 反向代理、不反序列化,字段原样透传)——`run_deepseek_v4.py` enable_r3 块 `is_hip()` gate 加 `--use-miles-router`(NV 不动、仍走 Rust router)+ 删原 ROCm disable guard。**1 个 flag、零重 build、不动容器**。**验证**:DSv4-Flash-FP8-4layer 3 rollout + 3 train step + 0 崩 replay(miles `b9047e3`)。fallback(若必须留 Rust router):把 `openai-protocol` bump 到 ≥1.8.0 重 maturin build gateway(高成本)。
- **E25** ✅(已绕) compute_log_prob 撞 `NotImplementedError: fuse_wgrad_accumulation (gradient_accumulation_fusion) is not yet supported in the ROCm blockwise grouped FP8 path`(TE `grouped_linear_blockwise.py`,MoE grouped linear)。修(message 自带解):`run_deepseek_v4.py` fp8_training 段加 `is_hip()` guard `--no-gradient-accumulation-fusion`——wgrad 返回 plain gradient,Megatron DDP post-hook 累加进 fp32 main_grad(bring-up 数值等价;NV 保留 fusion)。train37 实测 compute_log_prob 过。
- **E26** ✅(已绕) train step compressor wkv `linear_bf16_fp32` 的 `torch.mm(bf16,bf16,out_dtype=fp32)` ROCm hipblas 不支持(`gemm input bf16 output float not supported for ROCm`)。修:`precision_aligned_ops.py _BFloat16LinearFP32Func.forward` 加 `torch.version.hip` guard 用 fp32 matmul(精度 ≥ cublas bf16-in/fp32-accumulate;backward 本就 fp32;NV 保留 bf16 快路径)。train38 实测 rollout/generate 整链过。

## 当前状态:✅ 端到端 fp8 训练 sanity 验收通过(train38)

train38 完整跑通一步 RL 训练并进入第 2 步循环,**训练 step 真迭代、不报错、不 NaN**:
- **generate** ✅ 256/256(rollout forward 全链,E10-E23)
- **compute_log_prob** ✅ `Timer log_probs end 47.5s`,`rollout/log_probs=-1.71`、`rollout_log_probs=-1.59`(有限、不 NaN;E25 wgrad-fusion 关生效)
- **train step** ✅ `Timer train end 163.7s`,`actor_train_tflops=15.47`、`tok_per_s=9359`(真在算;E2 torch mhc + E26 fp32 gemm 生效)
- **update_weights** ✅ 第 2 次 10.9s → **第 2 个 rollout step 已开始**(pipeline 真迭代起来)

全程 `check_for_nan_in_loss_and_grad=True` 没 raise。fp8 blockwise e4m3 + `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`。

**待办(性能/严格性,非 sanity 必需)**:
- ① **mhc 提速**:actor_train 一步 163.7s 大头在 torch mhc(用户已指出慢)。提速 = mhc forward 换 aiter(`mhc_pre`/`mhc_post`/`mhc_pre_big_fuse`)+ backward 手写 autograd.Function;prenorm-GEMM/norm 借 Primus-Turbo,mixing/residual/sinkhorn 主体手写(见全地图:MHC backward 是 actor 侧唯一要自己写的)。
- ② **R3 严格 on-policy** ✅(已修通,loop 2026-06-30):真根因不是 aiter routing(误诊),是 Rust `sglang_router` 转发丢 `return_routed_experts` 字段;ROCm 改走 miles python router(`--use-miles-router`,is_hip gate)即修通,3 rollout+3 train step+0 崩(miles `b9047e3`)。详见 E24。
- ③ **indexer 提速(torch fn 慢,用户要求)**:当前 indexer = torch fn `fp8_paged_mqa_logits_torch`(E20,正确但慢:全 KV upcast fp32 + padded dense bmm + 多趟 elementwise)。两条提速路:
  - **(a) tilelang stopgap** —— 可立即试,但 **off-blessed-path、仅作 gated 实验,勿直接信其训练 metric**。AMD 在 #24933(2026-05-18)曾以 tilelang 为 indexer 主路,11 天后 PR 26662(2026-05-29,Thomas Wang)主动切到 aiter;gfx950 零 CI 验证、kernel 无 gfx95 tuning(hardcoded NUM_CU=256)、唯一正确性对比 PR 24989 在 CUDA 机上;dtype 崩 issue 27124 限 gfx94x(gfx950 likely 不触发,`is_fp8_fnuz=False` 选对 e4m3fn)。
  - **(b) aiter proper(目标,AMD blessed)** —— aiter gluon paged-MQA(快)需 `triton>=3.5`,容器 triton 3.4 死(E20/E21);aiter non-gluon `pa_mqa_logits` 是 block==1(`assert KVBlockSize==1`,不吃 DSv4 page=64)→ 需补 per-c4-token index 展开 + packed [K|scale] 地址/scale 指针重建(medium,triton 3.4 可跑)。或等 ROCm 7.2.5+/7.3(aiter gluon 复活 + 修 7.2 IPC 泄漏)→ 直接用 blessed gluon。
  - **TODO**:
    - [x] 扒 PR 26662 → **结论:弃因是「性能」非「正确性」**。决策实在 PR 26208(`[AMD] Dsv4/pr2 compressor opt`,co-author 含 **`@HaiShaw`**):明写 "improves inference PERFORMANCE... high-performance fused paths... while maintaining numerical correctness checks" → 换 aiter 是为性能(gfx950 调优 fused path),tilelang 没被判错、只是比 aiter 慢。PR 26662 只是把 CI test 同步到该 main recipe(非决策点)。**含义**:tilelang 当 stopgap 更靠谱(仍比 torch fn 快、且大概率正确),但正确性只在 CUDA 验过(PR 24989)、gfx950 未验 + gfx94x 有 dtype 崩(issue 27124)→ 信训练 metric 前仍做 top-k parity(风险中等非高)。
    - [ ] (a) 试跑:`indexer.py` weights squeeze 后加 `if USE_TILELANG_INDEXER and not use_fp4: weights = weights.float()`(kernel 要 fp32,否则调用时大声崩)+ env `SGLANG_OPT_USE_TILELANG_INDEXER=1` 且**保留 `FP8_PAGED_MQA_LOGITS_TORCH=1`**(metadata=None,缺了回 E12;tilelang 在 fn dispatch 优先于 torch)+ cuda-graph 保持关。崩则 revert torch fn。
    - [ ] (a) 若跑通且要信训练结果:离线验 tilelang logits + **top-k 索引** parity vs 真 fp32 dequant ground truth(⚠️ 不是 vs torch fn —— torch fn 按 e4m3**fnuz** 解、本身偏~2x;拿它当基准会把对的修成错的)。
    - [ ] (b) aiter non-gluon + per-c4-token index 展开(medium effort,AMD-aligned 真·快修);或评估等 ROCm 7.2.5+/7.3 直接用 aiter gluon。

## 性能优化实测结论(2026-06-28):三面墙,torch-fn 是稳定绿配置

sanity 验收后转「性能优先」,把上面三个待办(mhc / indexer / rollout)逐个**实测**,三个角度全撞 ROCm 墙或 toy 不 transfer;随后试 R3 严格 on-policy 当时也判为撞 ROCm 墙(误诊为 routing producer 未接线)——⚠️ **loop 2026-06-30 已推翻并修通**:真根因是 `sglang_router` 转发丢字段、非 aiter producer,详见 E24 + 下方 R3 段。**结论(更新):perf 三靶仍受限于 ROCm/toy(torch-fn 全链 rollout 3:04 + train step 110.5s 是稳定绿配置);R3 已不再是墙——修通端到端绿。bring-up 阶段交付完成(端到端 fp8 训练 sanity 绿)。** 实测细节:

- **indexer→aiter(待办③):手写 aiter `_aiter_fp8_paged_mqa_logits` 真实引擎越界崩,已搁置。** offline parity 过(单序列 aiter-vs-GT rel 2.75e-6、top-k recall@512=1.0),但真实引擎**批量多序列 prefill(56 序列/16384 tok)GPU Memory access fault**——`kv_indices`(`b*max_blk+pos`,page=64)在批量 page-table 布局下越界,单序列 parity 没覆盖此 case。代码留在 sglang fork 但 gated 在 `SGLANG_OPT_USE_AITER_INDEXER=false`(默认关),后续可离线建批量 repro 调索引。另:indexer 只占 rollout 时间 <10%,即便修好提速上限也低。**注意 on-policy abs_diff=0.32 是 fp8 固有量化噪声(actor-fp8 vs sglang-fp8 logprob),非 indexer fnuz 偏 —— top-k 对全局缩放不敏感,换 aiter 不改 abs_diff(train39 实测仍 0.32);原「换 aiter 顺手修 0.32」假设已证伪。**
- **mhc→aiter(待办①):microbench 实测仅占 train step ~5%,非大头,已降优先级。** standalone microbench(`loop-dsv4-fp8-train/mhc_microbench.py`,真实 shape hidden=4096/hc_mult=4/sinkhorn=20/4层)测得 mhc fwd+bwd=**15-17ms/microbatch 且几乎不随 token 变(512→2048: 15.3→16.7ms)= kernel-launch 开销 bound,非 compute bound**(20 迭代 sinkhorn + 数百 fp32 小算子)。按 global_batch 256/mbs 1/DP 1→~256 microbatch、actor_train 81.7s→~319ms/microbatch,mhc 含 recompute~22ms → **mhc 仅 ~5-7%(任何合理 microbatch 数下最多 ~9%)**。原「mhc 是 163.7s 大头」是没实测的假设,被推翻。
- **rollout decode(最大块 184s):唯一大杠杆 cuda-graph 被 ROCm 硬墙挡死(E17 深化)。** 从 train42.log 拆 rollout:decode 是绝对大头(~4096 步自回归、~2975 tok/s/120-144 并发,prefill 很快过)。decode launch-bound,教科书级解药是 cuda-graph,但 **E17 根因 = ROCm colocate 下 cuda-graph capture 直接 hang(GPU 0%、卡 `Capture cuda graph begin bs=256`),aiter/triton kernel 不兼容 HIP graph capture** → 重开撞回 hang。这是 ROCm 硬墙,非本项目能短期解。
- **R3 严格 on-policy(待办②)✅ 已修通(loop 2026-06-30,⚠️ 推翻本段旧"aiter producer 未接线"误诊)**:train44(2026-06-28)强开崩 `rollout_routed_experts is required` 当时归因"喂 capturer 的 producer 在 aiter/ROCm topk 没接线、deep sglang/aiter、超本阶段",**全是错的**。loop R42–R47 双探针(GEN_DEBUG 在 http_server `/generate`、RE_DEBUG 在 `batch_result_processor.py:114`)实测:capturer/producer/readout 三环全平台无关且已接好,真凶 = **Rust `sglang_router` v0.3.2(pin `openai-protocol` 1.0.0、无 `#[serde(flatten)]` catch-all)转发 /generate 时 typed-deserialize+re-serialize 丢掉 `return_routed_experts`**(rollout payload 设了 True、http_server 收到 obj.return_routed_experts=False、126 次全 False)→ scheduler Req False → `batch_result_processor.py:114 if not req.return_routed_experts: return` 提前返回 → meta_info 空。修法:ROCm R3 改走 miles python router(raw-bytes 透传,`--use-miles-router` is_hip gate)+ 删 disable guard,1 flag 零重 build,3 rollout+3 train+0 崩(miles `b9047e3`)。详见 E24。**教训(更新+反转)**:旧"R3 不可行"恰是"读代码/类比 `init_indexer_capturer` 的 CUDA-only 模式推断、没核到字段层"的反面教材——ROCm 受限结论同样要实跑+探针验到根因(打 flag 看 True/False),别停在似是而非的类比;否则会像这次一样把一个 1-flag 的 router bug 误判成"deep aiter 工程、超本阶段"而白白放弃。
- **运维根因(供后人避坑)**:① GPU memory fault 会把 core dump(`gpucore.*`,8 engine 各 ~16-105G)吐到进程 cwd `/root/miles`,**几秒填满 `/`**(`No space left`),清 `gpucore.*` 即恢复;debug 越界类 bug 别跑全训练,改 standalone repro。② env:旧镜像 `dev-20260627` 没 bake E18-E20 的 ROCm indexer env → 要 launch 时 export;新镜像 `dev-20260629` + 训练脚本 `scripts/amd/run-deepseek-v4-flash-fp8-4layer-amd.sh` 自带全套 env(取自 Dockerfile.rocm line 156-207),不用手 export。③ 停 colocate 训练:`pkill -9 sglang; ray stop --force; pkill -9 ray; pkill -9 python`(**按进程名,bash 不 self-match**),VRAM 自然回收到 ~0。**别 `pkill -f run_deepseek`**(命令行含该串 → self-match 杀掉自己的 shell → 杀残 → ray actors 变 zombie + KFD 卡住 VRAM);**别 docker restart**(进程名 pkill 杀干净后 GPU 句柄就放,没必要)。只有真 KFD 泄漏(进程全清光 VRAM 还卡 100GB+)才 restart 兜底。

## Gluon kernel 提速实测(2026-06-28):aiter gluon blockscale GEMM 候选 = 不成

接「性能优先」方向,试 PyTorch TokenSpeed-kernel blog 的路子——用 Gluon 写 kernel 对标 AITER 求提速,先挑 a8w8 blockscale fp8 GEMM。**结论:aiter 的 gluon 版 `gemm_a8w8_blockscale` 数值正确(cos=1.000、relerr_vs_ck=0)但比 aiter 的 CK/asm 生产 kernel 慢 2–8×**(真实 DSv4 形状,大 M geomean gl/ck=0.28x、gl/asm=0.42x;小 M gl/ck=0.13x;K=4096 最差 ~5×),无任何形状赢、对纯 triton 也基本不赢。根因:这版 gluon kernel 没用 blog 的关键技法(非硬件 mfma_scaled、无 async_copy、非 persistent),是早期未优化 kernel;blog 的胜绩在 attention/MoE 是别的 gluon kernel。落地分析:训练走 TE 自带 vendored triton kernel(`transformer_engine/.../blockwise_fp8_gemm.py`,不碰 aiter)、DSv4 稠密形状不在 sglang tuned 白名单(走 CK)+ 服务系统 triton 3.4(<3.6 跑不了 gluon,且为保底)→ gluon 落不到本训练/本容器服务,价值=kernel 表征(生态/未来可落)。完整数据表 + 方法学 + 运维坑(aiter baton 锁死等、SLURM 24h 分配超时回收容器)详见 `docs/gluon-gemm-bench-findings.md`。

## tilelang DSv4 indexer(2026-06-29):gfx950 可用,端到端 sanity 通过

用 **tilelang indexer** 跑 V4 4-layer training(替代 E20 定案的 torch-fn 绿底)。研究 + 实测:

**AMD 历史**(sglang git):`#24933`(2026-05-18,kkHuang-amd)把 tilelang 做成 AMD indexer 主路(recipe = `SGLANG_OPT_USE_TILELANG_INDEXER=true` + `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` + `SGLANG_HACK_FLASHMLA_BACKEND=tilelang` + `SGLANG_OPT_DPSK_V4_RADIX=1`,GSM8K 0.948);`#26662`(2026-05-29)CI 翻回 aiter —— **纯 CI 整合,无 perf/正确性回归记录**。tilelang 是 eager bring-up 占位 kernel(`NUM_CU=256` 硬编、正确性只在 CUDA 验过 `#24989`、gfx94x 有 fnuz 崩 `#27124`),被退成 off-path。真路径 = `dsa/tilelang_kernel.py:1495 tilelang_fp8_paged_mqa_logits`(gfx950 用 e4m3fn)。

**gfx950 启用**:`bash launch_dsv4.sh tilelang <log>`(= `SGLANG_OPT_USE_TILELANG_INDEXER=true`,保留 `FP8_PAGED_MQA_LOGITS_TORCH=1` 作 metadata companion;weights 已天然 fp32)。dispatch 优先级 tilelang > aiter > torch。

**实测(容器 dsv4-fp8-v15)**:
- **kernel 冒烟**(`loop-dsv4-fp8-train/tilelang_indexer_smoke.py` 直调):gfx950 真 JIT 编译(hipcc,"Unsupported FP8 type in HIP codegen" 守卫没触发)+ 跑出 logits finite ✓。装的 tilelang(`0.1.7.post3+cuda` 标签误导)二进制实含 HIP/ROCm 后端(`CodeGenTileLangHIP` + ROCm runtime)。
- **端到端**(`train_tilelang3.log`):引擎 fired up → rollout 256/256 → actor update_weights → eval → **train step 真迭代到 step 19+(每步 ~2.8min),无崩无 NaN**。验收口径"train step 真迭代不崩不 NaN"用 **tilelang indexer** 达成(非 torch-fn);`train/loss=0.0` 是 4-layer toy 乱码 rollout(reward 0→advantage 0)的正常结果。

**待办**:① 信训练 metric 前做 **top-k vs fp32 ground-truth parity**(⚠️ 不能拿 torch-fn 当 oracle —— gfx950 上 torch-fn `.view(e4m3fnuz)` 而 K cache 写的是 e4m3fn,格式不一致;按写入格式 e4m3fn dequant 算 GT)。② tilelang vs torch-fn/aiter rollout 提速对比。③ 均需先 `salloc --no-shell -w <node> --time=...` 占住节点再跑(本轮 19 步后容器被 SLURM 节点回收 = 第 3 次同款,不占节点必被抢)。

## 24h 目标 + TODO(先对再快)

sanity liveness 已过(train38),接下来按 **correctness before speed** 推进。**验收锚定 AMD TE blockwise FP8 PR(647)的事实标准**(无公开的真·NV DSv4 逐张量数字 → 要拿得有 NV 机):fp8-vs-bf16 **relerr ≤ 0.04**、on-policy train-vs-rollout logprob **abs-diff ≤ 0.04**(≥10 步不上升)。验收用 miles 现成工具:`tests/e2e` Comparator(`--diff-threshold 0.002 --logprob-threshold 0.06`)+ debug_dump 的 `train_rollout_abs_diff`。长期参考(非 4-layer gate):DeepSeek-V3 整轮 loss relerr <0.25%。

### Stage 1 — liveness(基本达成,补确认)
- [x] 清 actor ROCm 算子限制(E2 torch mhc/quant、E25 wgrad-fusion、E26 bf16→fp32 gemm),train step 真迭代不 NaN(train38)
- [ ] 连续 **≥10 步** loss/grad 全 finite、check_for_nan 不触发(现 1–2 步,再观察)

### Stage 2 — 数值对齐 bf16(真 bar,24h 主目标)
- [ ] **先跑 bf16 golden**:`fp8_training=False` 同 seed/同数据,dump 逐层 fwd / logprob / grad —— **不可跳的第一子任务**(无 golden 则 Stage 2 无从验)
- [ ] fp8 run dump → Comparator 对照 golden:端到端 **relerr ≤ 0.04**(0.04–0.05 报警并排查)
- [ ] **train_rollout_abs_diff ≤ 0.04** 且 ≥10 步不上升;`train_rollout_kl` ~1e-3 量级、无上趋势;importance ratio exp(Δlogprob) 围绕 1.0
- [ ] 比对前断言两边 quant 粒度一致(1×128 act / 128×128 weight)、fp8 侧 FP32 scale(`NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`,非 UE8M0)—— **recipe 差异 ≠ bug**
- [ ] (compressor 走 TP 时)backward FP32 all-reduce —— DeepSeek V4 的硬性正确性要求(= 我们的 E26 方向),bf16 跨 TP 累加会偏

### Stage 3 — 提速(⚠️ 已实测,见上「性能优化实测结论」:三个靶全撞 ROCm 墙/toy 不 transfer,torch-fn 为稳定绿配置,perf 不死磕)
- [ ] **mhc 提速 — 两条现成 kernel 路,都要测 gfx950(替当前 torch),先对再用**:
  - **(a) tilelang forward**:sglang `srt/layers/mhc.py` 的 tilelang forward kernel(`mhc_pre`/`mhc_post`/`hc_split_sinkhorn`/`mhc_fused_post_pre`)。tilelang 有 HIP 后端、indexer 已证明 gfx950 可编。替 torch forward,backward 仍 autograd。现成测试 `tests/kernels/test_mhc_kernels.py`,**不用装额外包**(sglang fork 自带)。**gfx950 实测(loop R2)**: 8 pass(非 fused mhc_pre/post/sinkhorn)/16 fail(`mhc_fused_post_pre` 全撞 tilelang HIP 缺 `tl.get_lane_idx`;且 tilelang mhc 只 forward)→ 非 fused 可用、fused 撞墙、无 backward,**降为备选**。
  - **(b) Liger triton fwd+bwd(最有前途,连 backward 都有)**:Liger-Kernel PR #1065(merged 2026-03-05)= mHC fused **fwd+bwd** Triton kernel(`src/liger_kernel/ops/mhc.py` + `LigerMHC`,对应 arXiv:2512.24880,还带 `_ascend` 多后端)。triton cross-platform,gfx950 大概率直接能跑 → **不用 autograd、不用自己写 backward**。测法:Dockerfile/环境加 **`pip install liger-kernel`**,跑它的 `test/transformers/test_mhc.py` + `benchmark/scripts/benchmark_mhc.py`。**gfx950 实测 ✓(loop R3)**: liger 0.8.0(triton 3.4 满足)`test_mhc.py` **33 passed**(backward + forward match-ref + LigerMHC module + mini-LM output 全过)→ **fwd+bwd 在 gfx950 坐实可用**,正接入 `hyper_connection.py`(is_hip gate 替 torch)。
  - gate(两条都适用):接入训练前过 `torch.autograd.gradcheck` / 对 pure-torch autograd 版达 bf16 容差(1e-2);过不了回退 torch。
  - ⚠️ **不装 `tile_kernels`**(deepseek-ai,CUDA/SM90-100-only,gfx950 装不了 = E2 根因)。
  - **✅ MHC 已落地上库(loop R3-R13)**:Liger triton fwd+bwd 接入 `hyper_connection.py`(is_hip gate,NV 保 torch),端到端 4-layer train step 80s + 进第 2 rollout + 无 NaN,数值 fp32 1.67e-6 / bf16 P0/P1 无;commit 5819771(+Dockerfile `liger-kernel==0.8.0`)。
- [x] **MoE SwiGLU 提速(backward scan rank1,loop R14-R18 ✅ 端到端绿)**:DSv4 routed/shared expert 的 clamped-glu(silu·gate,clamp=10,·probs)backward 原走 torch autograd(`--no-bias-swiglu-fusion`+clamp+`use_te_activation_func=False` → experts.py 内联 glu,actor 最热 256experts×topk6)。**方案(不引外部依赖)**:给 Megatron `fused_bias_swiglu.py` silu 路径加 clamp+offset(逐字镜像同文件 quick_geglu),靠 `@jit_fuser`(torch 2.9=torch.compile→ROCm inductor)自动 lower 成 triton fwd+bwd。改 3 Megatron 文件(fused_bias_swiglu.py 加 clamp+offset、experts.py/mlp.py silu 分支补 2 参)+ miles `deepseekv4.py` gate(`bias_activation_fusion = torch.version.hip is not None`,ROCm fused / NV inline byte-for-byte)。Megatron 改动已上 fork:**`XinyuJiangCMU/Megatron-LM@amd/dsv4-swiglu-fused-clamp-20260630`**(commit `262b82419`,base radixark/Megatron-LM miles-main @`79fc0894d`;diff 备份 `docs/megatron-swiglu-fused-clamp-offset.patch`)。**Dockerfile pin 已改**(`docker/Dockerfile.rocm`:24-27 `MEGATRON_REPO=XinyuJiangCMU/Megatron-LM`、`MEGATRON_BRANCH=amd/dsv4-swiglu-fused-clamp-20260630`,base 79fc0894d;下次 build 生效,现在不用重 build)。**验证**:单元 fp32 数学精确(fwd 7e-7/dx 5e-7/dprobs_rel 1.2e-7);端到端 4-layer actor_train 76.2s、step0 全 finite 无 NaN、fused 真走(inductor compile×6)、logprob_abs_diff 3.65≈inline baseline 3.59、进第 2 rollout。⚠️ `true_on_policy/config.py:98` 无条件加 `--no-bias-swiglu-fusion`(bitwise 模式单独决策)。
- **backward 提速路线图(loop R14 workflow scan,按 慢torch-autograd × actor热 × 有开源triton/tilelang kernel 排序)**:rank0 MHC ✅ / rank1 MoE SwiGLU ✅ / rank2 shared SwiGLU(rank1 mlp.py 已覆盖)/ **rank3 token permute ✅**(开 `--moe-permute-fusion`,permute/unpermute 走 TE `fused_permute` triton 自带 bwd 替 torch autograd;纯 flag、cross-platform,其他模型已用;gfx950 端到端绿 miles `6fc0600`,2 train step 84.8s/68.8s + 2 rollout,no NaN,logprob_abs_diff 3.71)/ rank4 RoPE(❌ 跳过:backward~1-2% < MHC,唯一要从头写 triton kernel,ROI 最差)/ **rank5 RMSNorm ✅**(inline q-rmsnorm 走 Liger triton,bf16 bitwise 一致 0% mismatch,端到端 2 train step+2 rollout,miles `f13349f`;compressor.norm 留 torch parity-pinned 不换)/ rank6-8 router/probs(LOW)/ **rank9 true_on_policy CE ✅接入(储备)**(true_on_policy log_prob 走 Liger `liger_cross_entropy`,is_hip gate,not with_entropy/entropy 保 torch;训练侧隔离 diff 数值完全对 fp32 log_prob 1.91e-6/bf16 bitwise,miles `688f2dd`)——**储备**:默认 recipe `--true-on-policy-mode` off 不触发;E2E true_on_policy 撞 sglang deterministic inference 拒 `dsv4` attention backend(sglang 侧 blocker,非 CE kernel,待 sglang 支持)。纠偏:per_token quant backward 常数级(无梯度流经),不值得换。换算子高 payoff 矿脉已采完(剩 <2% LOW),大杠杆=rollout decode ROCm cuda-graph 硬墙(非算子)。
- **⭐ fp8 边界 ground truth(NV 对齐确认,loop R39;四方交叉验证 TE 源码 + Megatron + TileKernels CUDA reference + miles config)**:DSv4 fp8 training 的 **fp8 只活在 GEMM/linear 内部**(qkv/o-proj、fc1/fc2、MoE grouped-GEMM:GEMM 入口 cast fp8、输出回 bf16);**夹在 GEMM 之间的 activation/norm/CE/permute/MHC/lm_head/embedding 全部 bf16/fp32**(标准 TE blockwise `Float8BlockScaling` 惯例)。所以本趟换的 5 个 kernel(MHC/SwiGLU/q-RMSNorm/CE/permute)本就**不在 fp8 边界内**,Liger/TE triton 用 bf16 跟 NV ground truth **一致、零返工** —— MHC=TileKernels CUDA reference bf16+fp32 整模块无 fp8(qat.py fp8 quant 是独立 KV 压缩模块、不碰 MHC);CE=bf16 logits+fp32 reduce(bf16 正是 bitwise 对齐 SGLang rollout 打分契约需要的);SwiGLU 的 `activation_func_fp8_input_store`=输入存 fp8 省显存但 forward/backward 都 bf16 算。**结论:fp8 训练里要换的就是这些 bf16 activation/norm kernel,fp8 本身在没动的 TE GEMM 里。**
- [ ] **indexer 提速**(见待办 ③:tilelang stopgap 仅 gated 实验 / aiter non-gluon 才是 blessed 真修)
- [ ] 任何提速接入后,**重跑 Stage 2 全部数值 gate 仍须通过**,否则回退
- 数学参考:mHC = Manifold-Constrained Hyper-Connections(Sinkhorn-Knopp 投影到 Birkhoff polytope,arXiv:2512.24880)。
- ⚠️ **校正之前"MHC backward 全世界只有 CUDA 版、ROCm 必须自己写"的说法**:(1) NV `deepseek-ai/TileKernels` 的 `mhc/`(expand/head_compute_mix/post/pre_*)其实**全是 forward** kernel(`@compile_ops`,no autograd backward),整个 repo 没有 mhc backward 文件;NV 训练的 mhc backward 也靠 **autograd** —— **这条 AMD 跟 NV 持平,不落后**。(2) **Liger-Kernel 有 mHC fused fwd+bwd Triton kernel(PR #1065,merged 2026-03-05)**,triton cross-platform、gfx950 大概率可用,是现成 backward kernel 来源(待测,见 Stage 3 mhc 提速 (b))。所以 **backward 不必自己写**;我们当前 torch forward + autograd 是与 NV 同模式的正确兜底,提速换 tilelang/triton forward(或 Liger 整套 fwd+bwd)即可。

### R3 严格 on-policy(见待办 ②)✅ 已修通(loop 2026-06-30)
- [x] **修通(loop R42–R47)**:train44(2026-06-28)强开崩 `rollout_routed_experts is required`,旧归因"aiter topk producer 没接线 / deep sglang/aiter / 超本阶段"是**误诊**。双探针(GEN_DEBUG/RE_DEBUG)实测真根因 = Rust `sglang_router` v0.3.2(pin openai-protocol 1.0.0、无 `#[serde(flatten)]` catch-all)转发 /generate 时丢掉 `return_routed_experts` 字段(rollout 设了 True、http_server 收到 False)。修法:ROCm R3 走 miles python router(raw-bytes 透传,`--use-miles-router` is_hip gate)+ 删 disable guard,1 flag 零重 build,3 rollout+3 train step+0 崩(miles `b9047e3`)。详见 E24 + 上「性能优化实测结论」R3 段。

### 显式不做(防 24h 空转)
- 不追 4-layer 收敛 / loss 下降 / eval 准确率(模型未训练,eval 无意义)
- 不拿 DeepSeek-V3 的 <0.25% 当 4-layer gate(那是整轮收敛指标,几步 sanity 测不出)
- ue8m0 vs fp32 scale 的 recipe 差异先归因 recipe 再归因 kernel,别死磕 0.04 修不存在的 bug
- rollout generate 慢 → on-policy 趋势用 `debug_minimal`(短响应、关 eval)保证步数够,与输出质量无关

## 产物(host mount,容器重启不丢)

- checkpoint `models/DeepSeek-V4-Flash-FP8-4layer`(27G)、bf16 `...-bf16`(52G)、torch_dist `..._torch_dist`(52G)。
- 运行日志 `train{N}.log`(最新 train38 = 端到端验收通过那次;中间过程 log 已清)。
