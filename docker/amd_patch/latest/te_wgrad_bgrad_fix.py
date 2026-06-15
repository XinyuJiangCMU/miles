"""Make the ROCm wgrad GEMM import-safe against the missing hipBLASLt BGRADB algorithm.

On AMD (gfx950), hipBLASLt has no algorithm for a bf16 -> fp32-accumulate weight-grad
GEMM that also fuses the bias-gradient (BGRADB) epilogue, so the heuristic returns no
algorithms and the GEMM raises "Unable to find any suitable algorithms". This hits any
LayerNormLinear/Linear/LayerNormMLP with bias trained with fp32 grad accumulation
(e.g. Qwen2.5 QKV with --add-qkv-bias + --accumulate-allreduce-grads-in-fp32).

Fix: in general_gemm (the single chokepoint every wgrad path funnels through), run the
GEMM with the default epilogue (no fused bias grad) and reduce the bias gradient
separately afterwards. Gated on ROCm + bias-grad (grad + bias) + fp32 main_grad
accumulate + non-quantized, so CUDA, the forward bias-add path (grad=False) and fp8/fp4
are untouched.

Applied at build time against the installed transformer_engine (v2.8_rocm). Upstreamed in
the ROCm/TransformerEngine fork PR; remove this patch once it lands in the pinned TE
branch. The asserts make a silent miss fail the build loudly.
"""

import transformer_engine.pytorch.cpp_extensions.gemm as _gemm_mod

F = _gemm_mod.__file__
src = open(F).read()

# --- edit 1: insert the gate + gemm_bias before the args tuple, swap bias -> gemm_bias ---
old_args = '''    args = (
        A,
        transa,  # transa
        B,
        transb,  # transb
        out,
        quantization_params,
        TE_DType[out_dtype] if out_dtype is not None else None,
        bias,
        bias_dtype,'''
new_args = '''    # ROCm: hipBLASLt has no algorithm for a bf16 -> fp32-accumulate wgrad GEMM
    # that also fuses the bias-gradient (BGRADB) epilogue; the heuristic returns
    # no algorithms and the GEMM raises "Unable to find any suitable algorithms".
    # Run the GEMM with the default epilogue (no fused bias grad) and reduce the
    # bias gradient separately afterwards. Every wgrad path (Linear,
    # LayerNormLinear, LayerNormMLP, and the delayed-wgrad store) reads the bias
    # gradient from this function's return value, so handling it here covers them
    # all. Gated on ROCm + bias-grad (grad + bias) + fp32 main_grad accumulate +
    # non-quantized, so CUDA, the forward bias-add path (grad=False) and fp8/fp4
    # are untouched.
    rocm_split_dbias = (
        torch.version.hip is not None
        and grad
        and bias is not None
        and accumulate
        and out is not None
        and out.dtype == torch.float32
        and quantization_params is None
    )
    gemm_bias = None if rocm_split_dbias else bias

    args = (
        A,
        transa,  # transa
        B,
        transb,  # transb
        out,
        quantization_params,
        TE_DType[out_dtype] if out_dtype is not None else None,
        gemm_bias,
        bias_dtype,'''

n = src.count(old_args)
assert n == 1, f"te_wgrad_bgrad_fix: args block expected 1 match, found {n}"
src = src.replace(old_args, new_args)

# --- edit 2: recompute bias_grad after the generic_gemm call ---
old_call = '''    out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)

    if debug_quantizer is not None:'''
new_call = '''    out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)

    if rocm_split_dbias:
        # dbias = column-sum of grad_output (operand B) over tokens, accumulated
        # in fp32 and cast to the bias dtype to match the fused BGRADB epilogue
        # this replaces.
        bias_grad = B.reshape(-1, B.shape[-1]).sum(dim=0, dtype=torch.float32).to(bias.dtype)

    if debug_quantizer is not None:'''

n = src.count(old_call)
assert n == 1, f"te_wgrad_bgrad_fix: generic_gemm call block expected 1 match, found {n}"
src = src.replace(old_call, new_call)

open(F, "w").write(src)
print(f"te_wgrad_bgrad_fix applied: {F}")
