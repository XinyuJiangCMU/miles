from __future__ import annotations

import torch
import triton.language as tl
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    invoke_fused_moe_kernel,
    moe_align_block_size,
    silu_and_mul,
)

# AMD MI300X benefits from larger block sizes due to wider memory bus and CDNA3 matrix cores
_IS_AMD = hasattr(torch.version, "hip") and torch.version.hip is not None
_DEFAULT_MOE_CONFIG = (
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}
    if _IS_AMD
    else {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}
)

def _get_moe_sum_reduce():
    """Get moe_sum_reduce with fallback for ROCm."""
    try:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import moe_sum_reduce

        return moe_sum_reduce
    except ImportError:
        pass

    # On AMD/ROCm, prefer AITER's moe_sum (54x faster than PyTorch fallback)
    if _IS_AMD:
        try:
            from aiter import moe_sum as _aiter_moe_sum

            def _moe_sum_reduce_aiter(input_tensor, output_tensor, scale=1.0):
                _aiter_moe_sum(input_tensor, output_tensor)
                if scale != 1.0:
                    output_tensor.mul_(scale)

            return _moe_sum_reduce_aiter
        except ImportError:
            pass

    try:
        from sgl_kernel import moe_sum_reduce as _moe_sum_reduce

        # Test if the kernel is actually available (may fail on ROCm)
        import torch

        _test_in = torch.zeros(1, 1, 1, device="cuda")
        _test_out = torch.zeros(1, 1, device="cuda")
        _moe_sum_reduce(_test_in, _test_out, 1.0)
        return _moe_sum_reduce
    except Exception:
        pass

    # Pure PyTorch fallback
    def _moe_sum_reduce_fallback(input_tensor, output_tensor, scale=1.0):
        output_tensor.copy_(input_tensor.sum(dim=1) * scale)

    return _moe_sum_reduce_fallback


moe_sum_reduce = _get_moe_sum_reduce()

from .fused_moe_triton_backward_kernels import invoke_fused_moe_backward_kernel


class GateUpProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        num_tokens, _ = hidden_states.shape
        E, N, _ = w1.shape
        # We execute the fused_moe kernel in chunks to circumvent this issue:
        # https://github.com/vllm-project/vllm/issues/5938
        CHUNK_SIZE = 64 * 1024

        config = _DEFAULT_MOE_CONFIG

        topk = topk_ids.shape[1]

        intermediate_cache1 = torch.empty(
            (num_tokens * topk, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )
            curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
            cur_intermediate_cache1 = intermediate_cache1[begin_chunk_idx * topk : end_chunk_idx * topk]

            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            invoke_fused_moe_kernel(
                curr_hidden_states,
                w1,
                None,
                cur_intermediate_cache1,
                None,
                None,
                None,
                curr_topk_weights,
                curr_topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                False,
                topk_ids.shape[1],
                config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                c_sorted=False,
                filter_expert=True,
            )

        ctx.save_for_backward(hidden_states, w1, topk_weights, topk_ids)
        ctx.config = config
        ctx.num_tokens = num_tokens
        ctx.topk = topk

        return intermediate_cache1

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for GateUpProjFunction using Triton kernels.

        Args:
            grad_output: shape (num_tokens * topk, N)

        Returns:
            (grad_hidden_states, grad_w1, grad_topk_weights, None)
        """

        hidden_states, w1, topk_weights, topk_ids = ctx.saved_tensors
        config = ctx.config
        num_tokens = ctx.num_tokens
        topk = ctx.topk

        E, N, D_in = w1.shape
        CHUNK_SIZE = 64 * 1024

        # Initialize gradient tensors
        grad_hidden_states = torch.zeros_like(hidden_states)
        grad_w1 = torch.zeros_like(w1)
        # GateUpProj stage doesn't need topk_weights gradient
        grad_topk_weights = torch.zeros_like(topk_weights)

        num_chunks = (num_tokens // CHUNK_SIZE) + 1

        # Process in chunks to match forward pass
        for chunk in range(num_chunks):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )

            curr_num_tokens = end_chunk_idx - begin_chunk_idx
            if curr_num_tokens == 0:
                continue

            curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
            curr_grad_output = grad_output[begin_chunk_idx * topk : end_chunk_idx * topk]

            # Get aligned metadata
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            # For single-chunk case, use grad_w1 directly (kernel uses atomic_add)
            curr_grad_w1 = grad_w1 if num_chunks == 1 else torch.zeros_like(w1)

            # Call Triton backward kernel with MUL_ROUTED_WEIGHT=False
            invoke_fused_moe_backward_kernel(
                grad_output=curr_grad_output,
                input=curr_hidden_states,
                weight=w1,
                grad_input=grad_hidden_states[begin_chunk_idx:end_chunk_idx],
                grad_weight=curr_grad_w1,
                grad_topk_weights=None,  # Not needed for GateUpProj
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False,
                top_k=topk,
                config=config,
                compute_type=tl.bfloat16,
            )

            # Accumulate weight gradients for multi-chunk case
            if num_chunks > 1:
                grad_w1 += curr_grad_w1

        return grad_hidden_states, grad_w1, grad_topk_weights, None


class SiluAndMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, intermediate_cache1: torch.Tensor):
        num_tokens, N = intermediate_cache1.shape
        intermediate_cache2 = torch.empty(
            (num_tokens, N // 2),
            device=intermediate_cache1.device,
            dtype=intermediate_cache1.dtype,
        )
        silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)

        ctx.save_for_backward(intermediate_cache1)
        return intermediate_cache2

    @staticmethod
    def backward(ctx, grad_output):
        (intermediate_cache1,) = ctx.saved_tensors
        N = intermediate_cache1.shape[-1]
        x1, x2 = intermediate_cache1.view(-1, N).chunk(2, dim=-1)
        silu_x1 = torch.nn.functional.silu(x1)

        sig = torch.sigmoid(x1)
        dsilu_dx1 = sig + x1 * sig * (1 - sig)
        grad_x1 = grad_output * x2 * dsilu_dx1
        grad_x2 = grad_output * silu_x1
        grad_input = torch.cat([grad_x1, grad_x2], dim=-1)

        return grad_input.view_as(intermediate_cache1)


class DownProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        intermediate_cache2: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        num_tokens, _ = intermediate_cache2.shape
        topk = topk_ids.shape[1]
        num_tokens //= topk
        E, _, _ = w2.shape
        # We execute the fused_moe kernel in chunks to circumvent this issue:
        # https://github.com/vllm-project/vllm/issues/5938
        CHUNK_SIZE = 64 * 1024

        config = _DEFAULT_MOE_CONFIG

        intermediate_cache3 = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=intermediate_cache2.device,
            dtype=intermediate_cache2.dtype,
        )

        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )
            cur_intermediate_cache2 = intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk]
            cur_intermediate_cache3 = intermediate_cache3[begin_chunk_idx:end_chunk_idx]

            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )
            invoke_fused_moe_kernel(
                cur_intermediate_cache2,
                w2,
                None,
                cur_intermediate_cache3,
                None,
                None,
                None,
                curr_topk_weights,
                curr_topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                True,
                1,
                config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                c_sorted=False,
                filter_expert=True,
            )

        ctx.save_for_backward(intermediate_cache2, w2, topk_weights, topk_ids)
        ctx.config = config
        ctx.num_tokens = num_tokens
        ctx.topk = topk

        return intermediate_cache3

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for DownProjFunction using Triton kernels.

        Args:
            grad_output: shape (num_tokens, topk, hidden_size)

        Returns:
            (grad_intermediate_cache2, grad_w2, grad_topk_weights, None)
        """
        intermediate_cache2, w2, topk_weights, topk_ids = ctx.saved_tensors
        config = ctx.config
        num_tokens = ctx.num_tokens
        topk = ctx.topk

        E, hidden_size, intermediate_size = w2.shape
        CHUNK_SIZE = 64 * 1024

        # Initialize gradient tensors
        grad_intermediate_cache2 = torch.zeros_like(intermediate_cache2)
        grad_w2 = torch.zeros_like(w2)
        grad_topk_weights = torch.zeros_like(topk_weights)

        num_chunks = (num_tokens // CHUNK_SIZE) + 1

        # Process in chunks to match forward pass
        for chunk in range(num_chunks):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )

            curr_num_tokens = end_chunk_idx - begin_chunk_idx
            if curr_num_tokens == 0:
                continue

            curr_intermediate_cache2 = intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
            curr_grad_output = grad_output[begin_chunk_idx:end_chunk_idx]

            # Get aligned metadata
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            # For single-chunk case, use output buffers directly (kernel uses atomic_add)
            if num_chunks == 1:
                curr_grad_w2 = grad_w2
                curr_grad_topk_weights = grad_topk_weights[begin_chunk_idx:end_chunk_idx]
            else:
                curr_grad_w2 = torch.zeros_like(w2)
                curr_grad_topk_weights = torch.zeros_like(curr_topk_weights)

            # Call Triton backward kernel with MUL_ROUTED_WEIGHT=True
            invoke_fused_moe_backward_kernel(
                grad_output=curr_grad_output,
                input=curr_intermediate_cache2,
                weight=w2,
                grad_input=grad_intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk],
                grad_weight=curr_grad_w2,
                grad_topk_weights=curr_grad_topk_weights,
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=True,
                top_k=1,
                config=config,
                compute_type=tl.bfloat16,
            )

            # Accumulate weight/topk gradients for multi-chunk case
            if num_chunks > 1:
                grad_w2 += curr_grad_w2
                grad_topk_weights[begin_chunk_idx:end_chunk_idx] = curr_grad_topk_weights

        return grad_intermediate_cache2, grad_w2, grad_topk_weights, None


class MoeSumReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        intermediate_cache3: torch.Tensor,
        hidden_states_shape,
    ):
        out_hidden_states = torch.empty(
            hidden_states_shape, device=intermediate_cache3.device, dtype=intermediate_cache3.dtype
        )
        moe_sum_reduce(
            intermediate_cache3,
            out_hidden_states,
            1.0,
        )
        ctx.save_for_backward(intermediate_cache3)
        return out_hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        (intermediate_cache3,) = ctx.saved_tensors
        return grad_output.unsqueeze(1).expand_as(intermediate_cache3), None
