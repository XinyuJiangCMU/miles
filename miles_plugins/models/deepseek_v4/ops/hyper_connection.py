"""DeepSeek V4 Hyper-Connection utility.

Public API (`HCHeadParams`, `DeepSeekV4HyperConnectionUtil`) preserved so that
the Megatron-LM patch (radixark/Megatron-LM PR #28) call sites in
``transformer_layer.py`` and ``transformer_block.py`` keep working.

The mHC ops (``mhc_pre*`` / ``mhc_post`` / ``sinkhorn_normalize`` / ``mhc_head*``)
are imported from ``tile_kernels.modeling.mhc`` on CUDA; on ROCm (gfx950, which has
no tile_kernels) they are dispatched to the in-tree torch reimplementation in
``miles_plugins/amd/models/deepseek_v4/mhc`` (drop-in, same names/signatures). On ROCm the
hot fwd+bwd path is further accelerated through Liger's triton fused kernels.
"""

import einops
import torch
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

# mHC ops: CUDA -> tile_kernels; ROCm -> in-tree torch reimplementation (amd/models/deepseek_v4/mhc),
# a drop-in with identical names/signatures so the call sites below are platform-agnostic.
if torch.version.hip is not None:
    from miles_plugins.amd.models.deepseek_v4.mhc import (
        mhc_head_compute_mix,
        mhc_post,
        mhc_pre_apply_mix,
        mhc_pre_big_fuse,
        mhc_pre_norm_fn,
        mhc_pre_split_mixes,
        sinkhorn_normalize,
    )
else:
    from tile_kernels.modeling.mhc.ops import (
        mhc_head_compute_mix,
        mhc_post,
        mhc_pre_apply_mix,
        mhc_pre_big_fuse,
        mhc_pre_norm_fn,
        mhc_pre_split_mixes,
        sinkhorn_normalize,
    )

# DeepSeek V4 originally used post = 2 * sigmoid(...) for the post-layer mix
# (see the legacy ``hc_split_sinkhorn`` kernel), passed through ``post_mult_value``.
_HC_POST_MULT_VALUE = 2.0

# ROCm: further accelerate the mHC fwd+bwd through Liger's triton fused kernels
# (tile_kernels' tilelang mhc is forward-only + hits a missing HIP op). NV keeps
# the torch/tile_kernels path byte-for-byte.
_USE_LIGER_MHC = torch.version.hip is not None
if _USE_LIGER_MHC:
    try:
        from liger_kernel.transformers.functional import (
            liger_mhc_coeffs,
            liger_mhc_pre,
            liger_mhc_post_res,
        )
    except Exception:
        _USE_LIGER_MHC = False


class HCHeadParams(MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        hc_mult = config.dsv4_hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = torch.nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = torch.nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = torch.nn.Parameter(torch.empty(1, dtype=torch.float32))

        for p in [self.hc_head_fn, self.hc_head_base, self.hc_head_scale]:
            p._keep_fp32 = True

    def forward(self):
        raise NotImplementedError


class DeepSeekV4HyperConnectionUtil:
    """Hyper-Connection helper backed by the dispatched mHC ops above."""

    def __init__(self, config: TransformerConfig):
        self.norm_eps = config.layernorm_epsilon
        self.hc_mult = config.dsv4_hc_mult
        self.hc_sinkhorn_iters = config.dsv4_hc_sinkhorn_iters
        self.hc_eps = config.dsv4_hc_eps

    def hc_pre_raw(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """``x`` is ``(B, S, hc_mult, hidden)``. Returns layer input + post/comb mixes.

        ``x`` is cast to bf16 (matching the original DeepSeek-V4 residual layout)
        and ``fn`` stays fp32; the math runs in fp32 internally.
        """
        dtype = x.dtype
        x_bf16 = (x if x.dtype == torch.bfloat16 else x.bfloat16()).contiguous()

        if _USE_LIGER_MHC:
            # Liger triton fused coeffs (sinkhorn + pre/post/comb) + pre-apply.
            # phi is the transpose of DSv4 fn ([M,K] -> [K,M]); eps defaults in
            # Liger differ from DSv4 so pre_eps/sinkhorn_eps/rms_eps are forced.
            phi = hc_fn.t().contiguous()
            h_pre, post, comb = liger_mhc_coeffs(
                x_bf16,
                phi,
                hc_base,
                hc_scale[0],
                hc_scale[1],
                hc_scale[2],
                allow_fp32=False,
                tmax=self.hc_sinkhorn_iters,
                rms_eps=self.norm_eps,
                pre_eps=self.hc_eps,
                sinkhorn_eps=self.hc_eps,
                post_mult=_HC_POST_MULT_VALUE,
            )
            layer_input = liger_mhc_pre(x_bf16, h_pre)
            # Keep post as Liger's native (..., hc); do NOT unsqueeze to (..., hc, 1).
            # LigerMHCPostResFunction.backward returns the post gradient as (..., hc),
            # so unsqueezing makes autograd reject it at the train step (invalid
            # gradient [., hc] vs expected [., hc, 1]). hc_post_raw feeds post
            # straight into liger_mhc_post_res, which wants (..., hc) anyway.
        elif not torch.is_grad_enabled():
            post, comb, layer_input = mhc_pre_big_fuse(
                x_bf16,
                hc_fn,
                hc_scale,
                hc_base,
                rms_eps=self.norm_eps,
                mhc_pre_eps=self.hc_eps,
                mhc_sinkhorn_eps=self.hc_eps,
                mhc_post_mult_value=_HC_POST_MULT_VALUE,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
                n_splits=16,
            )
        else:
            mixes = mhc_pre_norm_fn(
                x_bf16,
                hc_fn,
                None,
                self.norm_eps,
                fuse_grad_acc=False,
            )
            pre_mix, post, comb = mhc_pre_split_mixes(
                mixes,
                hc_scale,
                hc_base,
                self.hc_mult,
                _HC_POST_MULT_VALUE,
                self.hc_eps,
            )
            comb = sinkhorn_normalize(comb, repeat=self.hc_sinkhorn_iters, eps=self.hc_eps)
            layer_input = mhc_pre_apply_mix(x_bf16, pre_mix)
        return layer_input.to(dtype), post, comb

    def hc_post_raw(
        self,
        x: Tensor,
        residual: Tensor,
        post: Tensor,
        comb: Tensor,
    ) -> Tensor:
        """``x``: ``(B, S, hidden)``; ``residual``: ``(B, S, hc_mult, hidden)``."""
        dtype = x.dtype
        x_bf16 = (x if x.dtype == torch.bfloat16 else x.bfloat16()).contiguous()
        res_bf16 = (residual if residual.dtype == torch.bfloat16 else residual.bfloat16()).contiguous()
        if _USE_LIGER_MHC:
            # comb must be transposed: DSv4 sums comb over dim-2, Liger over dim-1
            # (h_res == comb elementwise but opposite reduction axis). Skipping the
            # transpose silently corrupts the output (~28% end-to-end error).
            out = liger_mhc_post_res(
                res_bf16, x_bf16, post, comb.transpose(-1, -2).contiguous()
            )
        else:
            out = mhc_post(x_bf16, res_bf16, post, comb)
        return out.to(dtype)

    def hc_head_raw(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> Tensor:
        """``x``: ``(B, S, hc_mult, hidden)``. Returns ``(B, S, hidden)``."""
        assert hc_fn.dtype == torch.float32
        assert hc_scale.dtype == torch.float32
        assert hc_base.dtype == torch.float32

        dtype = x.dtype
        x_bf16 = (x if x.dtype == torch.bfloat16 else x.bfloat16()).contiguous()

        # The head only consumes the first ``hc_mult`` mix outputs (the "pre"
        # slot). ``mhc_pre_norm_fn`` expects an ``fn`` with ``hc_mult3`` rows, so
        # pad the head ``fn`` and slice back the first ``hc_mult`` columns.
        mhc_mult = self.hc_mult
        mhc_mult3 = mhc_mult * (2 + mhc_mult)
        fn_padded = hc_fn
        if fn_padded.shape[0] < mhc_mult3:
            fn_padded = F.pad(fn_padded, (0, 0, 0, mhc_mult3 - fn_padded.shape[0]))

        mixes = mhc_pre_norm_fn(
            x_bf16,
            fn_padded,
            None,
            self.norm_eps,
            fuse_grad_acc=False,
        )
        mix_in = mixes[..., :mhc_mult].contiguous()
        scale = hc_scale.reshape(1) if hc_scale.numel() == 1 else hc_scale
        out_mix = mhc_head_compute_mix(mix_in, scale, hc_base, self.hc_eps)
        layer_input = mhc_pre_apply_mix(x_bf16, out_mix.unsqueeze(-1))
        return layer_input.to(dtype)

    def layer_pre(
        self,
        hidden_states: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert hc_fn.dtype == torch.float32
        assert hc_scale.dtype == torch.float32
        assert hc_base.dtype == torch.float32

        x = einops.rearrange(hidden_states, "s b hc d -> b s hc d")
        x, post, comb = self.hc_pre_raw(x=x, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base)
        hidden_states = einops.rearrange(x, "b s d -> s b d")
        return hidden_states, post, comb

    def layer_post(
        self,
        output_with_bias: Tensor | tuple[Tensor, Tensor | None],
        residual: Tensor,
        post: Tensor,
        comb: Tensor,
    ) -> Tensor:
        if isinstance(output_with_bias, tuple):
            out, bias = output_with_bias
            assert bias is None
        else:
            out = output_with_bias
        assert isinstance(out, torch.Tensor)

        out = einops.rearrange(out, "s b d -> b s d")
        residual_bshd = einops.rearrange(residual, "s b hc d -> b s hc d")
        hidden_states = self.hc_post_raw(x=out, residual=residual_bshd, post=post, comb=comb)
        return einops.rearrange(hidden_states, "b s hc d -> s b hc d")

    def block_expand(self, hidden_states: Tensor) -> Tensor:
        return einops.repeat(hidden_states, "s b d -> s b hc d", hc=self.hc_mult)

    def block_head(
        self,
        hidden_states: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> Tensor:
        x = einops.rearrange(hidden_states, "s b hc d -> b s hc d")
        x = self.hc_head_raw(x=x, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base)
        return einops.rearrange(x, "b s d -> s b d")
