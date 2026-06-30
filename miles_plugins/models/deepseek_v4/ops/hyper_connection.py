"""DeepSeek V4 Hyper-Connection utility.

Public API (`HCHeadParams`, `DeepSeekV4HyperConnectionUtil`) preserved so that
the Megatron-LM patch (radixark/Megatron-LM PR #28) call sites in
``transformer_layer.py`` and ``transformer_block.py`` keep working.

The mHC ops (``mhc_pre*``/``mhc_post``/``sinkhorn_normalize``/``mhc_head*``)
were originally imported from ``tile_kernels.modeling.mhc`` (CUDA/SM90-100-only
TileLang kernels). Those do not build on ROCm (gfx950), which blocked the
Megatron *training* actor from even importing this module. They are now
reimplemented below in plain PyTorch so autograd provides both forward and
backward (the legacy in-tree implementation only had a no-grad forward path).
"""

import einops
import torch
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

# DeepSeek V4 originally used post = 2 * sigmoid(...) for the post-layer mix
# (see the legacy ``hc_split_sinkhorn`` kernel). We pass the same factor through
# ``post_mult_value``.
_HC_POST_MULT_VALUE = 2.0

# ROCm: route MHC through Liger's triton fused fwd+bwd kernels. gfx950 has no
# tile_kernels (CUDA-only) and sglang's tilelang mhc is forward-only + its fused
# kernel hits a missing HIP op; Liger has both forward and backward in triton.
# NV (torch.version.hip is None) keeps the existing torch path byte-for-byte.
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


# ---------------------------------------------------------------------------
# ROCm / portable mHC implementation.
#
# The upstream DeepSeek-V4 plugin routes the hyper-connection ops through
# ``tile_kernels.modeling.mhc`` (CUDA/SM90-100 only TileLang kernels). Those
# kernels do not build on ROCm (gfx950), so the Megatron *training* actor could
# not even import this module.
#
# The functions below reimplement the exact same math in plain PyTorch ops so
# autograd produces forward + backward automatically (the actor trains, unlike
# the rollout path which is forward-only). The semantics are reverse-engineered
# from the TileLang reference kept in sglang
# (``sglang/srt/layers/mhc.py`` :: ``mhc_pre_big_fuse_tilelang`` /
# ``hc_split_sinkhorn_kernel`` / ``mhc_post_tilelang``) and the aiter op set
# (``aiter/ops/mhc.py``). aiter exposes only fused, forward-only compiled
# kernels (``@compile_ops`` with no autograd backward) at a granularity that
# does not match the per-step functions the call sites need, so it cannot be
# used for the training path; torch is the correct fallback here.
#
# Naming/signatures are kept identical to the previous ``tile_kernels`` imports
# so the call sites below are unchanged.
#
# Shapes (B/S may be flattened token dims):
#   residual / x_hc : (..., hc_mult, hidden)
#   fn              : (hc_mult3, hc_mult * hidden)   fp32, hc_mult3 = hc*(2+hc)
#   hc_scale        : (3,) fp32 for pre/post, (1,) fp32 for the head
#   hc_base         : (hc_mult3,) fp32 for pre/post, (hc_mult,) fp32 for head
# ---------------------------------------------------------------------------


def mhc_pre_norm_fn(
    x: Tensor,
    fn: Tensor,
    norm_weight: Tensor | None,
    rms_eps: float,
    fuse_grad_acc: bool = True,
) -> Tensor:
    """RMSNorm (over the flattened ``hc_mult*hidden`` axis) then GEMM with ``fn``.

    Returns the un-split ``mixes`` of shape ``(..., fn.shape[0])``. Equivalent to
    the TileLang ``mhc_pre_gemm_sqrsum`` + ``rms = rsqrt(sumsq/(hc*h)+eps)`` then
    ``mixes *= rms`` (a per-token scalar, so applying it before or after the GEMM
    is identical). ``fuse_grad_acc`` is a no-op here (it only controlled the
    TileKernels backward storage trick); autograd handles the gradient.
    """
    *outer, hc_mult, hidden = x.shape
    x_flat = x.reshape(*outer, hc_mult * hidden).to(torch.float32)
    rms = torch.rsqrt(x_flat.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
    x_norm = x_flat * rms
    if norm_weight is not None:
        x_norm = x_norm * norm_weight.to(torch.float32)
    mixes = torch.matmul(x_norm, fn.to(torch.float32).t())
    return mixes


def mhc_pre_split_mixes(
    mixes: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    post_mult_value: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Split ``mixes`` into (pre, post, comb-logits).

    pre[j]    = sigmoid(mixes[j]            * scale[0] + base[j])          + eps
    post[j]   = sigmoid(mixes[j+hc]         * scale[1] + base[j+hc]) * post_mult_value
    comb[j,k] =          mixes[2hc+j*hc+k]  * scale[2] + base[2hc+j*hc+k]
    (comb is returned as raw logits; sinkhorn is applied separately.)
    """
    scale = hc_scale.to(torch.float32)
    base = hc_base.to(torch.float32)
    mixes = mixes.to(torch.float32)

    pre_logits = mixes[..., :hc_mult]
    post_logits = mixes[..., hc_mult : 2 * hc_mult]
    comb_logits = mixes[..., 2 * hc_mult :].reshape(*mixes.shape[:-1], hc_mult, hc_mult)

    base_pre = base[:hc_mult]
    base_post = base[hc_mult : 2 * hc_mult]
    base_comb = base[2 * hc_mult :].reshape(hc_mult, hc_mult)

    pre = torch.sigmoid(pre_logits * scale[0] + base_pre) + eps
    post = torch.sigmoid(post_logits * scale[1] + base_post) * post_mult_value
    comb = comb_logits * scale[2] + base_comb
    return pre.unsqueeze(-1), post.unsqueeze(-1), comb


def sinkhorn_normalize(comb: Tensor, repeat: int, eps: float) -> Tensor:
    """Sinkhorn-normalize the ``(..., hc, hc)`` comb logits.

    Mirrors ``hc_split_sinkhorn_kernel`` / the big_fuse path: a softmax over the
    last axis (with +eps), a column normalization, then ``repeat-1`` extra
    row/column normalization iterations. ``[..., j, k]`` indexes
    ``[old_route, new_route]``.
    """
    comb = comb.to(torch.float32)
    comb = torch.exp(comb - comb.amax(dim=-1, keepdim=True))
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


def mhc_pre_apply_mix(x: Tensor, mix: Tensor) -> Tensor:
    """Collapse the ``hc_mult`` heads of ``x`` with per-head weights ``mix``.

    ``x``: (..., hc_mult, hidden); ``mix``: (..., hc_mult, 1) or (..., hc_mult).
    Returns ``(..., hidden)`` = ``sum_j mix[j] * x[j]``.
    """
    if mix.dim() == x.dim() - 1:
        mix = mix.unsqueeze(-1)
    out = (mix.to(torch.float32) * x.to(torch.float32)).sum(dim=-2)
    return out


def mhc_head_compute_mix(
    mix_in: Tensor, scale: Tensor, hc_base: Tensor, eps: float
) -> Tensor:
    """Head mixing weights: ``sigmoid(mix_in * scale[0] + base) + eps``.

    Same activation as the ``pre`` slot; ``scale`` is (1,), ``base`` is (hc_mult,).
    """
    scale = scale.to(torch.float32)
    base = hc_base.to(torch.float32)
    return torch.sigmoid(mix_in.to(torch.float32) * scale[0] + base) + eps


def mhc_post(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    """Hyper-connection post step.

    out[j, h] = post[j] * x[h] + sum_k comb[k, j] * residual[k, h]

    ``x``: (..., hidden); ``residual``: (..., hc_mult, hidden);
    ``post``: (..., hc_mult, 1) or (..., hc_mult); ``comb``: (..., hc_mult, hc_mult).
    Returns ``(..., hc_mult, hidden)``.
    """
    post_v = post
    if post_v.dim() == residual.dim() and post_v.shape[-1] == 1:
        post_v = post_v.squeeze(-1)
    post_v = post_v.to(torch.float32)
    x_f = x.to(torch.float32)
    res_f = residual.to(torch.float32)
    comb_f = comb.to(torch.float32)

    term1 = post_v.unsqueeze(-1) * x_f.unsqueeze(-2)  # (..., hc_mult, hidden)
    term2 = torch.einsum("...kj,...kh->...jh", comb_f, res_f)  # sum over old route k
    return term1 + term2


def mhc_pre_big_fuse(
    residual: Tensor,
    fn: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    n_splits: int = 16,
) -> tuple[Tensor, Tensor, Tensor]:
    """Full mHC pre step (norm + GEMM + split + sinkhorn + apply).

    Returns ``(post, comb, layer_input)`` matching the TileKernels call order.
    ``n_splits`` is ignored (it only tuned the CUDA split-k kernel).
    """
    hc_mult = residual.shape[-2]
    mixes = mhc_pre_norm_fn(residual, fn, None, rms_eps)
    pre, post, comb = mhc_pre_split_mixes(
        mixes, hc_scale, hc_base, hc_mult, mhc_post_mult_value, mhc_pre_eps
    )
    comb = sinkhorn_normalize(comb, repeat=sinkhorn_repeat, eps=mhc_sinkhorn_eps)
    layer_input = mhc_pre_apply_mix(residual, pre)
    return post, comb, layer_input


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
    """Hyper-Connection helper backed by the portable torch mHC ops above."""

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
