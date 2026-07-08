"""ROCm / portable torch mHC ops — drop-in for ``tile_kernels.modeling.mhc.ops``.

The upstream DeepSeek-V4 plugin routes the hyper-connection ops through
``tile_kernels.modeling.mhc`` (CUDA/SM90-100 only TileLang kernels), which do not
build on ROCm (gfx950), blocking even the import of the training actor. These
reimplement the exact same math in plain PyTorch so autograd produces forward +
backward automatically (the legacy in-tree impl only had a no-grad forward path).
Semantics are reverse-engineered from the TileLang reference in sglang
(``srt/layers/mhc.py`` :: ``mhc_pre_big_fuse_tilelang`` / ``hc_split_sinkhorn_kernel``
/ ``mhc_post_tilelang``) and the aiter op set (``aiter/ops/mhc.py``, forward-only,
so unusable for the training path).

Names/signatures are kept identical to the tile_kernels ops so the hyper_connection.py
call sites are unchanged — on CUDA those sites import the same names from
tile_kernels instead (single platform fork in hyper_connection.py).

Shapes (B/S may be flattened token dims):
  residual / x_hc : (..., hc_mult, hidden)
  fn              : (hc_mult3, hc_mult * hidden)   fp32, hc_mult3 = hc*(2+hc)
  hc_scale        : (3,) fp32 for pre/post, (1,) fp32 for the head
  hc_base         : (hc_mult3,) fp32 for pre/post, (hc_mult,) fp32 for head
"""

import torch
from torch import Tensor


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
