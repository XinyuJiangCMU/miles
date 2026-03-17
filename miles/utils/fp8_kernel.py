import torch
import triton
import triton.language as tl


def _get_fp8_dtype():
    """Get the appropriate FP8 dtype for the current platform.

    MI300X (gfx942) natively supports e4m3fnuz for scaled_mm,
    while NVIDIA GPUs use e4m3fn.
    """
    if torch.version.hip is not None:
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


fp8_dtype = _get_fp8_dtype()
fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
            x: the dividend.
            y: the divisor.

    Returns:
            The result of the ceiling division.
    """
    return (x + y - 1) // y


@triton.jit
def _blockwise_cast_to_fp8_triton(
    X,
    Y,
    S,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    M,
    N,
    eps,
    fp8_min,
    fp8_max,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 128,
):
    pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=1), tl.int64)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = off_m < M
    mask_n = off_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(x)), eps)
    x_s = _absmax / fp8_max
    s_inv = 1.0 / x_s
    y_q = tl.clamp(x * s_inv, fp8_min, fp8_max).to(Y.dtype.element_ty)

    tl.store(Y + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, y_q, mask=mask)
    tl.store(S + pid_m * stride_sm + pid_n * stride_sn, x_s)


def normalize_e4m3fn_to_e4m3fnuz(tensor: torch.Tensor, scale: torch.Tensor | None = None):
    """Convert e4m3fn tensors to e4m3fnuz for AMD MI300X.

    On AMD MI300X (gfx942), scaled_mm requires e4m3fnuz format.
    This handles the bit-level differences between the two FP8 formats:
    - e4m3fn: max=448, NaN=0x7F
    - e4m3fnuz: max=240, NaN=0x80, zero=0x00

    The scale factor is adjusted by ratio of max values (448/240).
    """
    if tensor.dtype != torch.float8_e4m3fn:
        return tensor, scale

    # View as uint8 to manipulate bits
    as_uint8 = tensor.view(torch.uint8)
    # In e4m3fn, 0xFF and 0x7F are NaN; in fnuz, 0x80 is NaN
    # Map e4m3fn NaN bits to zero in fnuz (safe default)
    as_uint8 = torch.where(
        (as_uint8 == 0x7F) | (as_uint8 == 0xFF),
        torch.zeros_like(as_uint8),
        as_uint8,
    )
    result = as_uint8.view(torch.float8_e4m3fnuz)

    # Adjust scale: e4m3fn max (448) vs e4m3fnuz max (240)
    if scale is not None:
        scale = scale * (torch.finfo(torch.float8_e4m3fn).max / torch.finfo(torch.float8_e4m3fnuz).max)

    return result, scale


def blockwise_cast_to_fp8_triton(x: torch.Tensor, block_size=None) -> tuple[torch.Tensor, torch.Tensor]:
    BLOCK_M, BLOCK_N = 128, 128
    if block_size:
        BLOCK_M, BLOCK_N = block_size[0], block_size[1]
    M, N = x.shape
    y = torch.empty(M, N, device=x.device, dtype=fp8_dtype)
    s = torch.empty(ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N), dtype=torch.float32, device=x.device)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    if x.is_contiguous():
        kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 8, "num_stages": 2}
    else:
        kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 1, "num_stages": 4}
    _blockwise_cast_to_fp8_triton[grid](
        x, y, s, *x.stride(), *y.stride(), *s.stride(), M, N, 1e-10, fp8_min, fp8_max, **kwargs
    )
    return y, s
