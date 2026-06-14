"""Make sglang's HIP capability probes import-safe when no GPU is visible.

Several HIP arch/capability probes in sglang read GPU 0 at import time on any
ROCm build. In a process with no visible HIP device (e.g. a Ray job driver
scheduled with 0 GPUs, HIP_VISIBLE_DEVICES=""), those reads crash and take down
`import sglang`. This guards them the same way sglang's own
qr_rocm_arch_available() already does.

Applied at build time against the sglang-miles source. Upstreamed in
https://github.com/XinyuJiangCMU/sglang/pull/22 (target sgl-project/sglang);
remove this patch once it lands in the pinned sglang branch.

Each replacement asserts exactly one match so a silent miss fails the build
loudly instead of producing an unguarded image.
"""

SGL = "/sgl-workspace/sglang/python/sglang"

patches = [
    # 1. common.py is_gfx95_supported  (get_device_properties -> try/except)
    (f"{SGL}/srt/utils/common.py",
'''def is_gfx95_supported():
    """
    Returns whether the current platform supports MX types.
    """
    if torch.version.hip:
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ["gfx95"])
    else:
        return False''',
'''def is_gfx95_supported():
    """
    Returns whether the current platform supports MX types.
    """
    if not torch.version.hip:
        return False
    try:
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        return any(gfx in gcn_arch for gfx in ["gfx95"])
    except Exception as e:
        logger.warning("Failed to determine gfx95 support: %s", e)
        return False'''),

    # 2. common.py mxfp_supported  (twin of is_gfx95_supported)
    (f"{SGL}/srt/utils/common.py",
'''def mxfp_supported():
    """
    Returns whether the current platform supports MX types.
    """
    if torch.version.hip:
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ["gfx95"])
    else:
        return False''',
'''def mxfp_supported():
    """
    Returns whether the current platform supports MX types.
    """
    if not torch.version.hip:
        return False
    try:
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        return any(gfx in gcn_arch for gfx in ["gfx95"])
    except Exception as e:
        logger.warning("Failed to determine mxfp support: %s", e)
        return False'''),

    # 3. fp8_kernel.py is_fp8_fnuz  (get_device_properties -> try/except)
    (f"{SGL}/srt/layers/quantization/fp8_kernel.py",
'''def is_fp8_fnuz() -> bool:
    if _is_hip:
        # only device 0 is checked, this assumes MI300 platforms are homogeneous
        return "gfx94" in torch.cuda.get_device_properties(0).gcnArchName
    return False''',
'''def is_fp8_fnuz() -> bool:
    if not _is_hip:
        return False
    try:
        # only device 0 is checked, this assumes MI300 platforms are homogeneous
        return "gfx94" in getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    except Exception as e:
        logger.warning("Failed to determine fp8 fnuz support: %s", e)
        return False'''),

    # 4. fp8_utils.py use_rowwise_torch_scaled_mm  (get_device_capability None -> is_available gate)
    (f"{SGL}/srt/layers/quantization/fp8_utils.py",
'''def use_rowwise_torch_scaled_mm():
    if _is_hip:''',
'''def use_rowwise_torch_scaled_mm():
    if _is_hip and torch.cuda.is_available():'''),

    # 5. quark_int4fp8_moe.py ON_GFX950  (module-level -> is_available gate)
    (f"{SGL}/srt/layers/quantization/quark_int4fp8_moe.py",
'''    ON_GFX950 = "gfx950" in torch.cuda.get_device_properties("cuda").gcnArchName''',
'''    ON_GFX950 = (
        torch.cuda.is_available()
        and "gfx950" in torch.cuda.get_device_properties("cuda").gcnArchName
    )'''),
]

for i, (path, old, new) in enumerate(patches, 1):
    with open(path) as f:
        src = f.read()
    n = src.count(old)
    assert n == 1, f"sglang import guard {i} ({path}): expected 1 match, found {n}"
    with open(path, "w") as f:
        f.write(src.replace(old, new))
    print(f"sglang import guard {i} applied: {path.split('/')[-1]}")
print("all 5 sglang import guards applied")
