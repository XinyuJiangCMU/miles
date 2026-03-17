#!/usr/bin/env python3
"""Convert FSDP checkpoint to FP8 quantized HuggingFace format.

Pipeline: FSDP checkpoint -> HuggingFace BF16 -> FP8 quantized

Usage:
    python tools/convert_fsdp_to_fp8.py \
        --input-dir /path/to/fsdp/iter_000001 \
        --output-dir /path/to/output_fp8 \
        --origin-hf-dir /path/to/original/hf/model \
        --strategy tensor  # or block --block-size 128 128

This is especially useful on AMD MI300X where FP8 inference is supported
via normalize_e4m3fn_to_e4m3fnuz at runtime.
"""
import argparse
import os
import shutil
import tempfile

from convert_fsdp_to_hf import _convert_fsdp_to_hf, _detect_model_dir, copy_assets
from convert_hf_to_fp8 import convert_fp8


def main():
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoint to FP8 HuggingFace format")
    parser.add_argument("--input-dir", type=str, required=True, help="FSDP checkpoint directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output FP8 model directory")
    parser.add_argument("--origin-hf-dir", type=str, required=True, help="Original HF model for config/tokenizer")
    parser.add_argument("--strategy", type=str, default="tensor", choices=["block", "channel", "tensor"])
    parser.add_argument("--block-size", type=int, nargs="*", default=None, help="Block size for block strategy")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel workers for FP8 conversion")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} exists. Use --force to overwrite.")

    # Step 1: FSDP -> HF (temporary directory)
    with tempfile.TemporaryDirectory(prefix="miles_fsdp2hf_") as tmp_hf_dir:
        print("=" * 60)
        print("Step 1: FSDP -> HuggingFace BF16")
        print("=" * 60)
        model_dir = _detect_model_dir(args.input_dir)
        _convert_fsdp_to_hf(args.origin_hf_dir, model_dir, tmp_hf_dir)
        copy_assets(args.origin_hf_dir, tmp_hf_dir)

        # Step 2: HF -> FP8
        print("\n" + "=" * 60)
        print(f"Step 2: HuggingFace BF16 -> FP8 ({args.strategy})")
        print("=" * 60)
        os.makedirs(args.output_dir, exist_ok=True)
        convert_fp8(tmp_hf_dir, args.output_dir, args.strategy, args.block_size, args.max_workers)

    print(f"\nFP8 model saved to: {args.output_dir}")
    print("Use with SGLang: python -m sglang.launch_server --model", args.output_dir)


if __name__ == "__main__":
    main()
