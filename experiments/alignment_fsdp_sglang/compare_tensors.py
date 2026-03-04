#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from utils.dump_resolver import load_dump_value, resolve_indices
from utils.tensor_specs import (
    ALIGN_NAMES,
    ALIGN_TO_SINGLE_STEP_DEFAULT,
    ALL_COMPARE_NAMES,
    FOCUS_TO_NAMES,
    HF_DROP_LAST_TOKEN_NAMES,
    HF_NAME_OVERRIDE,
    SG_NAME_OVERRIDE,
    SECTION_GROUPS,
    SQUEEZE_BATCH1_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Miles FSDP and SGLang dump tensors.")
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--sg-dir", type=Path, required=True)
    parser.add_argument("--hf-index-json", type=Path, default=None)
    parser.add_argument("--sg-index-json", type=Path, default=None)
    parser.add_argument("--auto-index", action="store_true", default=True)
    parser.add_argument("--focus", choices=sorted(FOCUS_TO_NAMES.keys()), default="full")
    parser.add_argument("--output-txt", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_indices(path: Path | None, dump_dir: Path, side: str) -> dict[str, int]:
    if path is not None and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if side == "hf":
        name_override = HF_NAME_OVERRIDE
    else:
        name_override = SG_NAME_OVERRIDE
    return resolve_indices(dump_dir, ALL_COMPARE_NAMES, name_override)


def load_value(dump_dir: Path, name: str, idx: int):
    path = dump_dir / f"forward_pass_id=0___rank=0___name={name}___dump_index={idx}.pt"
    return load_dump_value(path)


def normalize_for_compare(name: str, x: torch.Tensor, side: str) -> tuple[torch.Tensor, str]:
    rules: list[str] = []
    if name in SQUEEZE_BATCH1_NAMES and x.ndim >= 1 and x.shape[0] == 1:
        x = x[0]
        rules.append("squeeze batch dim 0")
    if side == "hf" and name in HF_DROP_LAST_TOKEN_NAMES:
        if x.ndim == 1 and x.shape[0] > 1:
            x = x[:-1]
            rules.append("drop last token x[:-1]")
        elif x.ndim >= 2 and x.shape[0] > 1:
            x = x[:-1, ...]
            rules.append(f"drop last token x[:-1] from seq_len={x.shape[0] + 1}")
    return x, "; ".join(rules) if rules else "no normalize"


def align_single_step(name: str, x: torch.Tensor) -> tuple[torch.Tensor, str]:
    if name not in ALIGN_NAMES or not ALIGN_TO_SINGLE_STEP_DEFAULT:
        return x, "no align"
    if x.ndim == 3:
        seq_len = x.shape[1]
        return x[:, -1, :], f"take last token x[:, -1, :] from original seq_len={seq_len}"
    if x.ndim == 2 and x.shape[0] > 1:
        seq_len = x.shape[0]
        return x[-1:, :], f"take last token x[-1:, :] from original seq_len={seq_len}"
    return x, "single-step align not needed"


def compare_one(
    name: str,
    hf_dir: Path,
    sg_dir: Path,
    hf_idx: int,
    sg_idx: int,
) -> tuple[list[str], dict]:
    lines: list[str] = [f"[{name}]"]
    payload: dict = {"name": name, "hf_index": hf_idx, "sg_index": sg_idx}
    if hf_idx <= 0 or sg_idx <= 0:
        reason = f"skip (index not set): hf={hf_idx}, sg={sg_idx}"
        lines.append(f"  -> {reason}")
        payload["skip_reason"] = reason
        return lines, payload

    hf_name = HF_NAME_OVERRIDE.get(name, name)
    sg_name = SG_NAME_OVERRIDE.get(name, name)
    x_hf_raw = load_value(hf_dir, hf_name, hf_idx)
    x_sg_raw = load_value(sg_dir, sg_name, sg_idx)
    payload["hf_raw_shape"] = list(x_hf_raw.shape)
    payload["sg_raw_shape"] = list(x_sg_raw.shape)
    lines.append(f"  hf raw shape: {tuple(x_hf_raw.shape)}")
    lines.append(f"  sg raw shape: {tuple(x_sg_raw.shape)}")

    x_hf, hf_norm_rule = normalize_for_compare(name, x_hf_raw, "hf")
    x_sg, sg_norm_rule = normalize_for_compare(name, x_sg_raw, "sg")
    x_hf, hf_align_rule = align_single_step(name, x_hf)
    x_sg, sg_align_rule = align_single_step(name, x_sg)

    lines.append(
        f"  hf slice rule: normalize=({hf_norm_rule}); align=({hf_align_rule}); final_shape={tuple(x_hf.shape)}"
    )
    lines.append(
        f"  sg slice rule: normalize=({sg_norm_rule}); align=({sg_align_rule}); final_shape={tuple(x_sg.shape)}"
    )

    lines.append(f"  hf shape/dtype: {tuple(x_hf.shape)} {x_hf.dtype}")
    lines.append(f"  sg shape/dtype: {tuple(x_sg.shape)} {x_sg.dtype}")
    shape_equal = tuple(x_hf.shape) == tuple(x_sg.shape)
    dtype_equal = x_hf.dtype == x_sg.dtype
    lines.append(f"  shape_equal: {shape_equal}")
    lines.append(f"  dtype_equal: {dtype_equal}")
    payload["shape_equal"] = shape_equal
    payload["dtype_equal"] = dtype_equal
    if not shape_equal:
        lines.append("  -> shape mismatch, skip")
        payload["skip_reason"] = "shape mismatch"
        return lines, payload

    if dtype_equal:
        eq = torch.equal(x_hf, x_sg)
        lines.append(f"  torch.equal: {eq}")
        payload["torch_equal"] = bool(eq)
    else:
        lines.append("  torch.equal: skip (dtype mismatch)")
        payload["torch_equal"] = None

    if x_hf.dtype.is_floating_point or x_sg.dtype.is_floating_point:
        x_hf_f = x_hf.float()
        x_sg_f = x_sg.float()
        value_equal = torch.equal(x_hf_f, x_sg_f)
        diff = (x_hf_f - x_sg_f).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        lines.append(f"  value_equal_after_cast: {value_equal}")
        lines.append(f"  max_abs: {max_abs}")
        lines.append(f"  mean_abs: {mean_abs}")
        payload["value_equal_after_cast"] = bool(value_equal)
        payload["max_abs"] = max_abs
        payload["mean_abs"] = mean_abs
    else:
        neq_cnt = int((x_hf != x_sg).sum().item())
        lines.append(f"  value_equal_after_cast: {neq_cnt == 0}")
        lines.append(f"  neq_cnt: {neq_cnt}")
        payload["value_equal_after_cast"] = neq_cnt == 0
        payload["neq_cnt"] = neq_cnt
    return lines, payload


def main() -> None:
    args = parse_args()
    names = FOCUS_TO_NAMES[args.focus]
    hf_indices = load_indices(args.hf_index_json, args.hf_dir, "hf")
    sg_indices = load_indices(args.sg_index_json, args.sg_dir, "sg")

    all_lines = [
        f"HF_DIR = {args.hf_dir}",
        f"SG_DIR = {args.sg_dir}",
        f"ALIGN_TO_SINGLE_STEP = {ALIGN_TO_SINGLE_STEP_DEFAULT}",
        f"FOCUS = {args.focus}",
        "",
    ]
    results: list[dict] = []
    for title, section_names in SECTION_GROUPS:
        section_subset = [name for name in section_names if name in names]
        if not section_subset:
            continue
        all_lines.append(title)
        for name in section_subset:
            lines, payload = compare_one(
                name=name,
                hf_dir=args.hf_dir,
                sg_dir=args.sg_dir,
                hf_idx=hf_indices.get(name, -1),
                sg_idx=sg_indices.get(name, -1),
            )
            all_lines.extend(lines)
            all_lines.append("")
            results.append(payload)

    text = "\n".join(all_lines).rstrip() + "\n"
    print(text, end="")
    if args.output_txt is not None:
        args.output_txt.parent.mkdir(parents=True, exist_ok=True)
        args.output_txt.write_text(text, encoding="utf-8")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"focus": args.focus, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
