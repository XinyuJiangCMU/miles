# Adapt from https://github.com/alibaba/Pai-Megatron-Patch/blob/2b201af08336dea0403df7c6b497c964cf5a2e75/toolkits/model_checkpoints_convertor/deepseek/fp8_cast_bf16.py
import json
import os
from argparse import ArgumentParser
from glob import glob

import torch
import triton
import triton.language as tl
from param_name_remap import get_param_name_remap
from safetensors.torch import load_file, save_file
from tqdm import tqdm


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))

    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# e2m1 code -> value. The index is the 4-bit code; the high bit carries the sign.
_FP4_E2M1_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_FP4_E2M1_TABLE = torch.tensor(
    _FP4_E2M1_MAGNITUDES + [-v for v in _FP4_E2M1_MAGNITUDES], dtype=torch.float32
)


def weight_dequant_fp4(x: torch.Tensor, s: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Dequantize an MXFP4 weight: each int8 packs two e2m1 values, low nibble first."""
    assert x.dim() == 2 and s.dim() == 2
    packed = x.view(torch.uint8)
    codes = torch.stack([packed & 0x0F, packed >> 4], dim=-1)
    y = _FP4_E2M1_TABLE.to(x.device)[codes.long()].flatten(1)
    assert s.size(0) == y.size(0) and s.size(1) * group_size == y.size(1)
    return (y * s.float().repeat_interleave(group_size, dim=1)).to(torch.get_default_dtype())


def main(fp8_path, bf16_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    os.system("cp -rf " + fp8_path + "/config.json " + bf16_path)
    os.system("cp -rf " + fp8_path + "/*.py " + bf16_path)
    os.system("cp -rf " + fp8_path + "/tokenizer* " + bf16_path)
    os.system("cp -rf " + fp8_path + "/chat_template* " + bf16_path)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file) as f:
        model_index = json.load(f)
    weight_map_raw = model_index["weight_map"]

    remap = get_param_name_remap(os.path.join(fp8_path, "config.json"), weight_map_raw)
    weight_map_renamed = {}
    raw_name_by_renamed = {}
    for raw_name, file_name in weight_map_raw.items():
        renamed_name = remap(raw_name)
        assert renamed_name not in raw_name_by_renamed, (
            f"Remapped tensor name collision: {renamed_name} from "
            f"{raw_name} and {raw_name_by_renamed[renamed_name]}"
        )
        weight_map_renamed[renamed_name] = file_name
        raw_name_by_renamed[renamed_name] = raw_name

    # Cache for loaded safetensor files
    loaded_files = {}
    dequantized_weight_names = []

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        raw_tensor_name = raw_name_by_renamed[tensor_name]
        file_name = weight_map_raw[raw_tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cuda")

        return loaded_files[file_name][raw_tensor_name]

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        print(f"Handling file: {safetensor_file}")
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for weight_name_raw, weight in current_state_dict.items():
            weight_name = remap(weight_name_raw)

            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1:  # FP8 or packed-FP4 weight
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    dequantized_weight_names.append(weight_name)
                    # Packed FP4 and FP8 weights are both one byte per element, so the
                    # scale dtype is what tells them apart.
                    if scale_inv.dtype == torch.float8_e8m0fnu:
                        new_state_dict[weight_name] = weight_dequant_fp4(weight, scale_inv)
                    else:
                        new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight

        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()

    # Update model index
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in dequantized_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map_renamed:
            weight_map_renamed.pop(scale_inv_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map_renamed}, f, indent=2)

    # `expert_dtype` describes the on-disk expert precision, which is BF16 after the cast.
    config_file = os.path.join(bf16_path, "config.json")
    with open(config_file) as f:
        config = json.load(f)
    if config.pop("expert_dtype", None) is not None:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
