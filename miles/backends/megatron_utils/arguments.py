import logging
import os

import torch
from megatron.training.arguments import parse_args, validate_args
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding

# Patch Megatron's get_device_arch_version to handle AMD/ROCm
# On AMD, this function may fail if GPU is not available at parse time (e.g., in Ray)
if torch.version.hip is not None:
    try:
        import megatron.training.utils as _mutils

        _orig_get_device_arch = _mutils.get_device_arch_version

        def _safe_get_device_arch_version():
            try:
                return _orig_get_device_arch()
            except RuntimeError:
                # On AMD in Ray, GPU may not be available during argument parsing
                # Return 9 (MI300X is gfx942, similar to SM 9.x)
                return 9

        _mutils.get_device_arch_version = _safe_get_device_arch_version
        # Also patch in the arguments module which imports it
        import megatron.training.arguments as _margs

        _margs.get_device_arch_version = _safe_get_device_arch_version
    except Exception:
        pass

__all__ = ["validate_args", "parse_args", "set_default_megatron_args"]

logger = logging.getLogger(__name__)


def set_default_megatron_args(args):
    # always use zero optimizer
    args.use_distributed_optimizer = True
    # TODO: maybe change this after megatron has good fp8 support
    args.bf16 = not args.fp16
    # placeholders
    if args.seq_length is None:
        args.seq_length = 4096
    args.max_position_embeddings = args.seq_length
    # Notice(Jiajun): new megatron has removed this argument and use dp_reshardable instead of fully_shard
    if os.getenv("DEPRECATED_MEGATRON_COMPATIBLE", "0") == "1":
        args.dist_ckpt_save_pre_mcore_014 = True
    # compatible for megatron
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"
    return args
