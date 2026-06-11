# SPDX-License-Identifier: Apache-2.0
"""Cross-vendor import shim for TransformerEngine's NVFP4 reference quantizer.

NVIDIA TE exposes NVFP4QuantizerRef at
  transformer_engine.pytorch.custom_recipes.quantization_nvfp4
ROCm TE ships the same reference under the experimental namespace
  transformer_engine.pytorch.experimental.quantization_microblock_ref
This helper resolves whichever one is present so callers stay vendor-agnostic.
"""


def get_nvfp4_quantizer_ref():
    try:
        from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import (
            NVFP4QuantizerRef,
        )

        return NVFP4QuantizerRef
    except ImportError:
        # ROCm TE relocates the reference to the experimental namespace.
        from transformer_engine.pytorch.experimental.quantization_microblock_ref import (
            NVFP4QuantizerRef,
        )

        return NVFP4QuantizerRef
