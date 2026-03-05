from __future__ import annotations

INPUT_NAMES = [
    "input_ids_for_compare",
    "embedding_output",
    "layer0_positions",
]

LAYER0_PREPARE_NAMES = [
    "layer0_attn_input_raw",
    "layer0_attn_after_input_layernorm_only",
    "layer0_attn_input_after_prepare",
    "layer0_hidden_in",
]

LAYER0_ATTN_NAMES = [
    "layer0_q_pre_norm",
    "layer0_k_pre_norm",
    "layer0_v_pre_norm",
    "layer0_q_norm_input_native",
    "layer0_k_norm_input_native",
    "layer0_q_post_norm_native",
    "layer0_k_post_norm_native",
    "layer0_q_post_norm",
    "layer0_k_post_norm",
    "layer0_q_post_rope",
    "layer0_k_post_rope",
    "layer0_attn_context_before_o_proj",
    "layer0_attn_out",
]

LAYER0_BLOCK_NAMES = [
    "layer0_residual",
    "layer0_block_out_before_residual_add",
    "layer0_block_out_after_residual_add",
    "layer0_block_out",
]

# ── Layer 1 ──

LAYER1_PREPARE_NAMES = [
    "layer1_attn_input_raw",
    "layer1_attn_input_after_prepare",
    "layer1_hidden_in",
]

LAYER1_ATTN_NAMES = [
    "layer1_q_pre_norm",
    "layer1_k_pre_norm",
    "layer1_v_pre_norm",
    "layer1_q_post_norm",
    "layer1_k_post_norm",
    "layer1_q_post_rope",
    "layer1_k_post_rope",
    "layer1_attn_context_before_o_proj",
    "layer1_attn_out",
]

LAYER1_BLOCK_NAMES = [
    "layer1_residual",
    "layer1_block_out_before_residual_add",
    "layer1_block_out_after_residual_add",
    "layer1_block_out",
]

# ── Last Layer ──

LAST_LAYER_ATTN_NAMES = [
    "attn_input_last_layer",
    "q_pre_norm",
    "k_pre_norm",
    "v_pre_norm",
    "q_norm_input_native",
    "k_norm_input_native",
    "q_post_norm_native",
    "k_post_norm_native",
    "q_post_norm",
    "k_post_norm",
    "q_post_rope",
    "k_post_rope",
    "attn_context_before_o_proj",
    "attn_out_last_layer",
]

LM_HEAD_NAMES = [
    "final_hidden_before_lm_head",
    "lm_head_weight",
    "next_token_logits_raw",
]

ROPE_NAMES = [
    "layer0_q_post_norm",
    "layer0_k_post_norm",
    "layer0_q_post_rope",
    "layer0_k_post_rope",
    "q_post_norm",
    "k_post_norm",
    "q_post_rope",
    "k_post_rope",
]

ALL_COMPARE_NAMES = (
    INPUT_NAMES
    + LAYER0_PREPARE_NAMES
    + LAYER0_ATTN_NAMES
    + LAYER0_BLOCK_NAMES
    + LAYER1_PREPARE_NAMES
    + LAYER1_ATTN_NAMES
    + LAYER1_BLOCK_NAMES
    + LAST_LAYER_ATTN_NAMES
    + LM_HEAD_NAMES
)

FOCUS_TO_NAMES = {
    "full": ALL_COMPARE_NAMES,
    "layer0": INPUT_NAMES + LAYER0_PREPARE_NAMES + LAYER0_ATTN_NAMES + LAYER0_BLOCK_NAMES,
    "layer1": LAYER1_PREPARE_NAMES + LAYER1_ATTN_NAMES + LAYER1_BLOCK_NAMES,
    "last_layer": LAST_LAYER_ATTN_NAMES + LM_HEAD_NAMES,
    "rope": ROPE_NAMES,
}

SECTION_GROUPS = [
    ("== Inputs ==", INPUT_NAMES),
    ("== Layer 0 Prepare ==", LAYER0_PREPARE_NAMES),
    ("== Layer 0 Attention ==", LAYER0_ATTN_NAMES),
    ("== Layer 0 Residual / Block ==", LAYER0_BLOCK_NAMES),
    ("== Layer 1 Prepare ==", LAYER1_PREPARE_NAMES),
    ("== Layer 1 Attention ==", LAYER1_ATTN_NAMES),
    ("== Layer 1 Residual / Block ==", LAYER1_BLOCK_NAMES),
    ("== Last Layer Attention ==", LAST_LAYER_ATTN_NAMES),
    ("== LM Head ==", LM_HEAD_NAMES),
]

HF_NAME_OVERRIDE = {
    "layer0_attn_after_input_layernorm_only": "layer0_attn_input_after_prepare",
}

SG_NAME_OVERRIDE = {
    "layer0_q_norm_input_native": "layer0_q_pre_norm",
    "layer0_k_norm_input_native": "layer0_k_pre_norm",
    "layer0_q_post_norm_native": "layer0_q_post_norm",
    "layer0_k_post_norm_native": "layer0_k_post_norm",
    "q_norm_input_native": "q_pre_norm",
    "k_norm_input_native": "k_pre_norm",
    "q_post_norm_native": "q_post_norm",
    "k_post_norm_native": "k_post_norm",
}

HF_DROP_LAST_TOKEN_NAMES = {
    "input_ids_for_compare",
    "embedding_output",
    "layer0_positions",
}

ALIGN_TO_SINGLE_STEP_DEFAULT = True

ALIGN_NAMES = {
    # layer 0
    "layer0_hidden_in",
    "layer0_q_pre_norm",
    "layer0_k_pre_norm",
    "layer0_v_pre_norm",
    "layer0_q_norm_input_native",
    "layer0_k_norm_input_native",
    "layer0_q_post_norm_native",
    "layer0_k_post_norm_native",
    "layer0_q_post_norm",
    "layer0_k_post_norm",
    "layer0_q_post_rope",
    "layer0_k_post_rope",
    "layer0_residual",
    "layer0_attn_context_before_o_proj",
    "layer0_attn_out",
    "layer0_block_out_before_residual_add",
    "layer0_block_out_after_residual_add",
    "layer0_block_out",
    # layer 1
    "layer1_hidden_in",
    "layer1_q_pre_norm",
    "layer1_k_pre_norm",
    "layer1_v_pre_norm",
    "layer1_q_post_norm",
    "layer1_k_post_norm",
    "layer1_q_post_rope",
    "layer1_k_post_rope",
    "layer1_residual",
    "layer1_attn_context_before_o_proj",
    "layer1_attn_out",
    "layer1_block_out_before_residual_add",
    "layer1_block_out_after_residual_add",
    "layer1_block_out",
    # last layer
    "attn_input_last_layer",
    "q_pre_norm",
    "k_pre_norm",
    "v_pre_norm",
    "q_norm_input_native",
    "k_norm_input_native",
    "q_post_norm_native",
    "k_post_norm_native",
    "q_post_norm",
    "k_post_norm",
    "q_post_rope",
    "k_post_rope",
    "attn_context_before_o_proj",
    "attn_out_last_layer",
    "final_hidden_before_lm_head",
    "next_token_logits_raw",
}

# next_token_logits_raw is NOT in SQUEEZE_BATCH1_NAMES:
# HF=(1,14,vocab) → align takes x[:,−1,:] → (1,vocab)
# SG=(1,vocab) → no align needed → (1,vocab)
# If squeezed, SG becomes (vocab,) [1D] which breaks shape matching.
SQUEEZE_BATCH1_NAMES = {
    "input_ids_for_compare",
    "embedding_output",
    "layer0_positions",
    # layer 0
    "layer0_attn_input_raw",
    "layer0_attn_after_input_layernorm_only",
    "layer0_attn_input_after_prepare",
    "layer0_hidden_in",
    "layer0_q_pre_norm",
    "layer0_k_pre_norm",
    "layer0_v_pre_norm",
    "layer0_q_norm_input_native",
    "layer0_k_norm_input_native",
    "layer0_q_post_norm_native",
    "layer0_k_post_norm_native",
    "layer0_q_post_norm",
    "layer0_k_post_norm",
    "layer0_q_post_rope",
    "layer0_k_post_rope",
    "layer0_residual",
    "layer0_attn_context_before_o_proj",
    "layer0_attn_out",
    "layer0_block_out_before_residual_add",
    "layer0_block_out_after_residual_add",
    "layer0_block_out",
    # layer 1
    "layer1_attn_input_raw",
    "layer1_attn_input_after_prepare",
    "layer1_hidden_in",
    "layer1_q_pre_norm",
    "layer1_k_pre_norm",
    "layer1_v_pre_norm",
    "layer1_q_post_norm",
    "layer1_k_post_norm",
    "layer1_q_post_rope",
    "layer1_k_post_rope",
    "layer1_residual",
    "layer1_attn_context_before_o_proj",
    "layer1_attn_out",
    "layer1_block_out_before_residual_add",
    "layer1_block_out_after_residual_add",
    "layer1_block_out",
    # last layer
    "attn_input_last_layer",
    "q_pre_norm",
    "k_pre_norm",
    "v_pre_norm",
    "q_norm_input_native",
    "k_norm_input_native",
    "q_post_norm_native",
    "k_post_norm_native",
    "q_post_norm",
    "k_post_norm",
    "q_post_rope",
    "k_post_rope",
    "attn_context_before_o_proj",
    "attn_out_last_layer",
    "final_hidden_before_lm_head",
}
