from utils.tensor_specs import ALIGN_NAMES, FOCUS_TO_NAMES


def test_focus_contains_layer0_core_names():
    names = set(FOCUS_TO_NAMES["layer0"])
    assert "layer0_hidden_in" in names
    assert "layer0_q_post_norm" in names
    assert "layer0_block_out" in names


def test_align_names_contains_layer0_qk_chain():
    assert "layer0_q_pre_norm" in ALIGN_NAMES
    assert "layer0_k_pre_norm" in ALIGN_NAMES
    assert "layer0_q_post_rope" in ALIGN_NAMES
