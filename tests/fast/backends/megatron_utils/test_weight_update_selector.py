"""weight_update_selector: which sglang runners a weight sync covers."""

from argparse import Namespace

import pytest

from miles.backends.megatron_utils.update_weight.common import weight_update_selector


def _args(spec_algo=None, mtp_num_layers=None, megatron_to_hf_mode="raw"):
    return Namespace(
        sglang_speculative_algorithm=spec_algo,
        mtp_num_layers=mtp_num_layers,
        megatron_to_hf_mode=megatron_to_hf_mode,
    )


@pytest.mark.parametrize(
    "args, expected",
    [
        (_args(spec_algo="EAGLE"), "target"),
        (_args(spec_algo="EAGLE", mtp_num_layers=1), "all"),
        # Bridge derives the block from the HF config, so an unset count proves nothing.
        (_args(spec_algo="EAGLE", megatron_to_hf_mode="bridge"), "all"),
        (_args(), "all"),
        (_args(mtp_num_layers=1), "all"),
    ],
)
def test_weight_update_selector(args, expected):
    assert weight_update_selector(args) == expected


def test_missing_attributes_default_to_all():
    assert weight_update_selector(Namespace()) == "all"
