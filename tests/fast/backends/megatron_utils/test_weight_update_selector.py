"""Which sglang runners a weight sync covers.

The draft is left out only when the training model provably has no MTP block, since it
then receives nothing and a session over it would restore and re-finalize weights that
never changed. Everything else keeps the previous "all".
"""

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
        # Frozen draft: speculative decoding on, no MTP block to send it.
        (_args(spec_algo="EAGLE"), "target"),
        # The trainer has an MTP block, so its weights do reach the draft. Several
        # recipes set the layer count without --enable-mtp-training, which is why the
        # loss flag cannot be the gate.
        (_args(spec_algo="EAGLE", mtp_num_layers=1), "all"),
        # Bridge builds the block from the HF config and leaves the arg unset, so an
        # unset layer count proves nothing there.
        (_args(spec_algo="EAGLE", megatron_to_hf_mode="bridge"), "all"),
        # No speculative decoding: there is no draft runner in the first place.
        (_args(), "all"),
        (_args(mtp_num_layers=1), "all"),
    ],
)
def test_weight_update_selector(args, expected):
    assert weight_update_selector(args) == expected


def test_missing_attributes_default_to_all():
    """Backends that never set these args must keep the pre-existing behaviour."""
    assert weight_update_selector(Namespace()) == "all"
