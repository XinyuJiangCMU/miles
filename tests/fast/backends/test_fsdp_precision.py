import sys

import torch
from accelerate import init_empty_weights

from miles.backends.experimental.fsdp_utils.arguments import parse_fsdp_cli
from miles.backends.experimental.fsdp_utils.precision import apply_fp32_master


def test_fp32_master_is_opt_in(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["miles"])
    assert not parse_fsdp_cli().keep_fp32_master

    monkeypatch.setattr(sys, "argv", ["miles", "--keep-fp32-master"])
    assert parse_fsdp_cli().keep_fp32_master


def test_fp32_master_records_checkpoint_dtypes_before_cast():
    model = torch.nn.Linear(4, 4).to(torch.bfloat16)
    model.register_parameter("score_bias", torch.nn.Parameter(torch.zeros(4, dtype=torch.float32)))

    model = apply_fp32_master(model)

    assert all(param.dtype is torch.float32 for param in model.parameters())
    assert model._fsdp_sync_dtypes == {
        "weight": torch.bfloat16,
        "bias": torch.bfloat16,
        "score_bias": torch.float32,
    }


def test_fp32_master_supports_meta_initialization():
    with init_empty_weights():
        model = torch.nn.Linear(4, 4, dtype=torch.bfloat16)

    model = apply_fp32_master(model)

    assert next(model.parameters()).is_meta
    assert all(param.dtype is torch.float32 for param in model.parameters())
    assert model._fsdp_sync_dtypes["weight"] is torch.bfloat16
