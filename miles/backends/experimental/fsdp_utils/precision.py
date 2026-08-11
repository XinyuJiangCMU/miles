import torch


def apply_fp32_master(model):
    sync_dtypes = {name: value.dtype for name, value in model.state_dict().items()}
    model = model.to(torch.float32)
    model._fsdp_sync_dtypes = sync_dtypes
    return model
