import logging

import torch
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync

logger = logging.getLogger(__name__)

_ROCM_WARNED = False


class ROCmFileSystemWriterAsync(FileSystemWriterAsync):
    """
    FileSystemWriterAsync wrapper for ROCm compatibility.

    On ROCm/HIP, using non_blocking=True causes tensors to be stored in pinned memory,
    which triggers segmentation faults when forking subprocesses afterward.
    """

    @staticmethod
    def preload_tensors(*args, **kwargs):
        global _ROCM_WARNED
        if torch.version.hip:
            if not _ROCM_WARNED:
                logger.info("ROCm: setting non_blocking=False in checkpoint preload_tensors")
                _ROCM_WARNED = True
            if "non_blocking" in kwargs:
                kwargs["non_blocking"] = False
            elif len(args) > 1 and isinstance(args[-1], bool):
                args = args[:-1] + (False,)

        return FileSystemWriterAsync.preload_tensors(*args, **kwargs)
