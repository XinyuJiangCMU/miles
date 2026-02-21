import logging
import os
from pathlib import Path

import torch

from miles.rollout.data_source import DataSource
from miles.utils.types import Sample

from .utils import load_result_paths, maybe_shuffle

logger = logging.getLogger(__name__)


class HarborResultDataSource(DataSource):
    """
    Minimal data source for Harbor result.json files.

    Use --prompt-data to point to:
    - a directory containing result.json files (recursively),
    - a .txt/.list file with one path per line,
    - a .jsonl file with { "result_path": "..."} entries,
    - a .json file with a list of paths.
    """

    def __init__(self, args):
        self.args = args
        self._paths = load_result_paths(args.prompt_data)
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0

        if getattr(args, "rollout_shuffle", False):
            self._paths = maybe_shuffle(self._paths, args.rollout_seed, self.epoch_id)

        # Expose for RolloutManager.get_num_rollout_per_epoch
        self.dataset = self._paths

    def next_result_paths(self, num_samples: int) -> list[Path]:
        if num_samples <= 0:
            return []

        if num_samples > len(self._paths):
            raise ValueError(
                f"Requested num_samples={num_samples} exceeds available Harbor results ({len(self._paths)})."
            )

        if self.sample_offset + num_samples <= len(self._paths):
            paths = self._paths[self.sample_offset : self.sample_offset + num_samples]
            self.sample_offset += num_samples
            return paths

        # Wrap to next epoch
        remaining = len(self._paths) - self.sample_offset
        if remaining <= 0:
            remaining = 0

        paths = self._paths[self.sample_offset :]
        num_samples -= len(paths)
        self.epoch_id += 1
        if getattr(self.args, "rollout_shuffle", False):
            self._paths = maybe_shuffle(self._paths, self.args.rollout_seed, self.epoch_id)

        take = min(num_samples, len(self._paths))
        paths += self._paths[:take]
        self.sample_offset = take
        return paths

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return placeholder Sample groups. The Harbor rollout loader does not use these,
        but some code paths expect this method to exist.
        """
        samples = []
        for _ in range(num_samples):
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = Sample()
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        # No-op for offline Harbor results.
        return

    def save(self, rollout_id):
        if self.args.save is None:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
        }
        path = os.path.join(self.args.save, f"rollout/harbor_result_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/harbor_result_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)

        if getattr(self.args, "rollout_shuffle", False):
            self._paths = maybe_shuffle(self._paths, self.args.rollout_seed, self.epoch_id)
