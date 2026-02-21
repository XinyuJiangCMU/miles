import logging
from pathlib import Path
from typing import Any

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.utils.types import Sample

from .data_source import HarborResultDataSource
from .utils import build_tokens_from_rollout_detail, flatten_logprobs, read_trial_result

logger = logging.getLogger(__name__)


def generate_rollout(
    args, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """
    Minimal Harbor rollout loader.

    Expects Harbor `result.json` files containing `agent_result.rollout_details`.
    """
    if not isinstance(data_source, HarborResultDataSource):
        raise TypeError(
            "Harbor rollout expects data_source to be HarborResultDataSource. "
            "Set --data-source-path=miles_plugins.harbor.data_source.HarborResultDataSource."
        )

    num_samples = args.rollout_batch_size * args.n_samples_per_prompt
    result_paths = data_source.next_result_paths(num_samples)

    samples = _convert_results_to_samples(args, result_paths, data_source)
    grouped = _group_samples(samples, args.n_samples_per_prompt)

    if evaluation:
        reward_key = args.eval_reward_key or args.reward_key
        flat = [s for group in grouped for s in group]
        return RolloutFnEvalOutput(
            data={
                "harbor": {
                    "rewards": [_extract_reward(s, reward_key) for s in flat],
                    "truncated": [s.status == Sample.Status.TRUNCATED for s in flat],
                    "samples": flat,
                }
            }
        )

    return RolloutFnTrainOutput(samples=grouped)


def _convert_results_to_samples(
    args, result_paths: list[Path], data_source: HarborResultDataSource
) -> list[Sample]:
    samples: list[Sample] = []
    group_index = data_source.sample_group_index
    sample_index = data_source.sample_index

    for i, path in enumerate(result_paths):
        trial = read_trial_result(path)

        agent_result = trial.get("agent_result") or {}
        rollout_details = agent_result.get("rollout_details") or []
        if not rollout_details:
            raise ValueError(
                f"Missing agent_result.rollout_details in Harbor result: {path}"
            )

        detail = rollout_details[0]
        prompt_token_ids = detail.get("prompt_token_ids") or []
        completion_token_ids = detail.get("completion_token_ids") or []
        logprobs = detail.get("logprobs")

        tokens, response_length = build_tokens_from_rollout_detail(
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
        )

        sample = Sample()
        sample.tokens = tokens
        sample.response_length = response_length
        sample.loss_mask = [1] * response_length
        sample.rollout_log_probs = flatten_logprobs(logprobs)

        verifier_result = trial.get("verifier_result") or {}
        rewards = verifier_result.get("rewards")
        if rewards is None:
            rewards = 0.0
        sample.reward = rewards

        if trial.get("exception_info") is not None:
            sample.status = Sample.Status.FAILED
        else:
            sample.status = Sample.Status.COMPLETED

        sample.metadata = {
            "harbor_result_path": str(path),
            "harbor_task_name": trial.get("task_name"),
            "harbor_trial_name": trial.get("trial_name"),
            "harbor_task_checksum": trial.get("task_checksum"),
        }

        sample.index = sample_index
        sample.group_index = group_index

        samples.append(sample)

        sample_index += 1
        if (i + 1) % args.n_samples_per_prompt == 0:
            group_index += 1

    data_source.sample_index = sample_index
    data_source.sample_group_index = group_index

    return samples


def _group_samples(samples: list[Sample], group_size: int) -> list[list[Sample]]:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    if len(samples) % group_size != 0:
        raise ValueError(
            f"Number of samples ({len(samples)}) is not divisible by n_samples_per_prompt ({group_size})."
        )

    grouped = []
    for i in range(0, len(samples), group_size):
        grouped.append(samples[i : i + group_size])
    return grouped


def _extract_reward(sample: Sample, reward_key: str | None):
    if not reward_key:
        return sample.reward
    if isinstance(sample.reward, dict):
        return sample.reward.get(reward_key)
    return sample.reward
