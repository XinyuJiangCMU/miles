from tests.ci.ci_register import register_amd_ci, register_cuda_ci

# Keep the CUDA lane disabled until the isolated client lifecycle is validated there.
register_cuda_ci(
    est_time=60,
    suite="stage-b-2-gpu-h200",
    labels=[],
    disabled="FIXME: validate the isolated HTTP client lifecycle on CUDA.",
)
register_amd_ci(
    est_time=60,
    suite="stage-b-2-gpu-mi35x",
    labels=[],
)

import pytest

from miles.utils import http_utils
from tests.fast.rollout.inference_rollout.integration.utils import integration_env_config, load_and_call_train

_DATA_ROWS = [{"input": f"What is 1+{i}?", "label": str(1 + i)} for i in range(10)]
_BASE_ARGV = ["--rollout-batch-size", "4", "--n-samples-per-prompt", "2"]


@pytest.fixture(autouse=True)
def reset_http_client(monkeypatch) -> None:
    monkeypatch.setattr(http_utils, "_http_client", None)


@pytest.mark.parametrize(
    "rollout_env,expected_range",
    [
        pytest.param(
            integration_env_config(
                ["--sglang-server-concurrency", "1"] + _BASE_ARGV, data_rows=_DATA_ROWS, latency=0.05
            ),
            (1, 1),
            id="limit_1",
        ),
        pytest.param(
            integration_env_config(
                ["--sglang-server-concurrency", "999"] + _BASE_ARGV, data_rows=_DATA_ROWS, latency=0.05
            ),
            (2, 999),
            id="no_limit",
        ),
    ],
    indirect=["rollout_env"],
)
def test_max_concurrent(rollout_env, expected_range):
    env = rollout_env
    load_and_call_train(env.args, env.data_source)
    min_expected, max_expected = expected_range
    assert min_expected <= env.mock_server.max_concurrent <= max_expected


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
