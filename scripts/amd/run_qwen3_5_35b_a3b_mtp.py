from dataclasses import replace


def configure_case(case):
    return replace(
        case,
        moe_token_dispatcher_type="alltoall",
        extra_sglang_args="--sglang-disable-shared-experts-fusion ",
    )
