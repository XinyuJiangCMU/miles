"""AMD q-RMSNorm via Liger's triton kernel (fwd+bwd).

ROCm override for the DSv4 weightless inline q-RMSNorm. gemma casting = fp32 internal
then cast back to input dtype, matching the NV torch path bit-wise (bf16). Selected by the
single platform fork in models/deepseek_v4/deepseek_v4.py; NV keeps its torch q_rmsnorm.

NOTE: only q_norm is routed here. compressor.norm is intentionally NOT — it is parity-pinned
to SGLang's compressor norm (pure fp32); routing it through triton risks rollout<->train
divergence for little (<1%) gain.
"""

from liger_kernel.transformers.functional import liger_rms_norm


def q_rmsnorm(q, eps):
    return liger_rms_norm(q, None, eps, offset=0.0, casting_mode="gemma", in_place=True)
