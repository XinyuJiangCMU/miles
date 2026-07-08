"""AMD cross-entropy log-prob via Liger's triton kernel (true_on_policy, no-entropy path).

ROCm override for the DSv4 true_on_policy log-prob. Liger computes -log_softmax[target] with
an in-place gradient, saving two [R, V] materializations. Only the no-entropy path is routed —
the entropy path keeps torch because Liger's in-place overwrite of full_logits would alias the
entropy backward (with_entropy is False whenever entropy_coef == 0, common GRPO default).
Selected by the single platform fork in loss_hub/math_utils.py; NV keeps its torch ce_log_prob.

Returns (log_prob, log_probs_full); log_probs_full is None on the liger fast path (only the
with_entropy=torch path needs it, for the caller's entropy computation).
"""

import torch
from liger_kernel.transformers.functional import liger_cross_entropy


def ce_log_prob(full_logits, tokens, with_entropy, dump_fn):
    if not with_entropy:
        return -liger_cross_entropy(full_logits, tokens, reduction="none"), None
    log_probs_full = torch.log_softmax(full_logits, dim=-1)
    dump_fn("log_probs_full", log_probs_full)
    log_prob = torch.gather(log_probs_full, dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
    return log_prob, log_probs_full
