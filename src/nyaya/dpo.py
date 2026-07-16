"""Self-contained DPO math (Rafailov et al. 2023) — no TRL dependency.

The cluster image pins transformers 5.x with torch 2.4: every TRL release's
DPOTrainer needs either older transformers or newer torch, so the loss and
the log-prob accounting live here instead (they are ~40 lines), unit-tested
against synthetic tensors. scripts/23 owns the training loop.
"""

import torch
import torch.nn.functional as F


def sequence_logprob(logits: torch.Tensor, input_ids: torch.Tensor,
                     prompt_len: int) -> torch.Tensor:
    """Sum of log P(token) over the completion tokens only.

    logits: [B, T, V]; input_ids: [B, T]. Position i's logits predict token
    i+1, so completion tokens (positions prompt_len..T-1) are scored by
    logits at positions prompt_len-1..T-2.
    """
    logprobs = torch.log_softmax(logits[:, prompt_len - 1:-1, :].float(), dim=-1)
    targets = input_ids[:, prompt_len:]
    picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return picked.sum(dim=-1)


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
             beta: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard sigmoid DPO. Returns (mean loss, mean reward margin)."""
    chosen_reward = beta * (policy_chosen - ref_chosen)
    rejected_reward = beta * (policy_rejected - ref_rejected)
    margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(margin)
    return loss.mean(), margin.detach().mean()
