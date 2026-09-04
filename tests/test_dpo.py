"""Tests for the self-contained DPO math (no TRL — image has a torch/
transformers version sandwich no TRL release fits)."""

import math

import pytest

try:
    import torch
except (ImportError, OSError) as exc:  # a broken CUDA DLL raises OSError, not ImportError
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from nyaya.dpo import dpo_loss, sequence_logprob


class TestSequenceLogprob:
    def test_sums_only_completion_token_logprobs(self):
        # vocab=4, seq=4: logits at position i predict token i+1
        logits = torch.zeros(1, 4, 4)
        logits[0, :, 2] = 10.0  # model strongly predicts token 2 everywhere
        input_ids = torch.tensor([[0, 1, 2, 2]])
        # prompt_len=2 -> completions are positions 2,3 (tokens 2 and 2),
        # predicted by logits at positions 1 and 2
        lp = sequence_logprob(logits, input_ids, prompt_len=2)
        per_tok = torch.log_softmax(logits[0, 0], dim=-1)[2].item()
        assert lp.item() == pytest.approx(2 * per_tok, rel=1e-4)

    def test_wrong_tokens_score_lower(self):
        logits = torch.zeros(1, 4, 4)
        logits[0, :, 2] = 10.0
        right = torch.tensor([[0, 1, 2, 2]])
        wrong = torch.tensor([[0, 1, 3, 3]])
        assert sequence_logprob(logits, right, 2) > sequence_logprob(logits, wrong, 2)


class TestDpoLoss:
    def test_zero_margin_gives_log2(self):
        z = torch.tensor([0.0])
        loss, margin = dpo_loss(z, z, z, z, beta=0.1)
        assert loss.item() == pytest.approx(math.log(2), rel=1e-4)
        assert margin.item() == pytest.approx(0.0, abs=1e-6)

    def test_policy_preferring_chosen_lowers_loss(self):
        ref = torch.tensor([0.0])
        good_loss, good_margin = dpo_loss(torch.tensor([2.0]), torch.tensor([-2.0]),
                                          ref, ref, beta=0.1)
        bad_loss, bad_margin = dpo_loss(torch.tensor([-2.0]), torch.tensor([2.0]),
                                        ref, ref, beta=0.1)
        assert good_loss < bad_loss
        assert good_margin > 0 > bad_margin

    def test_reference_offsets_cancel(self):
        # improving both logprobs equally vs ref changes nothing
        loss_a, _ = dpo_loss(torch.tensor([1.0]), torch.tensor([0.5]),
                             torch.tensor([0.0]), torch.tensor([-0.5]), beta=0.1)
        loss_b, _ = dpo_loss(torch.tensor([3.0]), torch.tensor([2.5]),
                             torch.tensor([2.0]), torch.tensor([1.5]), beta=0.1)
        assert loss_a.item() == pytest.approx(loss_b.item(), rel=1e-5)
