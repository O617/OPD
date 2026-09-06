"""Unit test for chunk-level credit assignment conservation.

Tests both 'semantic' and 'uniform' rules preserve:
    sum_{i in chunk} target_log_probs[i] == teacher_chunk_logp

Also checks that on 1:1 chunks (single-token) both rules give the same
per-token target as the standard teacher logprob.

This test is intentionally self-contained (no verl import) so it can be run
without a GPU / full training env: `python tests/test_opd_credit_conservation.py`.
"""

import math
import unittest

import torch


def apply_credit_rule(
    teacher_log_probs: torch.Tensor,   # (B, T)
    prior_log_probs: torch.Tensor,     # (B, T)
    chunk_ids: torch.Tensor,           # (B, T) long; -1 = sentinel
    response_mask: torch.Tensor,       # (B, T) bool
    credit_rule: str,
) -> torch.Tensor:
    """Reproduces the credit-assignment block of compute_policy_loss_opd.

    Returns target_log_probs of shape (B, T). Sentinel / non-response tokens
    keep their prior_log_probs value.
    """
    sentinel_mask = chunk_ids < 0
    target = prior_log_probs.clone()
    resp_bool = response_mask.bool()
    for b in range(teacher_log_probs.shape[0]):
        seq_valid = resp_bool[b] & (~sentinel_mask[b])
        if not seq_valid.any():
            continue
        for cid in torch.unique(chunk_ids[b][seq_valid]):
            m = seq_valid & (chunk_ids[b] == cid)
            L_T = teacher_log_probs[b][m].sum()
            n = m.sum().clamp_min(1)
            if credit_rule == "uniform":
                target[b][m] = L_T / n
            elif credit_rule == "semantic":
                L_S = prior_log_probs[b][m].sum()
                if torch.abs(L_S) < 1e-8:
                    target[b][m] = L_T / n
                else:
                    target[b][m] = (L_T / L_S) * prior_log_probs[b][m]
            else:
                raise ValueError(credit_rule)
    return target


class TestCreditConservation(unittest.TestCase):
    def _chunk_sum_equals_LT(self, target, teacher, chunk_ids, mask):
        """Assert sum(target[chunk]) == sum(teacher[chunk]) for every chunk."""
        for b in range(target.shape[0]):
            resp = mask[b].bool() & (chunk_ids[b] >= 0)
            for cid in torch.unique(chunk_ids[b][resp]):
                m = resp & (chunk_ids[b] == cid)
                lhs = target[b][m].sum().item()
                rhs = teacher[b][m].sum().item()
                self.assertTrue(
                    math.isclose(lhs, rhs, rel_tol=1e-5, abs_tol=1e-5),
                    f"chunk {cid.item()} not conserved: sum(target)={lhs} vs sum(teacher)={rhs}",
                )

    def test_uniform_conservation(self):
        torch.manual_seed(0)
        B, T = 2, 12
        teacher = -torch.rand(B, T) * 3       # in (-3, 0]
        prior = -torch.rand(B, T) * 3
        chunk_ids = torch.tensor([
            [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, -1],
            [0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4],
        ], dtype=torch.long)
        mask = torch.ones(B, T, dtype=torch.bool)
        target = apply_credit_rule(teacher, prior, chunk_ids, mask, "uniform")
        self._chunk_sum_equals_LT(target, teacher, chunk_ids, mask)

    def test_semantic_conservation(self):
        torch.manual_seed(1)
        B, T = 3, 10
        teacher = -torch.rand(B, T) * 5
        prior = -torch.rand(B, T) * 5 - 1e-3   # avoid degenerate
        chunk_ids = torch.tensor([
            [0, 0, 1, 1, 2, 2, 2, 3, 3, 3],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],   # 1:1 pure
            [0, 0, 0, 0, 1, 2, 3, 3, -1, -1],
        ], dtype=torch.long)
        mask = torch.ones(B, T, dtype=torch.bool)
        target = apply_credit_rule(teacher, prior, chunk_ids, mask, "semantic")
        self._chunk_sum_equals_LT(target, teacher, chunk_ids, mask)

    def test_one_to_one_chunks_recover_teacher_per_token(self):
        """On 1:1 chunks both rules must give target == teacher per token."""
        torch.manual_seed(2)
        B, T = 2, 8
        teacher = -torch.rand(B, T) * 4
        prior = -torch.rand(B, T) * 4 - 1e-3
        chunk_ids = torch.arange(T, dtype=torch.long).unsqueeze(0).expand(B, T).clone()
        mask = torch.ones(B, T, dtype=torch.bool)

        for rule in ("semantic", "uniform"):
            target = apply_credit_rule(teacher, prior, chunk_ids, mask, rule)
            diff = (target - teacher).abs().max().item()
            self.assertLess(
                diff, 1e-5,
                f"{rule}: 1:1 chunk target should equal teacher, max diff {diff}",
            )

    def test_sentinels_ignored(self):
        """Sentinel tokens (chunk_id = -1) keep prior; do not enter any chunk sum."""
        torch.manual_seed(3)
        B, T = 1, 6
        teacher = torch.tensor([[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]])
        prior   = torch.tensor([[-0.5, -0.6, -0.7, -0.8, -0.9, -1.0]])
        chunk_ids = torch.tensor([[0, 0, -1, 1, 1, 1]], dtype=torch.long)
        mask = torch.ones(B, T, dtype=torch.bool)
        for rule in ("semantic", "uniform"):
            target = apply_credit_rule(teacher, prior, chunk_ids, mask, rule)
            self.assertAlmostEqual(target[0, 2].item(), prior[0, 2].item(), places=6,
                                   msg=f"{rule}: sentinel token should be untouched")
            chunk0 = target[0, 0].item() + target[0, 1].item()
            chunk1 = target[0, 3:].sum().item()
            self.assertAlmostEqual(chunk0, teacher[0, 0].item() + teacher[0, 1].item(), places=5)
            self.assertAlmostEqual(chunk1, teacher[0, 3:].sum().item(), places=5)

    def test_degenerate_semantic_fallback_to_uniform(self):
        """If prior chunk logprob ~= 0, semantic must fall back to uniform."""
        B, T = 1, 4
        teacher = torch.tensor([[-2.0, -3.0, -1.0, -4.0]])
        prior   = torch.tensor([[0.0, 0.0, -0.5, -0.5]])  # first chunk sums to 0
        chunk_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        mask = torch.ones(B, T, dtype=torch.bool)
        target = apply_credit_rule(teacher, prior, chunk_ids, mask, "semantic")
        # chunk 0: L_T = -5, n=2 -> uniform fallback -> -2.5 each
        self.assertAlmostEqual(target[0, 0].item(), -2.5, places=5)
        self.assertAlmostEqual(target[0, 1].item(), -2.5, places=5)
        # chunk 1: normal semantic still works
        self.assertAlmostEqual(
            target[0, 2].item() + target[0, 3].item(),
            teacher[0, 2].item() + teacher[0, 3].item(),
            places=5,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
