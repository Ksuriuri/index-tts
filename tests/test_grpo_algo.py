"""Unit tests for the algorithm-only helpers in train_gpt_v2_grpo_multigpu.

Covers: compute_advantages, compute_grpo_loss, gpt_per_token_logp mask math,
and the metadata parsing helpers.  Pure CPU, no model load.
"""

import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trainers"))

# Avoid touching wandb / accelerate at import time.
os.environ.setdefault("WANDB_MODE", "disabled")

import train_gpt_v2_grpo_multigpu as G


# ------------------------------------------------------------------ #
# compute_advantages
# ------------------------------------------------------------------ #

def test_compute_advantages_intra_group_basic():
    # 2 groups of 2 each: (1, 0) and (1, 1).  Second group has zero std -> dropped.
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])
    group_idx = torch.tensor([0, 0, 1, 1])
    adv, valid = G.compute_advantages(rewards, group_idx, num_groups=2,
                                      norm_strategy="intra_group")
    assert torch.allclose(adv, torch.tensor([1.0, -1.0, 0.0, 0.0]), atol=1e-4), adv
    assert valid.tolist() == [True, True, False, False], valid
    print("[OK] compute_advantages intra_group basic:", adv.tolist(), "valid=", valid.tolist())


def test_compute_advantages_global_batch_basic():
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    group_idx = torch.tensor([0, 0, 1, 1, 2, 2])
    adv, valid = G.compute_advantages(rewards, group_idx, num_groups=3,
                                      norm_strategy="global_batch")
    expected = torch.tensor([1.0, -1.0, 0.0, 0.0, 1.0, -1.0])
    assert torch.allclose(adv, expected, atol=1e-4), (adv, expected)
    assert valid.tolist() == [True, True, False, False, True, True], valid
    print("[OK] compute_advantages global_batch basic:", adv.tolist(), "valid=", valid.tolist())


def test_compute_advantages_all_groups_degenerate():
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    group_idx = torch.tensor([0, 0, 1, 1])
    adv, valid = G.compute_advantages(rewards, group_idx, num_groups=2,
                                      norm_strategy="global_batch")
    assert torch.allclose(adv, torch.zeros_like(adv)), adv
    assert valid.any().item() is False, valid
    print("[OK] compute_advantages all_zero variance handled:", adv.tolist())


def test_compute_advantages_imbalanced_global_batch():
    rewards = torch.tensor([1.0, 1.0, 0.0])
    group_idx = torch.tensor([0, 0, 0])
    adv, valid = G.compute_advantages(rewards, group_idx, num_groups=1,
                                      norm_strategy="global_batch")
    centred = torch.tensor([1/3, 1/3, -2/3])
    expected = (centred - centred.mean()) / (centred.std(unbiased=False) + 1e-8)
    assert torch.allclose(adv, expected, atol=1e-4), (adv, expected)
    assert valid.tolist() == [True, True, True]
    print("[OK] compute_advantages global_batch unbalanced:", adv.tolist())


def test_compute_advantages_centred_zero_sample_kept_after_fix():
    """After the fix: a candidate whose reward equals the group mean (centred == 0)
    must still be marked as a valid sample, because validity is group-level.
    """
    rewards = torch.tensor([2.0, 1.0, 0.0])  # mean=1.0 -> centred=[+1, 0, -1]
    group_idx = torch.tensor([0, 0, 0])
    adv, valid = G.compute_advantages(rewards, group_idx, num_groups=1,
                                      norm_strategy="global_batch")
    # After global standardisation over [+1, 0, -1] (variance != 0), the middle
    # element stays exactly 0, but it is still flagged as valid.
    assert valid.tolist() == [True, True, True], (adv, valid)
    print("[OK] compute_advantages keeps centred==0 sample as valid:",
          adv.tolist(), "valid=", valid.tolist())


# ------------------------------------------------------------------ #
# compute_grpo_loss
# ------------------------------------------------------------------ #

def test_compute_grpo_loss_zero_when_policy_equals_ref():
    """At initialisation logp_pi == logp_ref, so:
      - ratio = 1
      - surr1 = surr2 = A -> pg = -A
      - kl(k3) = exp(0) - 0 - 1 = 0
    The reported KL must be 0; policy_loss must equal -mean(A_per_sample).
    """
    B, T = 4, 6
    torch.manual_seed(0)
    logp = -torch.rand(B, T)
    mask = torch.ones(B, T)
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])

    loss, metrics = G.compute_grpo_loss(
        logp_policy=logp,
        logp_ref=logp,
        mask=mask,
        advantages=advantages,
        clip_eps=0.2,
        kl_coeff=0.04,
        kl_estimator="k3",
    )

    assert torch.isclose(metrics["kl"], torch.tensor(0.0), atol=1e-6), metrics["kl"]
    assert torch.isclose(metrics["ratio_mean"], torch.tensor(1.0), atol=1e-6), metrics["ratio_mean"]
    # policy_loss = -mean_per_valid_sample = -mean([1, -1, 1, -1]) = 0
    assert torch.isclose(metrics["policy_loss"], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    print("[OK] grpo loss is zero at init when advantages are symmetric:", metrics)


def test_compute_grpo_loss_gradient_signal_at_init():
    """The same setup with asymmetric advantages should give non-zero loss but
    still zero KL.
    """
    B, T = 2, 5
    logp = torch.zeros(B, T, requires_grad=True)
    mask = torch.ones(B, T)
    advantages = torch.tensor([1.0, -2.0])
    loss, metrics = G.compute_grpo_loss(
        logp_policy=logp,
        logp_ref=logp.detach(),
        mask=mask,
        advantages=advantages,
        clip_eps=0.2,
        kl_coeff=0.04,
        kl_estimator="k3",
    )
    # policy_loss = -mean([1, -2]) = 0.5
    assert torch.isclose(metrics["policy_loss"], torch.tensor(0.5), atol=1e-5), metrics
    assert torch.isclose(metrics["kl"], torch.tensor(0.0), atol=1e-6), metrics
    # gradient w.r.t. logp_policy should be non-zero
    loss.backward()
    assert logp.grad is not None
    assert logp.grad.abs().sum().item() > 0
    print("[OK] grpo loss has gradient at init for asymmetric advantages:",
          metrics, "grad_norm=", logp.grad.norm().item())


def test_compute_grpo_loss_clipping_kicks_in():
    """If policy and ref differ a lot, the clip should bound the surrogate."""
    B, T = 2, 4
    logp_pi = torch.zeros(B, T)
    logp_ref = -torch.ones(B, T)  # ratio = e ~ 2.71, clipped to 1.2
    mask = torch.ones(B, T)
    advantages = torch.tensor([1.0, -1.0])

    loss, metrics = G.compute_grpo_loss(
        logp_policy=logp_pi,
        logp_ref=logp_ref,
        mask=mask,
        advantages=advantages,
        clip_eps=0.2,
        kl_coeff=0.0,
        kl_estimator="k3",
    )
    # For sample 0 (A>0): min(2.71*1, 1.2*1)  = 1.2  -> pg = -1.2
    # For sample 1 (A<0): min(2.71*-1, 1.2*-1) = -2.71 -> pg = +2.71
    # mean = (-1.2 + 2.71)/2 = 0.755
    expected_pg = (-1.2 + 2.71) / 2
    assert abs(metrics["policy_loss"].item() - expected_pg) < 1e-2, metrics
    print("[OK] grpo loss clipping correct:", metrics)


def test_compute_grpo_loss_zero_advantages_skip():
    """If every advantage is zero (via explicit sample_valid all False), loss
    and gradient must be zero."""
    B, T = 3, 5
    logp_pi = torch.randn(B, T, requires_grad=True)
    logp_ref = torch.randn(B, T)
    mask = torch.ones(B, T)
    advantages = torch.zeros(B)
    sample_valid = torch.zeros(B, dtype=torch.bool)

    loss, metrics = G.compute_grpo_loss(
        logp_policy=logp_pi,
        logp_ref=logp_ref,
        mask=mask,
        advantages=advantages,
        clip_eps=0.2,
        kl_coeff=0.1,
        kl_estimator="k3",
        sample_valid=sample_valid,
    )
    assert metrics["valid_samples"].item() == 0.0, metrics["valid_samples"]
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), loss
    loss.backward()
    assert logp_pi.grad is not None and logp_pi.grad.abs().sum().item() == 0
    print("[OK] grpo loss is zero when sample_valid is all False")


def test_kl_k1_vs_k3():
    """At policy == ref, both k1 and k3 must yield kl==0.  Otherwise k1 can be
    negative for individual samples while k3 is always >= 0.
    """
    B, T = 2, 4
    logp_pi = torch.tensor([[-0.5] * T, [-1.0] * T])
    logp_ref = torch.tensor([[-1.0] * T, [-0.5] * T])
    mask = torch.ones(B, T)
    advantages = torch.tensor([1.0, -1.0])

    _, m_k1 = G.compute_grpo_loss(logp_pi, logp_ref, mask, advantages,
                                  clip_eps=0.2, kl_coeff=1.0, kl_estimator="k1")
    _, m_k3 = G.compute_grpo_loss(logp_pi, logp_ref, mask, advantages,
                                  clip_eps=0.2, kl_coeff=1.0, kl_estimator="k3")
    # k3 KL is always non-negative when averaged over valid samples
    assert m_k3["kl"].item() >= -1e-6, m_k3
    # k1 reported KL can be negative for asymmetric setups
    print("[OK] k1 vs k3:", "k1=", m_k1["kl"].item(), "k3=", m_k3["kl"].item())


# ------------------------------------------------------------------ #
# Metadata helpers
# ------------------------------------------------------------------ #

def test_emotion_tag_parsing():
    text = "[🤯#Sadness:2;Surprise:5] What a pity. Indeed."
    cleaned, vec = G.parse_text_emotion_tags(text)
    assert cleaned == "What a pity. Indeed.", cleaned
    assert vec is not None
    assert vec.shape == (8,)
    # Sadness:2 -> 0.2, Surprise:5 -> 0.5
    idx_sad = G.EMOTION_INDEX["sadness"]
    idx_sur = G.EMOTION_INDEX["surprise"]
    assert math.isclose(vec[idx_sad].item(), 0.2, abs_tol=1e-6)
    assert math.isclose(vec[idx_sur].item(), 0.5, abs_tol=1e-6)
    print("[OK] emotion tag parsing:", cleaned, vec.tolist())


def test_emotion_tag_alias_happy():
    text = "[#Happy:7] Yay yay yay"
    _, vec = G.parse_text_emotion_tags(text)
    assert vec is not None
    assert math.isclose(vec[G.EMOTION_INDEX["joy"]].item(), 0.7, abs_tol=1e-6)
    print("[OK] emotion 'happy' alias maps to joy")


def test_parse_metadata_group_current_schema():
    item = {
        "target_text": "你好世界",
        "voice_id": "0005e877",
        "chosen": [{"file": "chosen/foo.flac", "gen_product_id": "foo"}],
        "rejected": [{"file": "rejected/bar.flac"}, {"file": "rejected/baz.flac"}],
        "total_count": 3,
    }
    entry = G._parse_metadata_group(item)
    assert entry is not None
    assert entry["target_text"] == "你好世界"
    assert entry["voice_id"] == "0005e877"
    assert entry["ref_audio_stem"] == "0005e877"
    assert entry["chosen_files"] == ["chosen/foo.flac"]
    assert entry["rejected_files"] == ["rejected/bar.flac", "rejected/baz.flac"]
    assert entry["group_count"] == 3
    print("[OK] metadata parsing - current schema:", entry["voice_id"],
          "chosen=", entry["chosen_files"], "rejected=", entry["rejected_files"])


def test_parse_metadata_group_rejects_no_pair():
    item = {
        "target_text": "x",
        "voice_id": "v",
        "chosen": [{"file": "a.flac"}],
        "rejected": [],
        "total_count": 1,
    }
    assert G._parse_metadata_group(item) is None
    print("[OK] metadata parsing - drops groups with only one side")


# ------------------------------------------------------------------ #
# Runner
# ------------------------------------------------------------------ #

def main():
    tests = [
        test_compute_advantages_intra_group_basic,
        test_compute_advantages_global_batch_basic,
        test_compute_advantages_all_groups_degenerate,
        test_compute_advantages_imbalanced_global_batch,
        test_compute_advantages_centred_zero_sample_kept_after_fix,
        test_compute_grpo_loss_zero_when_policy_equals_ref,
        test_compute_grpo_loss_gradient_signal_at_init,
        test_compute_grpo_loss_clipping_kicks_in,
        test_compute_grpo_loss_zero_advantages_skip,
        test_kl_k1_vs_k3,
        test_emotion_tag_parsing,
        test_emotion_tag_alias_happy,
        test_parse_metadata_group_current_schema,
        test_parse_metadata_group_rejects_no_pair,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {t.__name__}: {e!r}")
            failures.append(t.__name__)
    if failures:
        print(f"\n=== {len(failures)} test(s) failed: {failures} ===")
        sys.exit(1)
    print(f"\n=== All {len(tests)} tests passed ===")


if __name__ == "__main__":
    main()
