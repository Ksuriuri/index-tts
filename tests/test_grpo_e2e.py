"""End-to-end smoke test for train_gpt_v2_grpo_multigpu.

Single-process, single-GPU.  Builds the real UnifiedVoice policy + reference,
the feature preprocessor, materialises a couple of real groups from the
multigen jsonl, runs prepare_grpo_batch -> wrapper(batch, adv) -> backward,
and verifies:

  * loss is finite
  * KL is ~ 0 on the first step (policy == ref)
  * ratio is ~ 1 on the first step
  * gradients are non-zero on trainable params and exactly zero on frozen params
  * a second step after a (tiny) optimiser update produces a non-zero KL
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trainers"))

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import train_gpt_v2_grpo_multigpu as G  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


MODEL_DIR = Path("checkpoints/IndexTTS-2-vLLM")
CONFIG = MODEL_DIR / "config.yaml"
TOKENIZER = MODEL_DIR / "jp_es_bpe.model"
CKPT = MODEL_DIR / "gpt.pth"

METADATA = Path("/mnt/data_3t_1/datasets/raw_data/noiz-v2/multigen/metadata_v2.jsonl")
AUDIO_ROOT = Path("/mnt/data_3t_1/datasets/raw_data/noiz-v2/multigen")
REF_AUDIO_ROOT = Path("/mnt/data_3t_1/datasets/raw_data/noiz-v2/ref_audios")


def get_one_real_group(tokenizer, max_group_size=4) -> G.GroupItem:
    """Materialise the first valid group from a small slice of the jsonl."""
    tmp = Path("/tmp/grpo_e2e_meta.jsonl")
    with open(METADATA, "r", encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as g:
        for i, line in enumerate(f):
            if i >= 20:
                break
            g.write(line)

    ds = G.MultigenGRPODataset(
        metadata_path=tmp,
        audio_root=AUDIO_ROOT,
        ref_audio_root=REF_AUDIO_ROOT,
        tokenizer=tokenizer,
        max_group_size=max_group_size,
        max_audio_duration=8.0,
        min_audio_duration=0.5,
        max_ref_duration=8.0,
        max_text_tokens=600,
        min_group_count=2,
    )
    for i in range(len(ds)):
        x = ds[i]
        if x is not None:
            return x
    raise RuntimeError("No valid groups found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"\n>>> Device: {device}  Dtype: {args.dtype}")

    # ---------- Load everything ----------
    from indextts.utils.front import TextNormalizer, TextTokenizer
    normalizer = TextNormalizer()
    normalizer.load()
    tokenizer = TextTokenizer(str(TOKENIZER), normalizer)

    print(">>> Loading one real group from metadata ...")
    group = get_one_real_group(tokenizer, max_group_size=3)
    print(f"  group_key={group.group_key}")
    print(f"  text_ids.shape={group.text_ids.shape}")
    print(f"  ref_wav_16k.shape={group.ref_wav_16k.shape}")
    print(f"  num candidates={len(group.candidates)}")
    for i, c in enumerate(group.candidates):
        print(f"    cand[{i}] reward={c.reward}  wav.shape={c.wav_16k.shape}")
    rewards_set = {c.reward for c in group.candidates}
    assert len(rewards_set) >= 2, "Need both chosen and rejected"

    print(">>> Building policy GPT ...")
    policy = G.build_unified_voice(CONFIG, tokenizer, CKPT)
    G.configure_policy_trainable(policy, train_scope="body_and_head", gpt_train_mode="full")
    policy.to(device)
    print(">>> Building reference GPT ...")
    ref = G.build_unified_voice(CONFIG, tokenizer, CKPT)
    for q in ref.parameters():
        q.requires_grad = False
    ref.eval().to(device)

    print(">>> Building FeaturePreprocessor ...")
    cfg = OmegaConf.load(CONFIG)
    fp = G.FeaturePreprocessor(MODEL_DIR, cfg, device=device, dtype=torch.float32)

    wrapper = G.GRPOLossWrapper(
        policy_model=policy, ref_model=ref,
        clip_eps=0.2, kl_coeff=0.04, kl_estimator="k3",
        entropy_coeff=0.0, use_duration_control=False, duration_dropout=0.0,
    )
    wrapper.train()  # this internally pins ref + frozen submodules to eval

    # ---------- prepare_grpo_batch ----------
    with torch.no_grad():
        batch = G.prepare_grpo_batch(
            groups=[group],
            feature_extractor=fp,
            policy_model=policy,
            device=device,
            max_samples_per_batch=16,
        )
    assert batch is not None, "prepare_grpo_batch returned None"
    print("\n>>> batch tensors ----")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}  dtype={v.dtype}  device={v.device}")
        else:
            print(f"  {k}: {v}")

    # ---------- Advantages ----------
    adv, sample_valid = G.compute_advantages(batch["rewards"], batch["group_index"],
                                             batch["num_groups"], norm_strategy="global_batch")
    print(f">>> advantages: {adv.tolist()}")
    print(f">>> sample_valid: {sample_valid.tolist()}")

    # ---------- Step 1: should produce KL ~ 0 ----------
    print("\n>>> Step 1: policy == ref, expect KL=0, ratio=1 ...")
    t0 = time.time()
    optimizer = torch.optim.AdamW(
        filter(lambda q: q.requires_grad, policy.parameters()), lr=1e-4
    )
    loss, metrics = wrapper(batch, adv.to(device), sample_valid=sample_valid.to(device))
    print(f"  forward done in {time.time() - t0:.2f}s")
    print(f"  loss={loss.item():.6f}  metrics={ {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()} }")
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    assert abs(metrics["kl"].item()) < 1e-3, f"KL should be ~0 at init (dropout fix), got {metrics['kl'].item()}"
    assert abs(metrics["ratio_mean"].item() - 1.0) < 1e-3, f"ratio should be ~1 at init, got {metrics['ratio_mean'].item()}"

    # ---------- Backward ----------
    loss.backward()

    # Frozen modules must have NO gradients.
    frozen_with_grad = []
    for name in G._ALWAYS_FROZEN_MODULES:
        sub = getattr(policy, name, None)
        if sub is None:
            continue
        for pname, q in sub.named_parameters():
            if q.grad is not None and q.grad.abs().sum().item() > 0:
                frozen_with_grad.append(f"{name}.{pname}")
    assert not frozen_with_grad, f"Frozen modules received gradients: {frozen_with_grad[:5]}"
    print(f"  frozen modules check: OK ({len(G._ALWAYS_FROZEN_MODULES)} modules, none received grads)")

    # Trainable params must have grads.
    n_trainable_with_grad = 0
    n_trainable = 0
    grad_l2 = 0.0
    for q in policy.parameters():
        if q.requires_grad:
            n_trainable += 1
            if q.grad is not None and q.grad.abs().sum().item() > 0:
                n_trainable_with_grad += 1
                grad_l2 += q.grad.float().pow(2).sum().item()
    grad_l2 = grad_l2 ** 0.5
    print(f"  trainable params with grads: {n_trainable_with_grad}/{n_trainable}  total_grad_norm={grad_l2:.4f}")
    assert n_trainable_with_grad >= int(0.9 * n_trainable), "most trainable params should have grads"

    # ---------- Step 2: after optimiser update, KL must be != 0 ----------
    print("\n>>> Step 2: take one optimiser step at LR=1e-3, expect KL > 0 ...")
    # Big LR so KL clearly moves
    for q in optimizer.param_groups:
        q["lr"] = 1e-3
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        batch2 = G.prepare_grpo_batch(
            groups=[group],
            feature_extractor=fp,
            policy_model=policy,
            device=device,
            max_samples_per_batch=16,
        )
    adv2, valid2 = G.compute_advantages(batch2["rewards"], batch2["group_index"],
                                        batch2["num_groups"], norm_strategy="global_batch")
    loss2, metrics2 = wrapper(batch2, adv2.to(device), sample_valid=valid2.to(device))
    print(f"  loss={loss2.item():.6f}  KL={metrics2['kl'].item():.6f}  ratio={metrics2['ratio_mean'].item():.4f}")
    assert torch.isfinite(loss2), f"step-2 loss not finite: {loss2}"
    assert metrics2["kl"].item() > 1e-5, f"KL should have grown after an update, got {metrics2['kl'].item()}"

    print("\n=== END-TO-END OK ===")


if __name__ == "__main__":
    main()
