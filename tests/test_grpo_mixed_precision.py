"""Verify the post-fix behaviour under:
  - Accelerator(mixed_precision='fp16')
  - reference model held in fp16
  - feature preprocessor running in fp32 inside autocast
After all four fixes (no dropout, local-fp32 FE, no truncation, validity mask).
"""
import os
import sys
from pathlib import Path

import torch

ROOT = "/mnt/data_sdd/hhy/index-tts"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trainers"))
os.environ.setdefault("WANDB_MODE", "disabled")
os.chdir(ROOT)

import train_gpt_v2_grpo_multigpu as G  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from accelerate import Accelerator  # noqa: E402

MODEL_DIR = Path("checkpoints/IndexTTS-2-vLLM")
CONFIG = MODEL_DIR / "config.yaml"
TOKENIZER = MODEL_DIR / "jp_es_bpe.model"
CKPT = MODEL_DIR / "gpt.pth"


def build_setup(mixed_precision: str, ref_dtype: str):
    accel = Accelerator(mixed_precision=mixed_precision)

    from indextts.utils.front import TextNormalizer, TextTokenizer
    norm = TextNormalizer(); norm.load()
    tokenizer = TextTokenizer(str(TOKENIZER), norm)

    policy = G.build_unified_voice(CONFIG, tokenizer, CKPT)
    G.configure_policy_trainable(policy, train_scope="body_and_head", gpt_train_mode="full")

    ref = G.build_unified_voice(CONFIG, tokenizer, CKPT)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()
    if ref_dtype != "fp32":
        ref.to(dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[ref_dtype])

    wrapper = G.GRPOLossWrapper(
        policy_model=policy, ref_model=ref,
        clip_eps=0.2, kl_coeff=0.04, kl_estimator="k3",
        entropy_coeff=0.0, use_duration_control=False, duration_dropout=0.0,
    )
    opt = torch.optim.AdamW(filter(lambda q: q.requires_grad, policy.parameters()), lr=2e-6)
    wrapper, opt = accel.prepare(wrapper, opt)

    cfg = OmegaConf.load(CONFIG)
    fp = G.FeaturePreprocessor(MODEL_DIR, cfg, device=accel.device, dtype=torch.float32)

    from tests.test_grpo_e2e import get_one_real_group
    group = get_one_real_group(tokenizer, max_group_size=3)

    return accel, wrapper, opt, fp, group, tokenizer


def run_case(mixed_precision: str, ref_dtype: str):
    print(f"\n=== Case: accel.mixed_precision={mixed_precision}, ref_dtype={ref_dtype} ===")
    accel, wrapper, opt, fp, group, _ = build_setup(mixed_precision, ref_dtype)
    unwrap = accel.unwrap_model(wrapper)

    # This mimics the real training loop
    with torch.no_grad(), accel.autocast():
        batch = G.prepare_grpo_batch(
            [group], fp, unwrap.policy_model, accel.device, max_samples_per_batch=16
        )
    adv, valid = G.compute_advantages(
        batch["rewards"], batch["group_index"], batch["num_groups"],
        norm_strategy="global_batch",
    )
    adv = adv.to(accel.device)
    valid = valid.to(accel.device)

    # Step 1 -- should still be KL ~ 0
    wrapper.train()  # default in training loop
    with accel.accumulate(wrapper):
        loss, metrics = wrapper(batch, adv, sample_valid=valid)
        accel.backward(loss)
        if accel.sync_gradients:
            opt.step()
            opt.zero_grad()

    finite_loss = bool(torch.isfinite(loss).item())
    finite_kl = bool(torch.isfinite(metrics["kl"]).item())
    print(f"  loss={loss.item()}  KL={metrics['kl'].item():.8f}  "
          f"ratio={metrics['ratio_mean'].item():.6f}  "
          f"finite_loss={finite_loss}  finite_kl={finite_kl}")

    # gradient sanity check
    grad_l2 = 0.0
    for p in unwrap.policy_model.parameters():
        if p.requires_grad and p.grad is not None:
            grad_l2 += p.grad.float().pow(2).sum().item()
    grad_l2 = grad_l2 ** 0.5
    print(f"  policy grad L2 norm = {grad_l2:.6f}")

    assert finite_loss, f"loss not finite under mixed_precision={mixed_precision}"
    assert finite_kl, f"KL not finite under mixed_precision={mixed_precision}"
    # At step 0, with no dropout + ref==policy weights, |KL| should be tiny.
    # The tolerance reflects each dtype's mantissa precision (fp16: 10b,
    # bf16: 7b, fp32: 23b).  Pre-fix this value was ~0.078 regardless of
    # dtype because of dropout noise.
    kl_tol = {"fp16": 1e-3, "bf16": 1e-2, "fp32": 1e-5}[ref_dtype]
    assert abs(metrics["kl"].item()) < kl_tol, (
        f"KL too large at init: {metrics['kl'].item()} (tol={kl_tol})"
    )
    ratio_tol = {"fp16": 5e-3, "bf16": 1e-2, "fp32": 1e-4}[ref_dtype]
    assert abs(metrics["ratio_mean"].item() - 1.0) < ratio_tol, (
        f"ratio off from 1 at init: {metrics['ratio_mean'].item()}"
    )
    print("  [OK]")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mp", default=None, help="If set, run this single case (mp,ref_dtype)")
    p.add_argument("--ref-dtype", default=None)
    args = p.parse_args()

    if args.mp is not None:
        run_case(args.mp, args.ref_dtype)
        print("\n=== CASE PASSED ===")
        return

    # Top-level runner: fork a fresh interpreter per case so each Accelerator
    # init is the first one in its process.
    import subprocess
    cases = [
        # The shell script default
        ("fp16", "fp16"),
        # Safe fallback
        ("bf16", "bf16"),
        # Pure baseline
        ("no", "fp32"),
    ]
    failed = []
    for mp, rd in cases:
        cmd = [
            sys.executable, __file__,
            "--mp", mp, "--ref-dtype", rd,
        ]
        print(f"\n>>> spawning subprocess: mp={mp} ref={rd}")
        r = subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
        if r.returncode != 0:
            failed.append((mp, rd))
    if failed:
        print(f"\n=== FAILED CASES: {failed} ===")
        sys.exit(1)
    print("\n=== ALL MIXED-PRECISION CASES PASSED ===")


if __name__ == "__main__":
    main()
