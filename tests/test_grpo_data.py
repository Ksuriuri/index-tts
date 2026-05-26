"""Data-pipeline tests for train_gpt_v2_grpo_multigpu.

Covers:
  - Dataset can load the first N entries from the real metadata jsonl, audios
    are reachable, candidates contain both labels.
  - GPUFeatureExtractor numerically matches the HuggingFace
    SeamlessM4TFeatureExtractor on representative audio.

Run on a single GPU.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trainers"))

os.environ.setdefault("WANDB_MODE", "disabled")

import train_gpt_v2_grpo_multigpu as G  # noqa: E402
from transformers import SeamlessM4TFeatureExtractor  # noqa: E402


MODEL_DIR = Path("checkpoints/IndexTTS-2-vLLM")
METADATA = Path("/mnt/data_3t_1/datasets/raw_data/noiz-v2/multigen/metadata_v2.jsonl")
AUDIO_ROOT = Path("/mnt/data_3t_1/datasets/raw_data/noiz-v2/multigen")
REF_AUDIO_ROOT = Path("/mnt/data_3t_1/datasets/raw_data/noiz-v2/ref_audios")
TOKENIZER = MODEL_DIR / "jp_es_bpe.model"


# ------------------------------------------------------------------ #
def _load_tokenizer():
    from indextts.utils.front import TextNormalizer, TextTokenizer
    n = TextNormalizer()
    n.load()
    return TextTokenizer(str(TOKENIZER), n)


# ------------------------------------------------------------------ #
def test_dataset_first_n_groups(n=20):
    """Read first N metadata lines and try to materialise the GroupItem for each.

    Reports: how many succeed, how many fail, common failure reasons.
    """
    import json

    print(f"\n--- test_dataset_first_n_groups (N={n}) ---")
    tokenizer = _load_tokenizer()

    # Bypass the file scan inside Dataset.__init__ by writing a small slice.
    tmp_meta = Path("/tmp/grpo_test_meta.jsonl")
    with open(METADATA, "r", encoding="utf-8") as f, open(tmp_meta, "w", encoding="utf-8") as g:
        for i, line in enumerate(f):
            if i >= n:
                break
            g.write(line)

    ds = G.MultigenGRPODataset(
        metadata_path=tmp_meta,
        audio_root=AUDIO_ROOT,
        ref_audio_root=REF_AUDIO_ROOT,
        tokenizer=tokenizer,
        max_group_size=8,
        max_audio_duration=20.0,
        min_audio_duration=0.5,
        max_ref_duration=15.0,
        max_text_tokens=600,
        ref_audio_suffix=".flac",
        min_group_count=2,
    )
    print(f"  Loaded entries={len(ds)} (from N={n} jsonl lines)")
    assert len(ds) > 0, "no valid entries parsed -- broken metadata?"

    n_ok, n_none = 0, 0
    sample_keys, sample_text_lens, sample_code_dur_s = [], [], []
    for i in range(len(ds)):
        item = ds[i]
        if item is None:
            n_none += 1
            continue
        n_ok += 1
        rewards = [c.reward for c in item.candidates]
        assert len(set(rewards)) >= 2, ("group missing one label side", i, rewards)
        sample_keys.append(item.group_key)
        sample_text_lens.append(item.text_ids.numel())
        for c in item.candidates:
            sample_code_dur_s.append(c.wav_16k.numel() / 16000.0)

    print(f"  OK={n_ok}  None(skipped)={n_none}")
    if sample_text_lens:
        print(f"  text_len min/median/max = "
              f"{min(sample_text_lens)} / {int(np.median(sample_text_lens))} / {max(sample_text_lens)}")
    if sample_code_dur_s:
        print(f"  cand audio durations (s) min/median/max = "
              f"{min(sample_code_dur_s):.2f} / {np.median(sample_code_dur_s):.2f} / {max(sample_code_dur_s):.2f}")
    print(f"  first 3 group_keys: {sample_keys[:3]}")
    assert n_ok > 0, "all groups in the first N entries failed to materialise"
    return ds


# ------------------------------------------------------------------ #
def test_gpu_fe_matches_hf(device="cuda:0"):
    """Compare GPUFeatureExtractor against the HF SeamlessM4T extractor."""
    print("\n--- test_gpu_fe_matches_hf ---")
    hf = SeamlessM4TFeatureExtractor.from_pretrained(str(MODEL_DIR / "w2v-bert-2.0"))
    gpu_fe = G.GPUFeatureExtractor(
        mel_filters_np=hf.mel_filters,
        window_np=hf.window,
    ).to(device)
    gpu_fe.eval()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    audios_np = [rng.normal(0, 0.05, size=L).astype(np.float32)
                 for L in (16000 * 2, int(16000 * 3.7), int(16000 * 1.4))]

    hf_out = hf(audios_np, sampling_rate=16000, return_tensors="pt")
    audios_t = [torch.from_numpy(a) for a in audios_np]
    gpu_out = gpu_fe([a.to(device) for a in audios_t])

    hf_feat = hf_out["input_features"].to(device)
    hf_mask = hf_out["attention_mask"].to(device)
    gpu_feat = gpu_out["input_features"]
    gpu_mask = gpu_out["attention_mask"]

    # Equal frame counts per item (mask)
    assert hf_mask.shape == gpu_mask.shape, (hf_mask.shape, gpu_mask.shape)
    assert torch.equal(hf_mask, gpu_mask), "attention masks differ"

    # Feature shapes and values
    assert hf_feat.shape == gpu_feat.shape, (hf_feat.shape, gpu_feat.shape)
    diff = (hf_feat - gpu_feat).abs()
    # Only compare valid (unmasked) positions
    mask_3d = hf_mask.unsqueeze(-1).bool()
    masked_diff = diff[mask_3d.expand_as(diff)]
    max_err = masked_diff.max().item()
    mean_err = masked_diff.mean().item()
    print(f"  feature shape={tuple(hf_feat.shape)}  max_abs_err={max_err:.6f}  mean_abs_err={mean_err:.6e}")
    # Kaldi fbank + per-frame norm should be deterministic; allow only fp32 noise.
    assert max_err < 1e-3, f"GPUFeatureExtractor diverges from HF (max_err={max_err})"


# ------------------------------------------------------------------ #
def test_codec_roundtrip_on_real_audio(device="cuda:0"):
    """Build the feature preprocessor and quantize a couple of real candidate
    audios.  Mostly checks that the codes are within [0, codebook_size)."""
    from omegaconf import OmegaConf
    print("\n--- test_codec_roundtrip_on_real_audio ---")
    cfg = OmegaConf.load(MODEL_DIR / "config.yaml")
    fp = G.FeaturePreprocessor(MODEL_DIR, cfg, device=torch.device(device), dtype=torch.float32)

    # Grab the first cand from the dataset
    tokenizer = _load_tokenizer()
    tmp_meta = Path("/tmp/grpo_test_meta.jsonl")
    if not tmp_meta.exists():
        test_dataset_first_n_groups(n=5)

    ds = G.MultigenGRPODataset(
        metadata_path=tmp_meta,
        audio_root=AUDIO_ROOT,
        ref_audio_root=REF_AUDIO_ROOT,
        tokenizer=tokenizer,
        max_group_size=4,
        max_audio_duration=20.0,
        min_audio_duration=0.5,
        max_ref_duration=15.0,
        max_text_tokens=600,
        min_group_count=2,
    )
    chosen_item = None
    for i in range(len(ds)):
        x = ds[i]
        if x is not None:
            chosen_item = x
            break
    assert chosen_item is not None, "could not get a valid group from the dataset slice"
    cand_wavs = [c.wav_16k for c in chosen_item.candidates[:2]]
    durations = [w.numel() / 16000.0 for w in cand_wavs]
    print(f"  using {len(cand_wavs)} candidates, durations = {durations}")

    spk_emb, cond_len = fp.extract_spk_cond_emb(cand_wavs)
    print(f"  spk_cond_emb shape={tuple(spk_emb.shape)}  cond_len={cond_len.tolist()}")
    assert spk_emb.dim() == 3
    assert spk_emb.shape[0] == len(cand_wavs)
    assert spk_emb.shape[-1] == 1024

    codes, code_lens = fp.extract_codes(cand_wavs)
    print(f"  codes shape={tuple(codes.shape)}  code_lens={code_lens.tolist()}")
    assert codes.dim() == 2
    assert codes.shape[0] == len(cand_wavs)
    assert (codes >= 0).all() and (codes < cfg.semantic_codec.codebook_size + 2).all()
    # cond_len should equal code_lens (they come from same attention_mask)
    assert torch.equal(cond_len, code_lens), (cond_len, code_lens)

    # Inference uses ~16kHz->frames_per_second of ~ feature_extractor stride.
    # Each frame is ~20ms (stride 2 at 100fps -> 50fps).  Roughly check codes
    # length ~= duration_s * 50.
    for i, dur in enumerate(durations):
        approx_frames = dur * 50
        actual = code_lens[i].item()
        rel_err = abs(actual - approx_frames) / max(approx_frames, 1)
        assert rel_err < 0.1, ("frame count vs duration mismatch", i, approx_frames, actual)
    print(f"  code lengths match expected ~50fps within 10%")


# ------------------------------------------------------------------ #
def main():
    t0 = time.time()
    failed = []
    for fn in (test_dataset_first_n_groups,
               test_gpu_fe_matches_hf,
               test_codec_roundtrip_on_real_audio):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {fn.__name__}: {e!r}")
            failed.append(fn.__name__)
    print(f"\nElapsed: {time.time() - t0:.1f}s")
    if failed:
        print(f"=== FAILED: {failed} ===")
        sys.exit(1)
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
