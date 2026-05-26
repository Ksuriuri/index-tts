"""
GRPO finetune for IndexTTS2 GPT.

References:
- https://arxiv.org/abs/2509.21718
- https://arxiv.org/abs/2511.21270

Pipeline (on-the-fly, no offline preprocessing):
  metadata_v2.jsonl  ->  groups of (text, voice_id, [chosen audios], [rejected audios])
  per group:
    - load ref_audios/{voice_id}.flac  -> speaker prompt for `gpt.get_conditioning`
    - load each candidate flac          -> SeamlessM4T fbank -> semantic_model
                                          -> spk_cond_emb    -> semantic_codec.quantize
                                                              -> mel codes (the y_i token sequence)
  per sample:
    reward = 1.0 (chosen) / 0.0 (rejected)
  per group / batch:
    advantage = (R - mean(R)) / (std(R) + eps)  [optionally globally normalised across batch]
  GRPO loss (teacher-forced per-token):
    ratio_t        = exp(logp_pi - logp_ref)
    policy_loss_t  = -min(ratio_t * A, clip(ratio_t, 1-eps, 1+eps) * A)
    kl_t (k3)      = exp(logp_ref - logp_pi) - (logp_ref - logp_pi) - 1
    loss           = mean_t[policy_loss_t + beta * kl_t] - gamma * entropy

Emotion vector (`emo_vec`) mirrors inference: base emotion comes from the
reference audio, and optional text emotion tags are mixed through feat1/feat2.
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

import safetensors
from transformers import SeamlessM4TFeatureExtractor

# Repo path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from accelerate import Accelerator
from accelerate.utils import set_seed

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.utils.maskgct_utils import build_semantic_codec, build_semantic_model
from indextts.utils.front import TextNormalizer, TextTokenizer


TARGET_SR = 16000
EMOTION_ORDER = ["joy", "anger", "sadness", "fear", "disgust", "depression", "surprise", "calm"]
EMOTION_INDEX = {name: idx for idx, name in enumerate(EMOTION_ORDER)}
SUPPORTED_EMOTIONS = set(EMOTION_ORDER)
EMOTION_ALIASES = {"happy": "joy"}
EMO_SPEC = r"[A-Za-z]+:[0-9]+(?:;[A-Za-z]+:[0-9]+)*"
EMOTION_TAG_RE = re.compile(
    rf"\[(?:[^\]]*?#(?P<emo1>{EMO_SPEC})|(?P<emo2>{EMO_SPEC}))\]"
    rf"(?:\s*:\s*)?"
    rf"(?P<text>[\s\S]*?)"
    rf"(?=\[(?:[^\]]*?#(?:{EMO_SPEC})|(?:{EMO_SPEC}))\]|$)"
)
EMOTION_TAG_MARK_RE = re.compile(
    rf"\[(?:[^\]]*?#(?:{EMO_SPEC})|(?:{EMO_SPEC}))\](?:\s*:\s*)?"
)
NON_EMO_TAG_WITH_COLON_RE = re.compile(r"\[.*?\]:")


# =============================================================================
# Args
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO finetune for IndexTTS2 GPT.")

    # Data
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Path to metadata_v2.jsonl (multigen format).")
    parser.add_argument("--audio-root", type=Path, required=True,
                        help="Root dir holding the `chosen/` and `rejected/` audio dirs referenced by metadata.")
    parser.add_argument("--ref-audio-root", type=Path, required=True,
                        help="Dir holding `{voice_id}.flac` speaker reference audios.")
    parser.add_argument("--ref-audio-suffix", type=str, default=".flac",
                        help="Suffix of reference audio files (e.g. .flac, .wav).")
    parser.add_argument("--max-group-size", type=int, default=8,
                        help="Per-group cap on (chosen + rejected) candidates.")
    parser.add_argument("--min-group-count", type=int, default=3,
                        help="Drop groups whose total candidate count is below this value. "
                             "Uses metadata `total_count` when present, otherwise "
                             "len(chosen) + len(rejected).")
    parser.add_argument("--max-audio-duration", type=float, default=36.0,
                        help="Drop candidate (chosen/rejected) audios longer than this many seconds. "
                             "Candidates are NEVER truncated -- truncating would break the "
                             "text<->audio alignment used by teacher-forcing.")
    parser.add_argument("--min-audio-duration", type=float, default=0.5,
                        help="Drop audios shorter than this many seconds.")
    parser.add_argument("--max-ref-duration", type=float, default=36.0,
                        help="Drop the entire group when the speaker-reference audio is "
                             "longer than this many seconds.  Reference audio is NEVER "
                             "truncated either: a 36 s ref summarised through the (frozen) "
                             "conditioning encoder differs from a 36-s-truncated-to-15-s ref, "
                             "so training and inference would silently diverge.")

    # Model
    parser.add_argument("--tokenizer", type=Path,
                        default=Path("checkpoints/IndexTTS-2-vLLM/jp_es_bpe.model"),
                        help="SentencePiece BPE model path.")
    parser.add_argument("--config", type=Path,
                        default=Path("checkpoints/IndexTTS-2-vLLM/config.yaml"),
                        help="Model config YAML.")
    parser.add_argument("--base-checkpoint", type=Path,
                        default=Path("checkpoints/IndexTTS-2-vLLM/gpt.pth"),
                        help="Base GPT checkpoint (used for both policy and reference).")
    parser.add_argument("--ref-checkpoint", type=Path, default=None,
                        help="Optional separate checkpoint for the reference model. "
                             "Defaults to --base-checkpoint.")
    parser.add_argument("--ref-dtype", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp32"],
                        help="Dtype to hold the reference GPT in (default: fp16). "
                             "fp16 halves the per-GPU memory cost and is the closest "
                             "match to inference; switch to bf16 (or fp32) if you "
                             "observe NaNs / Infs in logp_ref during training.")
    parser.add_argument("--model-dir", type=Path,
                        default=Path("checkpoints/IndexTTS-2-vLLM"),
                        help="Dir containing w2v-bert-2.0/, semantic_codec/, wav2vec2bert_stats.pt.")

    # Training schedule
    parser.add_argument("--output-dir", type=Path, default=Path("trained_ckpts/grpo"))
    parser.add_argument("--groups-per-device", type=int, default=1,
                        help="Number of groups per device per micro-batch.")
    parser.add_argument("--max-samples-per-batch", type=int, default=32,
                        help="Safety cap on total candidates per micro-batch. "
                             "Groups will be sub-sampled if needed.")
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-6,
                        help="Recommended: SFT LR * 0.1, since post-training.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-ref-cache",
        action="store_true",
        help=(
            "Disable the in-memory cache of per-reference-audio features "
            "(conditioning / emo_vec_base / style).  Caching is enabled by "
            "default and skips the (frozen) w2v-bert + Conformer / Perceiver "
            "passes for any reference audio we've already seen this run.  "
            "Memory cost: ~166 KB / unique ref audio on CPU (~376 MB total "
            "for the noiz-v2 multigen metadata)."
        ),
    )
    parser.add_argument(
        "--ref-cache-max-entries",
        type=int,
        default=0,
        help=(
            "Upper bound on the number of cached ref-audio entries.  0 = "
            "unbounded (default), since the dataset only has a few thousand "
            "unique voice ids.  Set this if you ever train on a corpus with "
            "millions of unique speakers and want to cap CPU memory."
        ),
    )

    # GRPO hyper-params
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-coeff", type=float, default=0.04)
    parser.add_argument("--entropy-coeff", type=float, default=0.0,
                        help="Optional entropy regulariser to prevent token-distribution collapse "
                             "(see arXiv:2511.21270).")
    parser.add_argument("--adv-norm", type=str, default="global_batch",
                        choices=["intra_group", "global_batch"],
                        help="Advantage normalisation strategy.")
    parser.add_argument("--reward-chosen", type=float, default=1.0)
    parser.add_argument("--reward-rejected", type=float, default=0.0)
    parser.add_argument("--kl-estimator", type=str, default="k3", choices=["k1", "k3"],
                        help="KL approximation: k1=logp_ref - logp_pi, k3=exp(d) - d - 1, d=logp_ref-logp_pi.")

    # Trainable-parameter scope
    parser.add_argument(
        "--train-scope",
        type=str,
        default="body_and_head",
        choices=["body_and_head", "lm_core", "body_only"],
        help=(
            "Which UnifiedVoice parameters to update.  "
            "'body_and_head' (default, Hybrid): freeze all embeddings + conditioning/emo "
            "encoders, train GPT body + final_norm + mel_head.  "
            "'lm_core': also train text/mel embeddings and position embeddings (SFT-style).  "
            "'body_only': train GPT body + final_norm only (no mel_head)."
        ),
    )
    parser.add_argument(
        "--gpt-train-mode",
        type=str,
        default="full",
        choices=["full", "attention_only"],
        help=(
            "How much of the GPT-2 stack to update within --train-scope.  "
            "'full' (default): all transformer blocks (attn + MLP + LayerNorm).  "
            "'attention_only': only self-attention projections inside each block; "
            "FFN / block LayerNorm stay frozen."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "Enable HF GPT-2 gradient checkpointing on the *policy* model only "
            "(reference runs under no_grad so it gets nothing).  Saves ~30-50% "
            "peak VRAM on the transformer activations at ~20-30% slower forward.  "
            "Uses ``use_reentrant=False`` so it stays compatible with our frozen "
            "embedding / pre-computed conditioning inputs."
        ),
    )

    # Duration control (mirrors SFT script)
    parser.add_argument("--use-duration-control", action="store_true")
    parser.add_argument("--duration-dropout", type=float, default=0.3)

    # Logging / checkpoint
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--major-save-every", type=int, default=10000)
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument("--resume", type=str, default="")

    # WandB
    parser.add_argument("--wandb-project", type=str, default="indextts-grpo")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)

    return parser.parse_args()


# =============================================================================
# GPU Feature Extractor (Kaldi-style fbank, replicates SeamlessM4TFeatureExtractor)
# =============================================================================

class GPUFeatureExtractor(nn.Module):
    """Batched GPU drop-in replacement for SeamlessM4TFeatureExtractor.

    Always computes in fp32: the very first op is ``audio * 2**15`` which
    immediately overflows fp16 (max ~65504), and the downstream FFT / log
    operations are precision-sensitive.  The class explicitly disables AMP
    autocast in ``forward`` so it remains safe to call from inside a wider
    ``accelerator.autocast()`` block.
    """

    def __init__(self, mel_filters_np, window_np, num_mel_bins=80, stride=2):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.stride = stride
        self.frame_length = 400
        self.hop_length = 160
        self.fft_length = 512
        self.preemphasis_coeff = 0.97
        self.mel_floor = 1.192092955078125e-07
        self.scale = float(2 ** 15)
        self.register_buffer('window', torch.from_numpy(window_np).float())
        self.register_buffer('mel_filters', torch.from_numpy(mel_filters_np).float())

    @torch.no_grad()
    def forward(self, audios: List[torch.Tensor]) -> dict:
        with torch.amp.autocast(device_type="cuda", enabled=False):
            return self._forward_fp32(audios)

    def _forward_fp32(self, audios: List[torch.Tensor]) -> dict:
        device = audios[0].device
        batch_size = len(audios)
        # Ensure inputs are fp32 even if caller is inside autocast.
        audios = [a.float() if a.dtype != torch.float32 else a for a in audios]
        lengths = [a.shape[0] for a in audios]
        max_len = max(lengths)

        padded = torch.zeros(batch_size, max_len, device=device, dtype=torch.float32)
        for i, a in enumerate(audios):
            padded[i, :lengths[i]] = a
        padded = padded * self.scale

        frames = padded.unfold(1, self.frame_length, self.hop_length)
        frame_counts = [(l - self.frame_length) // self.hop_length + 1 for l in lengths]
        max_valid = max(frame_counts)
        frames = frames[:, :max_valid, :]

        frames = frames - frames.mean(dim=2, keepdim=True)

        preemph = torch.empty_like(frames)
        preemph[:, :, 0] = frames[:, :, 0] * (1.0 - self.preemphasis_coeff)
        preemph[:, :, 1:] = frames[:, :, 1:] - self.preemphasis_coeff * frames[:, :, :-1]

        preemph = preemph * self.window
        padded_frames = F.pad(preemph, (0, self.fft_length - self.frame_length))
        power = torch.fft.rfft(padded_frames).abs().square()

        mel = torch.matmul(power, self.mel_filters)
        mel = torch.clamp(mel, min=self.mel_floor)
        mel = torch.log(mel)

        arange = torch.arange(max_valid, device=device).unsqueeze(0)
        fc_tensor = torch.tensor(frame_counts, device=device).unsqueeze(1)
        attention_mask = (arange < fc_tensor).long()

        for i in range(batch_size):
            fc = frame_counts[i]
            seg = mel[i, :fc]
            mean = seg.mean(dim=0)
            var = seg.var(dim=0, unbiased=True)
            mel[i, :fc] = (seg - mean) / torch.sqrt(var + 1e-7)

        remainder = max_valid % self.stride
        if remainder != 0:
            pad_n = self.stride - remainder
            mel = F.pad(mel, (0, 0, 0, pad_n))
            attention_mask = F.pad(attention_mask, (0, pad_n))
            max_valid += pad_n

        mel = mel.reshape(batch_size, max_valid // self.stride,
                          self.num_mel_bins * self.stride)
        indices = torch.arange(max_valid, device=device)
        attention_mask = attention_mask[:, indices % self.stride == 1]

        return {"input_features": mel, "attention_mask": attention_mask}


# =============================================================================
# Dataset
# =============================================================================

@dataclass
class GroupSample:
    """One audio candidate within a group."""
    wav_16k: torch.Tensor  # float32, mono, 16kHz
    reward: float
    path: str


@dataclass
class GroupItem:
    """One GRPO group: same text + ref audio, multiple candidate generations."""
    text_ids: torch.Tensor       # long [text_len]
    ref_wav_16k: torch.Tensor    # float32, mono, 16kHz
    candidates: List[GroupSample]
    group_key: str
    emo_control_vector: Optional[torch.Tensor] = None  # float32 [8], from text tags
    ref_audio_stem: Optional[str] = None  # stable cache key for the speaker ref audio


def _read_audio_to_16k(
    path: str,
    max_seconds: Optional[float] = None,
    truncate: bool = False,
) -> Optional[torch.Tensor]:
    """Load audio, mono-mix, resample to 16k. Returns None on failure.

    When ``max_seconds`` is set:
      - ``truncate=False`` (default, used for chosen/rejected candidates):
        return ``None`` if the audio is longer than ``max_seconds``.  We must
        never silently truncate a candidate because that would break the
        text->audio alignment used by teacher forcing and corrupt the GRPO
        reward signal.
      - ``truncate=True`` (used for the speaker reference): clip to
        ``max_seconds``.  This matches inference's ``_load_and_cut_audio``
        which caps the reference voice at 15 s for the conditioning encoder.
    """
    try:
        wav, sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    if wav.size == 0:
        return None
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(np.ascontiguousarray(wav)).float()
    if sr != TARGET_SR:
        wav_t = torchaudio.functional.resample(wav_t, sr, TARGET_SR)
    if max_seconds is not None:
        max_samples = int(max_seconds * TARGET_SR)
        if wav_t.numel() > max_samples:
            if truncate:
                wav_t = wav_t[:max_samples]
            else:
                return None
    return wav_t.contiguous()


def _normalise_emotion_name(name: str) -> Optional[str]:
    key = name.strip().lower()
    key = EMOTION_ALIASES.get(key, key)
    return key if key in SUPPORTED_EMOTIONS else None


def _emotion_spec_to_vector(emo_part: str) -> Optional[List[float]]:
    vector = [0.0] * len(EMOTION_ORDER)
    found = False
    for item in emo_part.split(";"):
        if ":" not in item:
            continue
        raw_name, raw_value = item.split(":", 1)
        name = _normalise_emotion_name(raw_name)
        if name is None:
            continue
        try:
            value = float(raw_value) / 10.0
        except ValueError:
            continue
        vector[EMOTION_INDEX[name]] = max(0.0, min(1.4, value))
        found = True
    return vector if found else None


def parse_text_emotion_tags(raw_text: str) -> Tuple[str, Optional[torch.Tensor]]:
    """Strip request-style emotion tags and convert them to the inference vector.

    Examples:
      [🤯#Sadness:2;Surprise:5] What a pity.
      -> ("What a pity.", [0, 0, 0.2, 0, 0, 0, 0.5, 0])

    If multiple labelled spans exist in one training text, their vectors are
    averaged by labelled span length because the GRPO sample has one target audio.
    """
    raw_text = raw_text or ""
    weighted = [0.0] * len(EMOTION_ORDER)
    total_weight = 0.0

    for match in EMOTION_TAG_RE.finditer(raw_text):
        emo_part = match.group("emo1") or match.group("emo2") or ""
        vector = _emotion_spec_to_vector(emo_part)
        if vector is None:
            continue
        text_part = (match.group("text") or "").strip()
        weight = max(1, len(text_part))
        for idx, value in enumerate(vector):
            weighted[idx] += value * weight
        total_weight += weight

    cleaned = EMOTION_TAG_MARK_RE.sub("", raw_text)
    cleaned = NON_EMO_TAG_WITH_COLON_RE.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    if total_weight <= 0:
        return cleaned, None
    averaged = [value / total_weight for value in weighted]
    return cleaned, torch.tensor(averaged, dtype=torch.float32)


def _resolve_ref_audio_stem(item: Dict[str, Any]) -> Optional[str]:
    """Resolve the speaker-reference id/path stem from metadata.

    Supported schemas:
      - ``voice_id`` (current multigen metadata_v2.jsonl)
      - ``ref_audio_id`` / ``ref_audio_file`` (legacy pairs metadata)
    """
    if item.get("ref_audio_id"):
        return str(item["ref_audio_id"])
    if item.get("voice_id"):
        return str(item["voice_id"])
    ref_file = item.get("ref_audio_file")
    if ref_file:
        return Path(str(ref_file)).stem
    return None


def _extract_side_files(
    records: Any,
    side: str,
    gen_product_ids: Optional[List[str]] = None,
) -> List[str]:
    """Extract relative audio paths for one label side (chosen / rejected).

    Accepts the current multigen format (list of dicts with ``file``), and
    falls back to ``gen_product_id`` / ``{side}_gen_product_ids`` when needed.
    """
    files: List[str] = []
    if isinstance(records, list):
        for rec in records:
            if isinstance(rec, str) and rec.strip():
                files.append(rec.strip())
            elif isinstance(rec, dict):
                if rec.get("file"):
                    files.append(str(rec["file"]))
                elif rec.get("gen_product_id"):
                    gid = str(rec["gen_product_id"])
                    files.append(f"{side}/{gid}.flac")

    if not files and gen_product_ids:
        files = [f"{side}/{gid}.flac" for gid in gen_product_ids if gid]

    # De-duplicate while preserving order.
    seen = set()
    unique_files: List[str] = []
    for path in files:
        if path not in seen:
            seen.add(path)
            unique_files.append(path)
    return unique_files


def _parse_metadata_group(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalise one metadata json object into the internal dataset entry."""
    ref_stem = _resolve_ref_audio_stem(item)
    if not ref_stem:
        return None

    chosen = item.get("chosen") or []
    rejected = item.get("rejected") or []
    chosen_files = _extract_side_files(
        chosen, "chosen", item.get("chosen_gen_product_ids")
    )
    rejected_files = _extract_side_files(
        rejected, "rejected", item.get("rejected_gen_product_ids")
    )
    if not chosen_files or not rejected_files:
        return None

    text = item.get("target_text") or item.get("text") or ""
    if not text:
        return None

    text, emo_control_vector = parse_text_emotion_tags(text)
    if not text:
        return None

    group_count = item.get("total_count")
    if group_count is None:
        group_count = len(chosen_files) + len(rejected_files)
    else:
        group_count = int(group_count)

    return {
        "target_text": text,
        "emo_control_vector": emo_control_vector,
        "voice_id": str(item.get("voice_id") or ref_stem),
        "ref_audio_stem": ref_stem,
        "chosen_files": chosen_files,
        "rejected_files": rejected_files,
        "group_count": group_count,
        "create_time": item.get("create_time"),
    }


class MultigenGRPODataset(Dataset):
    """Reads metadata_v2.jsonl and emits one group per __getitem__.

    Supported line formats:

    Current multigen (metadata_v2.jsonl):
      {
        "target_text": str,
        "voice_id": str,
        "create_time": int,
        "chosen": [{"file": "chosen/xxx.flac", "gen_product_id": str, ...}, ...],
        "chosen_gen_product_ids": [str, ...],
        "rejected": [{"file": "rejected/xxx.flac", "gen_product_id": str, ...}, ...],
        "rejected_gen_product_ids": [str, ...],
        "total_count": int
      }

    Legacy pairs-style (still accepted):
      {
        "target_text": str,
        "ref_audio_id": str,
        "ref_audio_file": "ref_audios/xxx.flac",
        "chosen": [...], "rejected": [...]
      }
    """

    def __init__(
        self,
        metadata_path: Path,
        audio_root: Path,
        ref_audio_root: Path,
        tokenizer: TextTokenizer,
        max_group_size: int,
        max_audio_duration: float,
        min_audio_duration: float,
        max_ref_duration: float,
        max_text_tokens: int,
        ref_audio_suffix: str = ".flac",
        reward_chosen: float = 1.0,
        reward_rejected: float = 0.0,
        min_group_count: int = 3,
    ):
        self.audio_root = Path(audio_root)
        self.ref_audio_root = Path(ref_audio_root)
        self.tokenizer = tokenizer
        self.max_group_size = max_group_size
        self.max_audio_duration = max_audio_duration
        self.min_audio_duration = min_audio_duration
        self.max_ref_duration = max_ref_duration
        self.max_text_tokens = max_text_tokens
        self.ref_audio_suffix = ref_audio_suffix
        self.reward_chosen = reward_chosen
        self.reward_rejected = reward_rejected
        self.min_group_count = min_group_count

        print(f"[Dataset] Loading metadata from {metadata_path} ...")
        self.entries: List[Dict[str, Any]] = []
        kept, dropped = 0, 0
        dropped_reasons = {
            "parse_error": 0,
            "invalid_group": 0,
            "too_small": 0,
            "empty_text": 0,
        }
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    dropped += 1
                    dropped_reasons["parse_error"] += 1
                    continue

                entry = _parse_metadata_group(item)
                if entry is None:
                    dropped += 1
                    dropped_reasons["invalid_group"] += 1
                    continue
                if entry["group_count"] < self.min_group_count:
                    dropped += 1
                    dropped_reasons["too_small"] += 1
                    continue

                self.entries.append(entry)
                kept += 1
        print(
            f"[Dataset] Loaded {kept} valid groups (dropped {dropped}, "
            f"min_group_count={self.min_group_count})."
        )
        if dropped:
            print(
                f"[Dataset] Drop reasons: parse_error={dropped_reasons['parse_error']}, "
                f"invalid_group={dropped_reasons['invalid_group']}, "
                f"too_small={dropped_reasons['too_small']}"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def _sample_candidates(self, chosen_files: List[str], rejected_files: List[str]
                           ) -> List[Tuple[str, float]]:
        """Pick up to ``max_group_size`` files, guaranteeing at least one of each label."""
        nc, nr = len(chosen_files), len(rejected_files)
        total = nc + nr
        if total <= self.max_group_size:
            samples = [(f, self.reward_chosen) for f in chosen_files] + \
                      [(f, self.reward_rejected) for f in rejected_files]
            random.shuffle(samples)
            return samples

        # stratified down-sample
        target_c = max(1, round(self.max_group_size * nc / total))
        target_c = min(target_c, nc)
        target_r = self.max_group_size - target_c
        target_r = min(max(1, target_r), nr)
        target_c = min(nc, self.max_group_size - target_r)
        c = random.sample(chosen_files, target_c)
        r = random.sample(rejected_files, target_r)
        samples = [(f, self.reward_chosen) for f in c] + \
                  [(f, self.reward_rejected) for f in r]
        random.shuffle(samples)
        return samples

    def __getitem__(self, idx: int) -> Optional[GroupItem]:
        entry = self.entries[idx]
        text = entry["target_text"]
        voice_id = entry["voice_id"]
        emo_control_vector = entry["emo_control_vector"]
        ref_stem = entry.get("ref_audio_stem", voice_id)

        # Tokenise text up-front (cheap) and length-filter.
        try:
            tokens = self.tokenizer.tokenize(text)
            text_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        except Exception:
            return None
        if len(text_ids) == 0 or len(text_ids) > self.max_text_tokens:
            return None

        # Load reference audio.  Drop the whole group if the ref is longer than
        # `max_ref_duration`; never truncate (same reasoning as for candidates:
        # the frozen conditioning encoder produces a different speaker summary
        # from truncated vs. full audio, which would silently mismatch
        # inference and corrupt the policy/ref logp ratio).
        ref_path = self.ref_audio_root / f"{ref_stem}{self.ref_audio_suffix}"
        ref_wav = _read_audio_to_16k(
            str(ref_path), max_seconds=self.max_ref_duration, truncate=False
        )
        if ref_wav is None or ref_wav.numel() < int(self.min_audio_duration * TARGET_SR):
            # Ref audio missing / unreadable / too long: skip the whole group.
            return None

        # Sample candidates.  Never truncate: candidate audios that exceed
        # ``max_audio_duration`` are dropped because text<->audio alignment
        # would be broken otherwise.
        picks = self._sample_candidates(entry["chosen_files"], entry["rejected_files"])
        candidates: List[GroupSample] = []
        for rel_path, reward in picks:
            full_path = self.audio_root / rel_path
            wav = _read_audio_to_16k(
                str(full_path), max_seconds=self.max_audio_duration, truncate=False
            )
            if wav is None:
                continue
            if wav.numel() < int(self.min_audio_duration * TARGET_SR):
                continue
            candidates.append(GroupSample(wav_16k=wav, reward=float(reward), path=str(full_path)))

        # Need at least one chosen + one rejected.
        rewards = {c.reward for c in candidates}
        if len(candidates) < 2 or len(rewards) < 2:
            return None

        return GroupItem(
            text_ids=torch.tensor(text_ids, dtype=torch.long),
            ref_wav_16k=ref_wav,
            candidates=candidates,
            group_key=f"{voice_id}::{idx}",
            emo_control_vector=emo_control_vector,
            ref_audio_stem=ref_stem,
        )


def collate_groups(batch: List[Optional[GroupItem]]) -> List[GroupItem]:
    """Drop empty / failed groups; return the rest as a python list of GroupItem."""
    return [g for g in batch if g is not None]


# =============================================================================
# Reference / policy GPT loader
# =============================================================================

def _load_gpt_checkpoint(model: UnifiedVoice, checkpoint_path: Path, verbose: bool = True):
    if verbose:
        print(f"Loading GPT checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw_state_dict = checkpoint.get("model", checkpoint)

    filtered_state_dict = {}
    for key, value in raw_state_dict.items():
        if key.startswith("inference_model."):
            continue
        if ".lora_" in key:
            continue
        new_key = key.replace(".base_layer.", ".")
        if new_key == "gpt.wte.weight":
            continue
        filtered_state_dict[new_key] = value
    state_dict = filtered_state_dict

    # Resize embeddings (text-vocab) to match the current model.
    resizable_keys = {
        "text_embedding.weight": model.text_embedding.weight,
        "text_head.weight": model.text_head.weight,
        "text_head.bias": model.text_head.bias,
    }
    for key, param in resizable_keys.items():
        weight = state_dict.pop(key, None)
        if weight is None:
            continue
        with torch.no_grad():
            slices = tuple(min(a, b) for a, b in zip(param.shape, weight.shape))
            if param.ndim == 1:
                param[: slices[0]].copy_(weight[: slices[0]])
            else:
                param[: slices[0], : slices[1]].copy_(weight[: slices[0], : slices[1]])
        state_dict[key] = param.detach().clone()

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if verbose:
        if missing:
            print(f"[Warn] Missing keys: {missing}")
        if unexpected:
            print(f"[Warn] Unexpected keys: {unexpected}")


def _disable_dropout(model: nn.Module) -> int:
    """Set every ``nn.Dropout`` (and Dropout1d/2d/3d) module's probability to 0.

    Critical for GRPO: with GPT-2 default ``attn_pdrop=resid_pdrop=embd_pdrop=0.1``,
    the policy model (in train mode) would produce stochastic log-probabilities
    while the reference (frozen, in eval mode) is deterministic.  This makes the
    importance-sampling ratio ``exp(logp_pi - logp_ref)`` fluctuate randomly
    around 1.0 from step 0 (KL ~ 0.08 measured empirically), defeating the PPO
    clip and KL penalty.  Returns the number of dropout modules zeroed.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.p = 0.0
            n += 1
    return n


def build_unified_voice(
    cfg_path: Path,
    tokenizer: TextTokenizer,
    checkpoint_path: Path,
    gradient_checkpointing: bool = False,
) -> UnifiedVoice:
    cfg = OmegaConf.load(cfg_path)
    vocab_size = tokenizer.vocab_size
    if cfg.gpt.number_text_tokens != vocab_size:
        cfg.gpt.number_text_tokens = vocab_size
    # Note: we pass ``checkpointing`` through to ``UnifiedVoice`` so the
    # underlying ``GPT2Config`` is built with ``gradient_checkpointing`` +
    # ``use_cache=False`` from the start, then re-enable via the modern
    # ``gradient_checkpointing_enable`` API to force ``use_reentrant=False``.
    # Reentrant checkpointing requires at least one input tensor to have
    # ``requires_grad=True``, but our GPT inputs are pre-computed embeddings
    # (conditioning under ``no_grad`` + frozen text/mel embeddings) so the
    # reentrant path would silently break the backward graph.
    model = UnifiedVoice(**cfg.gpt, checkpointing=gradient_checkpointing)
    _load_gpt_checkpoint(model, checkpoint_path)
    n_drop = _disable_dropout(model)
    print(f"[Init] Disabled {n_drop} nn.Dropout module(s) for deterministic policy/ref logp.")
    if gradient_checkpointing:
        try:
            model.gpt.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            # Older transformers without ``gradient_checkpointing_kwargs`` arg.
            model.gpt.gradient_checkpointing_enable()
        model.gpt.config.use_cache = False
        print("[Init] Gradient checkpointing enabled on GPT (use_reentrant=False).")
    return model


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def _set_module_trainable(module: nn.Module, trainable: bool):
    for p in module.parameters():
        p.requires_grad = trainable


_ALWAYS_FROZEN_MODULES = (
    "conditioning_encoder",
    "perceiver_encoder",
    "emo_conditioning_encoder",
    "emo_perceiver_encoder",
    "emo_layer",
    "emovec_layer",
    "speed_emb",
    "text_head",
)


def configure_policy_trainable(
    model: UnifiedVoice,
    train_scope: str = "body_and_head",
    gpt_train_mode: str = "full",
) -> Dict[str, int]:
    """Configure which UnifiedVoice weights receive gradients.

    Returns a summary dict with trainable / total parameter counts.
    """
    for p in model.parameters():
        p.requires_grad = False

    for name in _ALWAYS_FROZEN_MODULES:
        sub = getattr(model, name, None)
        if sub is not None:
            freeze_module(sub)

    trainable_entries: List[str] = []

    if train_scope == "lm_core":
        for name in ("text_embedding", "text_pos_embedding", "mel_embedding", "mel_pos_embedding"):
            sub = getattr(model, name, None)
            if sub is not None:
                _set_module_trainable(sub, True)
                trainable_entries.append(name)
    else:
        for name in ("text_embedding", "text_pos_embedding", "mel_embedding", "mel_pos_embedding"):
            freeze_module(getattr(model, name))

    _set_module_trainable(model.final_norm, True)
    trainable_entries.append("final_norm")

    if train_scope in ("body_and_head", "lm_core"):
        _set_module_trainable(model.mel_head, True)
        trainable_entries.append("mel_head")
    else:
        freeze_module(model.mel_head)

    if gpt_train_mode == "full":
        _set_module_trainable(model.gpt, True)
        trainable_entries.append("gpt (full)")
    elif gpt_train_mode == "attention_only":
        freeze_module(model.gpt)
        for block in model.gpt.h:
            _set_module_trainable(block.attn, True)
        trainable_entries.append("gpt.h.*.attn")
    else:
        raise ValueError(f"Unsupported gpt_train_mode: {gpt_train_mode}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "train_scope": train_scope,
        "gpt_train_mode": gpt_train_mode,
        "modules": trainable_entries,
    }


def log_trainable_summary(model: UnifiedVoice, summary: Dict[str, int], prefix: str = ""):
    pct = 100.0 * summary["trainable"] / max(summary["total"], 1)
    print(
        f"{prefix}[Trainable] scope={summary['train_scope']} "
        f"gpt_mode={summary['gpt_train_mode']} | "
        f"{summary['trainable']:,} / {summary['total']:,} params ({pct:.2f}%)"
    )
    print(f"{prefix}[Trainable] modules: {', '.join(summary['modules'])}")


def find_most_similar_cosine(query_vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    query_vector = query_vector.float()
    matrix = matrix.float()
    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    return torch.argmax(similarities)


# =============================================================================
# Feature pre-processor (audio -> spk_cond_emb + codes)
# =============================================================================

class FeaturePreprocessor(nn.Module):
    """Holds the (frozen) modules needed to derive conditioning + target codes
    from raw 16kHz audio: GPUFeatureExtractor + semantic_model + semantic_codec.

    The policy model's `conditioning_encoder` / `perceiver_encoder` are used
    (under ``torch.no_grad``) to map spk_cond_emb -> conditioning. Since those
    layers are frozen, the policy and reference models would produce identical
    conditioning so we just reuse the policy's.
    """

    def __init__(self, model_dir: Path, cfg, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype

        hf_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            str(model_dir / "w2v-bert-2.0")
        )
        self.gpu_feature_extractor = GPUFeatureExtractor(
            mel_filters_np=hf_extractor.mel_filters,
            window_np=hf_extractor.window,
        )
        self.gpu_feature_extractor.eval()
        del hf_extractor

        self.semantic_model, sem_mean, sem_std = build_semantic_model(
            str(model_dir / cfg.w2v_stat),
            str(model_dir / "w2v-bert-2.0"),
        )
        self.semantic_model.eval()
        self.register_buffer("semantic_mean", sem_mean.to(dtype=dtype))
        self.register_buffer("semantic_std", sem_std.to(dtype=dtype))

        self.semantic_codec = build_semantic_codec(cfg.semantic_codec)
        safetensors.torch.load_model(
            self.semantic_codec,
            str(model_dir / "semantic_codec/model.safetensors"),
        )
        self.semantic_codec.eval()

        campplus_ckpt_path = model_dir / "campplus/campplus_cn_common.bin"
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model.eval()

        emo_matrix = torch.load(model_dir / str(cfg.emo_matrix).strip(), map_location="cpu").to(device=device, dtype=dtype)
        spk_matrix = torch.load(model_dir / str(cfg.spk_matrix).strip(), map_location="cpu").to(device=device, dtype=dtype)
        self.emo_num = list(cfg.emo_num)
        self.emo_matrix = torch.split(emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(spk_matrix, self.emo_num)

        freeze_module(self)
        self.to(device=device, dtype=dtype)

    @torch.no_grad()
    def get_spk_emb(self, input_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Force fp32: semantic_model (w2v-bert) is loaded in fp32 and is sensitive
        # to mixed precision (its conv front-end overflows in fp16 for some inputs).
        with torch.amp.autocast(device_type="cuda", enabled=False):
            vq_emb = self.semantic_model(
                input_features=input_features.float(),
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            feat = vq_emb.hidden_states[17].float()
            feat = (feat - self.semantic_mean) / self.semantic_std
            return feat

    @torch.no_grad()
    def extract_spk_cond_emb(self, wavs_16k: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (spk_cond_emb [B, T, 1024], cond_lengths [B])."""
        with torch.amp.autocast(device_type="cuda", enabled=False):
            wavs_on_dev = [w.to(self.device, non_blocking=True).float() for w in wavs_16k]
            inputs = self.gpu_feature_extractor(wavs_on_dev)
            spk_cond_emb = self.get_spk_emb(inputs["input_features"], inputs["attention_mask"])
            cond_lengths = inputs["attention_mask"].sum(dim=1).long()
            return spk_cond_emb, cond_lengths

    @torch.no_grad()
    def extract_codes(self, wavs_16k: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (codes [B, T_max], code_lengths [B])."""
        with torch.amp.autocast(device_type="cuda", enabled=False):
            spk_cond_emb, code_lengths = self.extract_spk_cond_emb(wavs_16k)
            # semantic_codec.quantize uses dist/argmin -> keep in fp32.
            codes, _ = self.semantic_codec.quantize(spk_cond_emb.float())
            return codes.long(), code_lengths

    @torch.no_grad()
    def extract_styles(self, wavs_16k: List[torch.Tensor]) -> torch.Tensor:
        """Return CAM++ style embeddings [B, 192] for reference audios."""
        with torch.amp.autocast(device_type="cuda", enabled=False):
            styles = []
            for wav in wavs_16k:
                audio = wav.to(self.device, non_blocking=True).float().unsqueeze(0)
                feat = torchaudio.compliance.kaldi.fbank(
                    audio,
                    num_mel_bins=80,
                    dither=0,
                    sample_frequency=TARGET_SR,
                )
                feat = feat - feat.mean(dim=0, keepdim=True)
                style = self.campplus_model(feat.unsqueeze(0).to(device=self.device, dtype=self.dtype))
                styles.append(style)
            return torch.cat(styles, dim=0)

    @torch.no_grad()
    def build_label_emovec(self, styles: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Match inference's feat1/feat2 emotion-vector lookup and weighted sum."""
        with torch.amp.autocast(device_type="cuda", enabled=False):
            out = []
            for style, weight_vector in zip(styles, weights):
                random_index = [
                    find_most_similar_cosine(style.unsqueeze(0), matrix)
                    for matrix in self.spk_matrix
                ]
                emo_matrix = [
                    matrix[index].unsqueeze(0)
                    for index, matrix in zip(random_index, self.emo_matrix)
                ]
                emo_matrix = torch.cat(emo_matrix, dim=0).to(device=self.device, dtype=self.dtype)
                weight_vector = weight_vector.to(device=self.device, dtype=self.dtype)
                emovec_mat = torch.sum(weight_vector.unsqueeze(1) * emo_matrix, dim=0)
                out.append(emovec_mat.unsqueeze(0))
            return torch.cat(out, dim=0)


# =============================================================================
# Policy / Reference per-token logp
# =============================================================================

def gpt_per_token_logp(
    model: UnifiedVoice,
    conditioning: torch.Tensor,         # [B, 32, d]
    emo_vec: torch.Tensor,              # [B, d]   (zeros for now)
    text_ids: torch.Tensor,             # [B, T_text]   long
    text_lengths: torch.Tensor,         # [B]            long
    codes: torch.Tensor,                # [B, T_code]   long
    code_lengths: torch.Tensor,         # [B]            long
    use_duration_control: bool = False,
    duration_dropout_mask: Optional[torch.Tensor] = None,  # [B] bool, True = drop
    return_entropy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Run a teacher-forced GPT forward pass and return (per-token logp, mask,
    optional per-position entropy).

    The mel target sequence is built exactly like the SFT path:
        target  = [code_0, code_1, ..., code_{L-1}, STOP_MEL, STOP_MEL, ...]
        input   = [START_MEL, code_0, ..., code_{L-2}, code_{L-1}, STOP_MEL, ...]
    The valid mask covers the first ``code_len + 1`` positions (predicting up to
    and including the EOS).
    """
    target_device = text_ids.device
    batch_size = text_ids.size(0)
    use_speed = torch.zeros(batch_size, dtype=torch.long, device=target_device)

    # Make sure the float conditioning tensors match the model's parameter
    # dtype (relevant when the reference GPT is stored in fp16 / bf16 to save
    # memory).  Otherwise the cat below would mix fp32 conditioning with the
    # fp16 embedding output and raise a dtype mismatch.
    param_dtype = next(model.parameters()).dtype
    if conditioning.dtype != param_dtype:
        conditioning = conditioning.to(param_dtype)
    if emo_vec.dtype != param_dtype:
        emo_vec = emo_vec.to(param_dtype)

    text_inputs = model.set_text_padding(text_ids.clone(), text_lengths)
    text_inputs = F.pad(text_inputs, (0, 1), value=model.stop_text_token)
    text_inputs, text_targets = model.build_aligned_inputs_and_targets(
        text_inputs, model.start_text_token, model.stop_text_token
    )

    mel_inputs = model.set_mel_padding(codes.clone(), code_lengths)
    mel_inputs = F.pad(mel_inputs, (0, 1), value=model.stop_mel_token)
    mel_inputs, mel_targets = model.build_aligned_inputs_and_targets(
        mel_inputs, model.start_mel_token, model.stop_mel_token
    )

    duration_free = model.speed_emb(torch.zeros_like(use_speed))
    if use_duration_control:
        duration_ctrl = model.get_duration_embeddings(code_lengths)
        if duration_dropout_mask is not None and duration_dropout_mask.any():
            duration_ctrl = torch.where(
                duration_dropout_mask.unsqueeze(1), duration_free, duration_ctrl
            )
    else:
        duration_ctrl = model.speed_emb(torch.ones_like(use_speed))

    cond_input = conditioning + emo_vec.unsqueeze(1)
    conds = torch.cat(
        (cond_input, duration_ctrl.unsqueeze(1), duration_free.unsqueeze(1)),
        dim=1,
    )

    text_emb = model.text_embedding(text_inputs) + model.text_pos_embedding(text_inputs)
    mel_emb = model.mel_embedding(mel_inputs) + model.mel_pos_embedding(mel_inputs)

    _, mel_logits = model.get_logits(conds, text_emb, model.text_head,
                                     mel_emb, model.mel_head)
    # mel_logits: [B, V, T] -- promote to fp32 for log_softmax stability with
    # an 8194-class output; fp16 log_softmax over wide vocabs has poor tail
    # precision (matters for the KL/ratio terms).
    log_probs = F.log_softmax(mel_logits.float(), dim=1)
    per_token_logp = log_probs.gather(1, mel_targets.unsqueeze(1)).squeeze(1)  # [B, T]

    mel_mask = (
        torch.arange(mel_targets.size(1), device=target_device).unsqueeze(0)
        < (code_lengths + 1).unsqueeze(1)
    ).float()  # [B, T]

    entropy = None
    if return_entropy:
        # Per-position categorical entropy of the policy (token-distribution).
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=1)  # [B, T]

    return per_token_logp, mel_mask, entropy


# =============================================================================
# Per-reference-audio feature cache
# =============================================================================

class RefFeatureCache:
    """In-memory cache for the per-reference-audio features that feed the GPT.

    All three cached tensors are *pure deterministic functions* of the ref
    audio plus the (frozen) sub-modules
    ``policy_model.conditioning_encoder / perceiver_encoder /
    emo_conditioning_encoder / emo_perceiver_encoder / emovec_layer / emo_layer``
    and ``feature_extractor.campplus_model``.  None of those weights change
    during GRPO, so cache entries are valid for the full training run.

    Each entry stores (per ref audio stem):
      * ``conditioning``  : [cond_num, model_dim]  (e.g. [32, 1280])
      * ``emo_vec_base``  : [model_dim]            (output of ``get_emovec``)
      * ``style``         : [192]                  (CAM++ output, used only by
                                                    the text-emo control branch)

    Storage is fp32 on CPU, ~166 KB / ref for model_dim=1280, ~376 MB for the
    full multigen metadata (~2.3k unique voice ids).  Each DDP rank keeps its
    own cache; no cross-process synchronisation is required.

    The cache deliberately stores the *base* emo vec (before mixing with the
    text-side emo control vector), since the mixing weights depend on the
    text and must therefore be recomputed every step.
    """

    def __init__(self, max_entries: Optional[int] = None):
        self._cond: Dict[str, torch.Tensor] = {}
        self._emo: Dict[str, torch.Tensor] = {}
        self._style: Dict[str, torch.Tensor] = {}
        self._max_entries = max_entries
        self.hits = 0
        self.misses = 0

    def __contains__(self, key: str) -> bool:
        return key in self._cond

    def __len__(self) -> int:
        return len(self._cond)

    def get(self, key: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._cond[key], self._emo[key], self._style[key]

    def put(
        self,
        key: str,
        conditioning: torch.Tensor,
        emo_vec: torch.Tensor,
        style: torch.Tensor,
    ) -> None:
        if self._max_entries is not None and len(self._cond) >= self._max_entries and key not in self._cond:
            # Simple FIFO eviction; this codebase only ever fills a few thousand
            # entries so we never expect to hit this in practice.
            evict_key = next(iter(self._cond))
            self._cond.pop(evict_key, None)
            self._emo.pop(evict_key, None)
            self._style.pop(evict_key, None)
        self._cond[key] = conditioning.detach().to(device="cpu", dtype=torch.float32).contiguous()
        self._emo[key] = emo_vec.detach().to(device="cpu", dtype=torch.float32).contiguous()
        self._style[key] = style.detach().to(device="cpu", dtype=torch.float32).contiguous()

    def memory_bytes(self) -> int:
        if not self._cond:
            return 0
        sample_key = next(iter(self._cond))
        per_ref = (
            self._cond[sample_key].numel()
            + self._emo[sample_key].numel()
            + self._style[sample_key].numel()
        ) * 4  # fp32
        return per_ref * len(self._cond)

    def stats_str(self) -> str:
        total = self.hits + self.misses
        rate = (self.hits / total * 100.0) if total > 0 else 0.0
        return (
            f"entries={len(self)} hit_rate={rate:.1f}% "
            f"(hits={self.hits} misses={self.misses}) "
            f"mem={self.memory_bytes() / (1024 * 1024):.1f}MB"
        )


# =============================================================================
# Batch preparation (groups -> flattened tensors ready for forward)
# =============================================================================

def prepare_grpo_batch(
    groups: List[GroupItem],
    feature_extractor: FeaturePreprocessor,
    policy_model: UnifiedVoice,
    device: torch.device,
    max_samples_per_batch: int,
    ref_cache: Optional[RefFeatureCache] = None,
) -> Optional[Dict[str, Any]]:
    """Turn a list of GroupItems into ready-to-train tensors.

    Returns ``None`` if the resulting batch is empty.

    When ``ref_cache`` is provided, per-reference-audio features
    (``conditioning``, ``emo_vec_base``, ``style``) are looked up by
    ``GroupItem.ref_audio_stem``; misses run the full feature pipeline on
    just the missing subset and are then written back to the cache.
    """
    if not groups:
        return None

    # Optional sub-sampling at the batch level to stay under ``max_samples_per_batch``.
    total = sum(len(g.candidates) for g in groups)
    if total > max_samples_per_batch:
        kept_groups: List[GroupItem] = []
        budget = max_samples_per_batch
        for g in groups:
            if budget <= 1:
                break
            if len(g.candidates) <= budget:
                kept_groups.append(g)
                budget -= len(g.candidates)
            else:
                # Stratified pick from this group
                chosen = [c for c in g.candidates if c.reward >= 0.5]
                rejected = [c for c in g.candidates if c.reward < 0.5]
                if not chosen or not rejected or budget < 2:
                    continue
                target_c = max(1, round(budget * len(chosen) / len(g.candidates)))
                target_c = min(target_c, len(chosen))
                target_r = max(1, budget - target_c)
                target_r = min(target_r, len(rejected))
                target_c = max(1, budget - target_r)
                target_c = min(target_c, len(chosen))
                picks = random.sample(chosen, target_c) + random.sample(rejected, target_r)
                random.shuffle(picks)
                kept_groups.append(GroupItem(
                    text_ids=g.text_ids,
                    ref_wav_16k=g.ref_wav_16k,
                    candidates=picks,
                    group_key=g.group_key,
                    emo_control_vector=g.emo_control_vector,
                    ref_audio_stem=g.ref_audio_stem,
                ))
                budget -= len(picks)
        groups = kept_groups

    # Final integrity check.
    valid_groups: List[GroupItem] = []
    for g in groups:
        rewards = {c.reward for c in g.candidates}
        if len(g.candidates) >= 2 and len(rewards) >= 2:
            valid_groups.append(g)
    groups = valid_groups
    if not groups:
        return None

    # 1) Reference-audio conditioning (1 per group).
    #    Matches `infer_v2.py` which calls:
    #        spk_cond_emb = get_emb(SeamlessM4T(audio_16k))    # [b, T, 1024]
    #        speech_conditioning_latent = get_conditioning(spk_cond_emb.transpose(1, 2), cond_lengths)
    #    Note: inference passes `cond_lengths=spk_cond_emb.shape[-1]=1024` (model_dim),
    #    which acts as "no masking" since T<<1024.  We pass the actual frame count from
    #    the GPU feature extractor's attention_mask, which is strictly more correct
    #    and equivalent when each batch item's audio is processed alone.
    # All preprocessing (FE + frozen conditioning / emo encoders) is forced to
    # fp32; both because the conv front-end of w2v-bert overflows in fp16 and
    # because we want the conditioning to be reproducible bit-for-bit between
    # the policy and reference forwards.
    # Split groups by cache hit/miss.  A group is a miss if no cache was
    # provided, the group has no ref_audio_stem, or its stem is not in cache.
    G = len(groups)
    cond_slots: List[Optional[torch.Tensor]] = [None] * G
    emo_base_slots: List[Optional[torch.Tensor]] = [None] * G
    style_slots: List[Optional[torch.Tensor]] = [None] * G
    miss_idx: List[int] = []
    for i, g in enumerate(groups):
        key = g.ref_audio_stem
        if ref_cache is not None and key is not None and key in ref_cache:
            cond_c, emo_c, style_c = ref_cache.get(key)
            cond_slots[i] = cond_c
            emo_base_slots[i] = emo_c
            style_slots[i] = style_c
            ref_cache.hits += 1
        else:
            miss_idx.append(i)
            if ref_cache is not None:
                ref_cache.misses += 1

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
        # Run the full feature pipeline only on the cache misses.
        if miss_idx:
            miss_ref_wavs = [groups[i].ref_wav_16k for i in miss_idx]
            spk_cond_emb_ref, cond_len_ref = feature_extractor.extract_spk_cond_emb(miss_ref_wavs)
            feat_t = spk_cond_emb_ref.transpose(1, 2)  # (G_miss, 1024, T)
            cond_miss = policy_model.get_conditioning(feat_t, cond_len_ref)  # [G_miss, 32, d]
            # Base emo vec from the same reference audio.  Mirrors inference
            # ``gpt.merge_emovec(..., emo_audio == spk_audio, alpha=1.0)``.
            emo_base_miss = policy_model.get_emovec(spk_cond_emb_ref, cond_len_ref)  # [G_miss, d]
            # Always compute CAM++ style on a miss so the cache stays uniform,
            # even if the current batch has no text-emo control vector.
            style_miss = feature_extractor.extract_styles(miss_ref_wavs)  # [G_miss, 192]

            for j, i in enumerate(miss_idx):
                cond_slots[i] = cond_miss[j]
                emo_base_slots[i] = emo_base_miss[j]
                style_slots[i] = style_miss[j]
                key = groups[i].ref_audio_stem
                if ref_cache is not None and key is not None:
                    ref_cache.put(key, cond_miss[j], emo_base_miss[j], style_miss[j])

        # Stack into per-group tensors.  Each slot was either fetched from
        # the (CPU fp32) cache or just computed on-device; ``.to(device)``
        # is a no-op for the latter.
        cond_per_group = torch.stack(
            [t.to(device=device, dtype=torch.float32) for t in cond_slots], dim=0
        )  # [G, 32, d]
        emo_vec_per_group = torch.stack(
            [t.to(device=device, dtype=torch.float32) for t in emo_base_slots], dim=0
        )  # [G, d]

        if any(g.emo_control_vector is not None for g in groups):
            # Text labels like [Sadness:2;Surprise:5] are mixed the same way as
            # api_server -> infer_vllm_v2_batch.py: emovec_mat + residual base.
            emo_controls = torch.stack([
                g.emo_control_vector if g.emo_control_vector is not None
                else torch.zeros(len(EMOTION_ORDER), dtype=torch.float32)
                for g in groups
            ]).to(device=device, dtype=cond_per_group.dtype)
            styles = torch.stack(
                [t.to(device=device, dtype=torch.float32) for t in style_slots], dim=0
            )  # [G, 192]
            label_emovec = feature_extractor.build_label_emovec(styles, emo_controls)
            residual_scale = 1.0 - emo_controls.sum(dim=1, keepdim=True)
            emo_vec_per_group = label_emovec + residual_scale * emo_vec_per_group

        # 2) Candidate audio -> semantic codes (the y_i token sequence).
        cand_wavs: List[torch.Tensor] = []
        for g in groups:
            cand_wavs.extend([c.wav_16k for c in g.candidates])
        codes_padded, code_lens = feature_extractor.extract_codes(cand_wavs)

    # 3) Per-sample text / conditioning / emo / reward / group index.
    text_ids_list: List[torch.Tensor] = []
    text_lens_list: List[int] = []
    rewards_list: List[float] = []
    group_index_list: List[int] = []
    for g_idx, g in enumerate(groups):
        for cand in g.candidates:
            text_ids_list.append(g.text_ids)
            text_lens_list.append(int(g.text_ids.numel()))
            rewards_list.append(cand.reward)
            group_index_list.append(g_idx)

    text_padded = pad_sequence(text_ids_list, batch_first=True, padding_value=0)
    text_lengths = torch.tensor(text_lens_list, dtype=torch.long)

    group_index = torch.tensor(group_index_list, dtype=torch.long,
                               device=cond_per_group.device)
    conditioning = cond_per_group.index_select(0, group_index)  # [B, 32, d]
    emo_vec = emo_vec_per_group.index_select(0, group_index)    # [B, d]
    rewards = torch.tensor(rewards_list, dtype=torch.float32)

    return {
        "conditioning": conditioning.to(device),
        "emo_vec": emo_vec.to(device),
        "text_ids": text_padded.to(device),
        "text_lengths": text_lengths.to(device),
        "codes": codes_padded.to(device),
        "code_lengths": code_lens.to(device),
        "rewards": rewards.to(device),
        "group_index": group_index.to(device),
        "num_groups": len(groups),
    }


# =============================================================================
# Advantage + GRPO loss
# =============================================================================

def compute_advantages(rewards: torch.Tensor, group_index: torch.Tensor, num_groups: int,
                       norm_strategy: str, eps: float = 1e-8
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-sample advantages and a per-sample validity mask.

    Returns ``(advantages, valid_mask)`` where ``valid_mask[i] == True`` means
    candidate ``i`` should contribute to the policy / KL loss.  A candidate is
    invalid iff its group has fewer than 2 samples or all rewards in the group
    are identical (zero variance).

    ``norm_strategy``:
      - ``intra_group``: (R - mean_g) / (std_g + eps).
      - ``global_batch``: intra-group centring first, then standardise across
        the whole batch over the *valid* samples (see arXiv:2511.21270).

    The validity is decided strictly from ``group_index`` + ``rewards``, not
    from the numerical value of the centred advantage.  This avoids a subtle
    bug where a candidate whose reward equals its group mean (centred=0) used
    to be silently dropped as if it belonged to a degenerate group.
    """
    advantages = torch.zeros_like(rewards)
    valid_mask = torch.zeros_like(rewards, dtype=torch.bool)

    for g in range(num_groups):
        mask = (group_index == g)
        if mask.sum() < 2:
            continue
        r_g = rewards[mask]
        std_g = r_g.std(unbiased=False)
        if std_g.item() < 1e-6:
            continue
        mean_g = r_g.mean()
        if norm_strategy == "intra_group":
            advantages[mask] = (r_g - mean_g) / (std_g + eps)
        else:
            advantages[mask] = r_g - mean_g
        valid_mask[mask] = True

    if norm_strategy == "global_batch":
        if valid_mask.sum().item() >= 2:
            sub = advantages[valid_mask]
            advantages = (advantages - sub.mean()) / (sub.std(unbiased=False) + eps)
            # Re-zero anything outside the valid mask (the centring above pushed
            # invalid samples away from 0).
            advantages = advantages * valid_mask.to(advantages.dtype)
        else:
            advantages = torch.zeros_like(advantages)

    return advantages, valid_mask


def compute_grpo_loss(
    logp_policy: torch.Tensor,    # [B, T]
    logp_ref: torch.Tensor,       # [B, T]
    mask: torch.Tensor,           # [B, T] float (1 valid, 0 pad)
    advantages: torch.Tensor,     # [B]
    clip_eps: float,
    kl_coeff: float,
    kl_estimator: str = "k3",
    entropy: Optional[torch.Tensor] = None,
    entropy_coeff: float = 0.0,
    sample_valid: Optional[torch.Tensor] = None,  # [B] bool/float, explicit per-sample mask
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Mean-over-tokens, mean-over-samples GRPO loss with PPO-style clipping.

    ``sample_valid`` is the authoritative per-sample mask (1 = include in the
    loss, 0 = skip).  It should come from ``compute_advantages`` and reflects
    group-level validity, not the numerical value of ``advantages``.  When not
    provided, falls back to the legacy ``advantages.abs() > 0`` heuristic.
    """
    valid_token_counts = mask.sum(dim=1).clamp_min(1.0)  # [B]

    diff = logp_policy - logp_ref  # [B, T]
    ratio = torch.exp(diff)

    adv_expand = advantages.unsqueeze(1)  # [B, 1]
    surr1 = ratio * adv_expand
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_expand
    pg_token = -torch.min(surr1, surr2)  # [B, T]

    # KL: positive (ref relative to pi); higher = pi diverges more.
    if kl_estimator == "k1":
        kl_token = logp_ref - logp_policy
    else:  # k3 (unbiased, low-variance)
        log_ratio = logp_ref - logp_policy
        kl_token = torch.exp(log_ratio) - log_ratio - 1.0

    pg_per_sample = (pg_token * mask).sum(dim=1) / valid_token_counts
    kl_per_sample = (kl_token * mask).sum(dim=1) / valid_token_counts

    if sample_valid is None:
        sample_valid = (advantages.abs() > 0)
    valid_samples = sample_valid.to(pg_per_sample.dtype)
    n_valid = valid_samples.sum().clamp_min(1.0)

    policy_loss = (pg_per_sample * valid_samples).sum() / n_valid
    kl_loss = (kl_per_sample * valid_samples).sum() / n_valid

    loss = policy_loss + kl_coeff * kl_loss

    metrics = {
        "policy_loss": policy_loss.detach(),
        "kl": kl_loss.detach(),
        "ratio_mean": ((ratio * mask).sum() / mask.sum().clamp_min(1)).detach(),
        "valid_samples": valid_samples.sum().detach(),
    }

    if entropy is not None and entropy_coeff > 0:
        ent_per_sample = (entropy * mask).sum(dim=1) / valid_token_counts
        ent_mean = (ent_per_sample * valid_samples).sum() / n_valid
        loss = loss - entropy_coeff * ent_mean
        metrics["entropy"] = ent_mean.detach()

    metrics["total_loss"] = loss.detach()
    return loss, metrics


# =============================================================================
# GRPO loss wrapper (so Accelerate can wrap it like the SFT version)
# =============================================================================

_POLICY_FROZEN_SUBMODULES = _ALWAYS_FROZEN_MODULES


def _pin_frozen_submodules_eval(model: nn.Module):
    """Keep every fully-frozen submodule in eval mode (LayerNorm / Dropout)."""
    for module in model.modules():
        params = list(module.parameters(recurse=False))
        if params and all(not p.requires_grad for p in params):
            module.eval()


class GRPOLossWrapper(nn.Module):
    """Wraps the policy GPT and the (frozen) reference GPT for end-to-end
    forward + GRPO loss in a single ``nn.Module`` call.

    The reference model is held inside this wrapper but its parameters are not
    registered for training (``requires_grad=False`` + ``eval()`` mode), and is
    *always* kept in ``eval()`` mode even when ``train()`` is called on the
    wrapper.  The frozen submodules of the policy model are similarly pinned to
    eval mode so that any BatchNorm / dropout inside ``conditioning_encoder`` etc.
    does not get updated.
    """

    def __init__(
        self,
        policy_model: UnifiedVoice,
        ref_model: UnifiedVoice,
        clip_eps: float,
        kl_coeff: float,
        kl_estimator: str,
        entropy_coeff: float,
        use_duration_control: bool,
        duration_dropout: float,
    ):
        super().__init__()
        self.policy_model = policy_model
        self.ref_model = ref_model
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()

        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff
        self.kl_estimator = kl_estimator
        self.entropy_coeff = entropy_coeff
        self.use_duration_control = use_duration_control
        self.duration_dropout = duration_dropout

    def _pin_eval_submodules(self):
        """Force ref_model and fully-frozen policy submodules into eval mode."""
        self.ref_model.eval()
        _pin_frozen_submodules_eval(self.policy_model)

    def train(self, mode: bool = True):
        super().train(mode)
        self._pin_eval_submodules()
        return self

    def eval(self):
        super().eval()
        self._pin_eval_submodules()
        return self

    def forward(self, batch: Dict[str, Any], advantages: torch.Tensor,
                sample_valid: Optional[torch.Tensor] = None):
        text_ids = batch["text_ids"]
        text_lengths = batch["text_lengths"]
        codes = batch["codes"]
        code_lengths = batch["code_lengths"]
        conditioning = batch["conditioning"]
        emo_vec = batch["emo_vec"]

        # Duration-control dropout shared between policy & ref so that ratios match.
        duration_dropout_mask = None
        if self.use_duration_control and self.duration_dropout > 0.0:
            duration_dropout_mask = (
                torch.rand(code_lengths.size(0), device=code_lengths.device)
                < self.duration_dropout
            )

        # Policy forward (with grad).
        logp_pi, mask, entropy = gpt_per_token_logp(
            self.policy_model,
            conditioning=conditioning,
            emo_vec=emo_vec,
            text_ids=text_ids,
            text_lengths=text_lengths,
            codes=codes,
            code_lengths=code_lengths,
            use_duration_control=self.use_duration_control,
            duration_dropout_mask=duration_dropout_mask,
            return_entropy=(self.entropy_coeff > 0),
        )

        # Reference forward (no grad).
        with torch.no_grad():
            logp_ref, _, _ = gpt_per_token_logp(
                self.ref_model,
                conditioning=conditioning,
                emo_vec=emo_vec,
                text_ids=text_ids,
                text_lengths=text_lengths,
                codes=codes,
                code_lengths=code_lengths,
                use_duration_control=self.use_duration_control,
                duration_dropout_mask=duration_dropout_mask,
                return_entropy=False,
            )

        loss, metrics = compute_grpo_loss(
            logp_policy=logp_pi,
            logp_ref=logp_ref,
            mask=mask,
            advantages=advantages,
            clip_eps=self.clip_eps,
            kl_coeff=self.kl_coeff,
            kl_estimator=self.kl_estimator,
            entropy=entropy,
            entropy_coeff=self.entropy_coeff,
            sample_valid=sample_valid,
        )
        return loss, metrics


# =============================================================================
# Misc helpers
# =============================================================================

def load_tokenizer(tokenizer_path: Path) -> TextTokenizer:
    normalizer = TextNormalizer()
    normalizer.load()
    return TextTokenizer(str(tokenizer_path), normalizer)


def rotate_checkpoints(output_dir: Path, keep_last: int, major_save_every: int, is_main: bool):
    if not is_main:
        return
    glob_checkpoints = list(output_dir.glob("checkpoint-*"))
    checkpoints = []
    for path in glob_checkpoints:
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-")[-1])
            checkpoints.append((step, path))
        except ValueError:
            continue
    checkpoints.sort(key=lambda x: x[0])

    regular_checkpoints = []
    for step, path in checkpoints:
        if major_save_every > 0 and step % major_save_every == 0:
            continue
        regular_checkpoints.append((step, path))

    if len(regular_checkpoints) > keep_last:
        num_to_delete = len(regular_checkpoints) - keep_last
        to_delete = regular_checkpoints[:num_to_delete]
        for step, folder_path in to_delete:
            print(f"[Checkpoint] Rotate: deleting old step {step} ...")
            try:
                shutil.rmtree(folder_path)
            except OSError as e:
                print(f"  - Failed to delete {folder_path}: {e}")
            pth_path = output_dir / f"model_step{step}.pth"
            if pth_path.exists():
                try:
                    os.remove(pth_path)
                except OSError as e:
                    print(f"  - Failed to delete {pth_path}: {e}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accumulation,
        log_with="wandb",
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.wandb_run_name, "entity": args.wandb_entity}},
        )

    output_dir = args.output_dir.resolve()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- Models -------------------------------------------------------------
    tokenizer = load_tokenizer(args.tokenizer)

    accelerator.print("[Init] Building policy GPT ...")
    policy_model = build_unified_voice(
        args.config,
        tokenizer,
        args.base_checkpoint,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainable_summary = configure_policy_trainable(
        policy_model,
        train_scope=args.train_scope,
        gpt_train_mode=args.gpt_train_mode,
    )
    if accelerator.is_main_process:
        log_trainable_summary(policy_model, trainable_summary, prefix="[Policy] ")

    accelerator.print("[Init] Building reference GPT ...")
    ref_ckpt = args.ref_checkpoint or args.base_checkpoint
    ref_model = build_unified_voice(args.config, tokenizer, ref_ckpt)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    ref_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.ref_dtype]
    if ref_dtype != torch.float32:
        ref_model.to(dtype=ref_dtype)
        if accelerator.is_main_process:
            print(f"[Init] Reference GPT cast to {args.ref_dtype} (saves ~{50 if args.ref_dtype != 'fp32' else 0}% memory).")

    # --- Feature extractor (on accelerator's device, no DDP wrapping) -------
    cfg = OmegaConf.load(args.config)
    feature_extractor = FeaturePreprocessor(args.model_dir, cfg,
                                            device=accelerator.device,
                                            dtype=torch.float32)

    # --- Dataset ------------------------------------------------------------
    dataset = MultigenGRPODataset(
        metadata_path=args.metadata,
        audio_root=args.audio_root,
        ref_audio_root=args.ref_audio_root,
        tokenizer=tokenizer,
        max_group_size=args.max_group_size,
        max_audio_duration=args.max_audio_duration,
        min_audio_duration=args.min_audio_duration,
        max_ref_duration=args.max_ref_duration,
        max_text_tokens=cfg.gpt.max_text_tokens,
        ref_audio_suffix=args.ref_audio_suffix,
        reward_chosen=args.reward_chosen,
        reward_rejected=args.reward_rejected,
        min_group_count=args.min_group_count,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.groups_per_device,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_groups,
        pin_memory=False,  # variable-length tensors per item -> manual transfer
        drop_last=True,
    )

    # --- Ref-audio feature cache -------------------------------------------
    if args.no_ref_cache:
        ref_cache: Optional[RefFeatureCache] = None
        if accelerator.is_main_process:
            print("[RefCache] Disabled via --no-ref-cache.")
    else:
        max_entries = args.ref_cache_max_entries if args.ref_cache_max_entries > 0 else None
        ref_cache = RefFeatureCache(max_entries=max_entries)
        if accelerator.is_main_process:
            cap = "unbounded" if max_entries is None else f"<= {max_entries}"
            print(f"[RefCache] Enabled (CPU fp32, {cap}).")

    # --- Wrapper / Optimiser ------------------------------------------------
    wrapper = GRPOLossWrapper(
        policy_model=policy_model,
        ref_model=ref_model,
        clip_eps=args.clip_eps,
        kl_coeff=args.kl_coeff,
        kl_estimator=args.kl_estimator,
        entropy_coeff=args.entropy_coeff,
        use_duration_control=args.use_duration_control,
        duration_dropout=args.duration_dropout,
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, wrapper.policy_model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = math.ceil(len(loader) / args.grad_accumulation)
    total_steps = args.epochs * num_update_steps_per_epoch
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    if accelerator.is_main_process:
        print(f"[Train] total_steps={total_steps} (steps/epoch={num_update_steps_per_epoch})")

    wrapper, optimizer, loader, scheduler = accelerator.prepare(
        wrapper, optimizer, loader, scheduler
    )

    # --- Resume -------------------------------------------------------------
    global_step = 0
    start_epoch = 0
    resume_batches_to_skip = 0
    if args.resume:
        if args.resume == "auto":
            dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
            resume_path = None
            if dirs:
                dirs.sort(key=lambda x: os.path.getmtime(x))
                resume_path = str(dirs[-1])
        else:
            resume_path = args.resume
        if resume_path and os.path.exists(resume_path):
            accelerator.print(f"[Info] Resuming from {resume_path}")
            accelerator.load_state(resume_path)
            try:
                step_val = int(Path(resume_path).name.split("-")[-1])
                global_step = step_val
                start_epoch = global_step // num_update_steps_per_epoch
                resume_step_in_epoch = global_step % num_update_steps_per_epoch
                resume_batches_to_skip = resume_step_in_epoch * args.grad_accumulation
                accelerator.print(
                    f"[Info] Resume: epoch={start_epoch + 1} step={global_step} "
                    f"skip={resume_batches_to_skip}"
                )
            except Exception:
                pass
        else:
            accelerator.print(f"[Warn] Resume path '{args.resume}' not found.")

    # --- Train loop ---------------------------------------------------------
    wrapper.train()  # GRPOLossWrapper.train() keeps ref + frozen submodules in eval

    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_main_process)
    progress_bar.update(global_step)

    def _unwrap_policy():
        return accelerator.unwrap_model(wrapper).policy_model

    for epoch in range(start_epoch, args.epochs):
        active_loader = loader
        skipped = 0
        if epoch == start_epoch and resume_batches_to_skip > 0:
            active_loader = accelerator.skip_first_batches(loader, resume_batches_to_skip)
            skipped = resume_batches_to_skip

        for batch_idx, groups in enumerate(active_loader):
            # ----- Feature extraction + advantages (no grad, no DDP sync) -----
            with torch.no_grad(), accelerator.autocast():
                batch = prepare_grpo_batch(
                    groups=groups,
                    feature_extractor=feature_extractor,
                    policy_model=_unwrap_policy(),
                    device=accelerator.device,
                    max_samples_per_batch=args.max_samples_per_batch,
                    ref_cache=ref_cache,
                )

            if batch is not None and batch["rewards"].numel() > 0:
                advantages, sample_valid = compute_advantages(
                    rewards=batch["rewards"],
                    group_index=batch["group_index"],
                    num_groups=batch["num_groups"],
                    norm_strategy=args.adv_norm,
                )
                local_valid = float(sample_valid.any().item())
            else:
                advantages = None
                sample_valid = None
                local_valid = 0.0

            # All processes must agree before doing a backward, otherwise DDP
            # desyncs.  We require *every* process to have a non-empty batch.
            valid_tensor = torch.tensor([local_valid], device=accelerator.device)
            gathered = accelerator.gather(valid_tensor)
            if (gathered <= 0).any().item():
                # Skip the iteration uniformly across processes.  No backward, so
                # the accumulate context's step counter does not tick.
                continue

            # ----- Forward + backward (under accumulate context) -----
            with accelerator.accumulate(wrapper):
                loss, metrics = wrapper(batch, advantages, sample_valid=sample_valid)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if args.grad_clip > 0:
                        accelerator.clip_grad_norm_(
                            _unwrap_policy().parameters(),
                            args.grad_clip,
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if global_step % args.log_interval == 0:
                    lr = scheduler.get_last_lr()[0]
                    reward_mean = batch["rewards"].float().mean().item()
                    accelerator.log(
                        {
                            "train/loss": metrics["total_loss"].item(),
                            "train/policy_loss": metrics["policy_loss"].item(),
                            "train/kl": metrics["kl"].item(),
                            "train/ratio_mean": metrics["ratio_mean"].item(),
                            "train/valid_samples": metrics["valid_samples"].item(),
                            "train/reward_mean": reward_mean,
                            "train/lr": lr,
                            "train/epoch": epoch + ((batch_idx + skipped) / max(len(loader), 1)),
                            **(
                                {"train/entropy": metrics["entropy"].item()}
                                if "entropy" in metrics else {}
                            ),
                        },
                        step=global_step,
                    )
                    accelerator.print(
                        f"[GRPO] e={epoch + 1} s={global_step} "
                        f"loss={metrics['total_loss'].item():.4f} "
                        f"pg={metrics['policy_loss'].item():.4f} "
                        f"kl={metrics['kl'].item():.4f} "
                        f"ratio={metrics['ratio_mean'].item():.4f} "
                        f"n={int(metrics['valid_samples'].item())} "
                        f"lr={lr:.2e}"
                    )
                    if ref_cache is not None and accelerator.is_main_process:
                        accelerator.print(f"[RefCache] {ref_cache.stats_str()}")

                is_regular_save = (global_step % args.save_every == 0)
                is_major_save = (args.major_save_every > 0 and global_step % args.major_save_every == 0)
                if (is_regular_save or is_major_save) and global_step > 0:
                    accelerator.wait_for_everyone()
                    save_path = output_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(save_path)
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(wrapper)
                        weight_path = output_dir / f"model_step{global_step}.pth"
                        torch.save({"model": unwrapped.policy_model.state_dict()}, weight_path)
                        print(f"[Checkpoint] Saved {save_path} + {weight_path.name}")
                        rotate_checkpoints(output_dir, args.keep_last,
                                           args.major_save_every,
                                           is_main=True)

                if args.max_steps and global_step >= args.max_steps:
                    break

        if args.max_steps and global_step >= args.max_steps:
            break

    # --- Final ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = output_dir / "checkpoint-final"
        accelerator.save_state(save_path)
        unwrapped = accelerator.unwrap_model(wrapper)
        torch.save({"model": unwrapped.policy_model.state_dict()},
                   output_dir / "model_final.pth")

    accelerator.end_training()
    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
