#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from trainers.train_gpt_v2_grpo_multigpu import (
    EMOTION_ORDER,
    TARGET_SR,
    FeaturePreprocessor,
    _read_audio_to_16k,
    build_unified_voice,
    configure_policy_trainable,
    gpt_per_token_logp,
    load_tokenizer,
    log_trainable_summary,
    parse_text_emotion_tags,
    rotate_checkpoints,
)


@dataclass
class JsonlSFTSample:
    text_ids: torch.Tensor
    wav_16k: torch.Tensor
    ref_wav_16k: torch.Tensor
    audio_path: str
    ref_audio_path: str
    ref_is_self: bool
    emo_control_vector: Optional[torch.Tensor] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="On-the-fly SFT finetune for IndexTTS2 GPT from jsonl metadata."
    )

    parser.add_argument("--metadata", type=Path, required=True, help="Path to jsonl metadata.")
    parser.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="Root used to resolve relative `audio_path` values from metadata.",
    )
    parser.add_argument("--audio-path-key", type=str, default="audio_path")
    parser.add_argument("--text-key", type=str, default="text")
    parser.add_argument("--speaker-key", type=str, default="speaker")
    parser.add_argument(
        "--language-filter",
        type=str,
        default="",
        help="Optional comma-separated language whitelist matched against metadata `language`.",
    )
    parser.add_argument("--min-audio-duration", type=float, default=0.5)
    parser.add_argument("--max-audio-duration", type=float, default=36.0)
    parser.add_argument("--max-text-tokens", type=int, default=600)

    parser.add_argument("--tokenizer", type=Path, default=Path("checkpoints/IndexTTS-2-vLLM/bpe.model"))
    parser.add_argument("--config", type=Path, default=Path("checkpoints/IndexTTS-2-vLLM/config.yaml"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/IndexTTS-2-vLLM/gpt.pth"))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("checkpoints/IndexTTS-2-vLLM"),
        help="Dir containing w2v-bert-2.0/, semantic_codec/, wav2vec2bert_stats.pt.",
    )

    parser.add_argument("--output-dir", type=Path, default=Path("trained_ckpts/sft_jsonl"))
    parser.add_argument("--batch-size-per-device", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-data-size", type=int, default=128)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--major-save-every", type=int, default=25000)
    parser.add_argument("--keep-last", type=int, default=4)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use-duration-control", action="store_true")
    parser.add_argument("--duration-dropout", type=float, default=0.3)
    parser.add_argument(
        "--ref-dropout",
        type=float,
        default=0.1,
        help="Probability of zeroing speaker-reference conditioning when the ref falls back to target audio.",
    )
    parser.add_argument(
        "--emo-dropout",
        type=float,
        default=0.1,
        help="Probability of replacing the emotion vector with zeros during training.",
    )
    parser.add_argument("--no-emo-vec", action="store_true")
    parser.add_argument(
        "--train-scope",
        type=str,
        default="lm_core",
        choices=["body_and_head", "lm_core", "body_only"],
        help="SFT defaults to lm_core, matching the older arrow-based SFT trainer.",
    )
    parser.add_argument(
        "--gpt-train-mode",
        type=str,
        default="full",
        choices=["full", "attention_only"],
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--wandb-project", type=str, default="IndexTTS2-SFT")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb/accelerate trackers.")

    return parser.parse_args()


class JsonlSFTDataset(Dataset):
    def __init__(
        self,
        metadata_path: Path,
        audio_root: Path,
        tokenizer,
        audio_path_key: str,
        text_key: str,
        speaker_key: str,
        language_filter: str,
        min_audio_duration: float,
        max_audio_duration: float,
        max_text_tokens: int,
    ):
        self.audio_root = Path(audio_root)
        self.tokenizer = tokenizer
        self.audio_path_key = audio_path_key
        self.text_key = text_key
        self.speaker_key = speaker_key
        self.min_audio_duration = min_audio_duration
        self.max_audio_duration = max_audio_duration
        self.max_text_tokens = max_text_tokens
        self.languages = {
            x.strip() for x in language_filter.split(",") if x.strip()
        }

        self.entries: List[Dict[str, Any]] = []
        dropped = {
            "parse_error": 0,
            "missing_audio": 0,
            "empty_text": 0,
            "language": 0,
            "duration": 0,
        }

        print(f"[Dataset] Loading metadata from {metadata_path} ...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    dropped["parse_error"] += 1
                    continue

                rel_audio = item.get(audio_path_key)
                if not rel_audio:
                    dropped["missing_audio"] += 1
                    continue

                text = item.get(text_key) or ""
                text, emo_control_vector = parse_text_emotion_tags(str(text))
                if not text:
                    dropped["empty_text"] += 1
                    continue

                if self.languages and str(item.get("language", "")) not in self.languages:
                    dropped["language"] += 1
                    continue

                duration = item.get("duration")
                if duration is not None:
                    duration = float(duration)
                    if duration < min_audio_duration or duration > max_audio_duration:
                        dropped["duration"] += 1
                        continue

                audio_path = Path(str(rel_audio))
                if not audio_path.is_absolute():
                    audio_path = self.audio_root / audio_path

                raw_speaker = item.get(speaker_key)
                speaker = None
                if raw_speaker is not None:
                    speaker = str(raw_speaker).strip() or None

                self.entries.append(
                    {
                        "text": text,
                        "audio_path": str(audio_path),
                        "speaker": speaker,
                        "emo_control_vector": emo_control_vector,
                    }
                )

        self.speaker_to_indices: Dict[str, List[int]] = {}
        for idx, entry in enumerate(self.entries):
            speaker = entry.get("speaker")
            if speaker:
                self.speaker_to_indices.setdefault(speaker, []).append(idx)

        same_speaker_samples = sum(
            len(indices) for indices in self.speaker_to_indices.values() if len(indices) > 1
        )
        multi_speaker_groups = sum(
            1 for indices in self.speaker_to_indices.values() if len(indices) > 1
        )
        print(f"[Dataset] Loaded {len(self.entries)} samples. Dropped: {dropped}")
        print(
            f"[Dataset] Same-speaker reference candidates: "
            f"{same_speaker_samples} samples across {multi_speaker_groups} speakers."
        )

    def __len__(self) -> int:
        return len(self.entries)

    def _candidate_ref_indices(self, idx: int) -> List[int]:
        speaker = self.entries[idx].get("speaker")
        if not speaker:
            return []
        current_audio = self.entries[idx]["audio_path"]
        return [
            cand_idx
            for cand_idx in self.speaker_to_indices.get(speaker, [])
            if cand_idx != idx and self.entries[cand_idx]["audio_path"] != current_audio
        ]

    def _load_reference_wav(self, idx: int, target_wav: torch.Tensor) -> tuple[torch.Tensor, str, bool]:
        candidates = self._candidate_ref_indices(idx)
        if candidates:
            random.shuffle(candidates)
            for cand_idx in candidates[:8]:
                ref_path = self.entries[cand_idx]["audio_path"]
                ref_wav = _read_audio_to_16k(
                    ref_path,
                    max_seconds=self.max_audio_duration,
                    truncate=False,
                )
                if ref_wav is None:
                    continue
                if ref_wav.numel() < int(self.min_audio_duration * TARGET_SR):
                    continue
                return ref_wav, ref_path, False

        return target_wav, self.entries[idx]["audio_path"], True

    def __getitem__(self, idx: int) -> Optional[JsonlSFTSample]:
        entry = self.entries[idx]
        try:
            tokens = self.tokenizer.tokenize(entry["text"])
            text_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        except Exception:
            return None

        if len(text_ids) == 0 or len(text_ids) > self.max_text_tokens:
            return None

        wav = _read_audio_to_16k(
            entry["audio_path"],
            max_seconds=self.max_audio_duration,
            truncate=False,
        )
        if wav is None or wav.numel() < int(self.min_audio_duration * TARGET_SR):
            return None
        ref_wav, ref_audio_path, ref_is_self = self._load_reference_wav(idx, wav)

        return JsonlSFTSample(
            text_ids=torch.tensor(text_ids, dtype=torch.long),
            wav_16k=wav,
            ref_wav_16k=ref_wav,
            audio_path=entry["audio_path"],
            ref_audio_path=ref_audio_path,
            ref_is_self=ref_is_self,
            emo_control_vector=entry["emo_control_vector"],
        )


def collate_sft_samples(batch: List[Optional[JsonlSFTSample]]) -> List[JsonlSFTSample]:
    return [x for x in batch if x is not None]


def prepare_sft_batch(
    samples: List[JsonlSFTSample],
    feature_extractor: FeaturePreprocessor,
    policy_model,
    device: torch.device,
    ref_dropout: float = 0.0,
    emo_dropout: float = 0.0,
) -> Optional[Dict[str, torch.Tensor]]:
    if not samples:
        return None

    target_wavs = [s.wav_16k for s in samples]
    ref_wavs = [s.ref_wav_16k for s in samples]
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
        ref_spk_cond_emb, ref_lengths = feature_extractor.extract_spk_cond_emb(ref_wavs)
        ref_feat_t = ref_spk_cond_emb.transpose(1, 2)
        conditioning = policy_model.get_conditioning(ref_feat_t, ref_lengths)
        emo_vec = policy_model.get_emovec(ref_spk_cond_emb, ref_lengths)

        target_spk_cond_emb, code_lengths = feature_extractor.extract_spk_cond_emb(target_wavs)
        codes, _ = feature_extractor.semantic_codec.quantize(target_spk_cond_emb.float())
        codes = codes.long()

        ref_drop_mask = None
        if ref_dropout > 0.0:
            self_ref_mask = torch.tensor(
                [s.ref_is_self for s in samples],
                dtype=torch.bool,
                device=device,
            )
            ref_drop_mask = (
                self_ref_mask
                & (torch.rand(conditioning.size(0), device=device) < ref_dropout)
            )
            if ref_drop_mask.any():
                conditioning = torch.where(
                    ref_drop_mask.view(-1, 1, 1),
                    torch.zeros_like(conditioning),
                    conditioning,
                )

        if any(s.emo_control_vector is not None for s in samples):
            controls = torch.stack(
                [
                    s.emo_control_vector
                    if s.emo_control_vector is not None
                    else torch.zeros(len(EMOTION_ORDER), dtype=torch.float32)
                    for s in samples
                ]
            ).to(device=device, dtype=torch.float32)
            styles = feature_extractor.extract_styles(ref_wavs)
            label_emovec = feature_extractor.build_label_emovec(styles, controls)
            residual_scale = 1.0 - controls.sum(dim=1, keepdim=True)
            emo_vec = label_emovec + residual_scale * emo_vec

        if emo_dropout > 0.0:
            emo_drop_mask = torch.rand(emo_vec.size(0), device=device) < emo_dropout
            if emo_drop_mask.any():
                emo_vec = torch.where(
                    emo_drop_mask.view(-1, 1),
                    torch.zeros_like(emo_vec),
                    emo_vec,
                )

    text_ids = pad_sequence(
        [s.text_ids for s in samples],
        batch_first=True,
        padding_value=0,
    )
    text_lengths = torch.tensor([int(s.text_ids.numel()) for s in samples], dtype=torch.long)

    return {
        "conditioning": conditioning.to(device=device, dtype=torch.float32),
        "emo_vec": emo_vec.to(device=device, dtype=torch.float32),
        "text_ids": text_ids.to(device=device),
        "text_lengths": text_lengths.to(device=device),
        "codes": codes.to(device=device),
        "code_lengths": code_lengths.to(device=device),
    }


class SFTLossWrapper(nn.Module):
    def __init__(
        self,
        model,
        use_duration_control: bool = False,
        duration_dropout: float = 0.3,
        use_emo_vec: bool = True,
    ):
        super().__init__()
        self.model = model
        self.use_duration_control = use_duration_control
        self.duration_dropout = duration_dropout
        self.use_emo_vec = use_emo_vec

    def forward(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["text_ids"].size(0)
        drop_mask = None
        if self.use_duration_control and self.duration_dropout > 0:
            drop_mask = (
                torch.rand(batch_size, device=batch["text_ids"].device)
                < self.duration_dropout
            )

        emo_vec = batch["emo_vec"]
        if not self.use_emo_vec:
            emo_vec = torch.zeros_like(emo_vec)

        per_token_logp, mel_mask, _ = gpt_per_token_logp(
            self.model,
            batch["conditioning"],
            emo_vec,
            batch["text_ids"],
            batch["text_lengths"],
            batch["codes"],
            batch["code_lengths"],
            use_duration_control=self.use_duration_control,
            duration_dropout_mask=drop_mask,
            return_entropy=False,
        )
        token_count = mel_mask.sum().clamp_min(1.0)
        mel_loss = -(per_token_logp * mel_mask).sum() / token_count
        return mel_loss, {"tokens": token_count.detach()}


def serializable_config(args: argparse.Namespace) -> Dict[str, Any]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def evaluate(
    model: SFTLossWrapper,
    loader: DataLoader,
    feature_extractor: FeaturePreprocessor,
    accelerator: Accelerator,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0.0
    policy_model = accelerator.unwrap_model(model).model

    for samples in loader:
        batch = prepare_sft_batch(samples, feature_extractor, policy_model, accelerator.device)
        if batch is None:
            continue
        with torch.no_grad():
            mel_loss, metrics = model(batch)
        count = metrics["tokens"].detach()
        stats = torch.stack([mel_loss.detach() * count, count])
        gathered = accelerator.gather(stats)
        total_loss += gathered[0::2].sum().item()
        total_count += gathered[1::2].sum().item()

    model.train()
    return {"mel_loss": total_loss / max(total_count, 1.0)}


def resolve_resume_path(output_dir: Path, resume: str) -> Optional[str]:
    if not resume:
        return None
    if resume != "auto":
        return resume
    dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not dirs:
        return None
    dirs.sort(key=lambda x: os.path.getmtime(x))
    return str(dirs[-1])


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accumulation,
        log_with=None if args.no_wandb else "wandb",
    )
    set_seed(args.seed)

    if accelerator.is_main_process and not args.no_wandb:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=serializable_config(args),
            init_kwargs={"wandb": {"name": args.wandb_run_name, "entity": args.wandb_entity}},
        )

    output_dir = args.output_dir.resolve()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer)
    accelerator.print("[Init] Building GPT ...")
    policy_model = build_unified_voice(
        args.config,
        tokenizer,
        args.base_checkpoint,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    summary = configure_policy_trainable(
        policy_model,
        train_scope=args.train_scope,
        gpt_train_mode=args.gpt_train_mode,
    )
    if accelerator.is_main_process:
        log_trainable_summary(policy_model, summary)

    model = SFTLossWrapper(
        policy_model,
        use_duration_control=args.use_duration_control,
        duration_dropout=args.duration_dropout,
        use_emo_vec=not args.no_emo_vec,
    )

    cfg = OmegaConf.load(args.config)
    feature_extractor = FeaturePreprocessor(
        args.model_dir,
        cfg,
        accelerator.device,
        dtype=torch.float32,
    )
    feature_extractor.eval()

    full_dataset = JsonlSFTDataset(
        metadata_path=args.metadata,
        audio_root=args.audio_root,
        tokenizer=tokenizer,
        audio_path_key=args.audio_path_key,
        text_key=args.text_key,
        speaker_key=args.speaker_key,
        language_filter=args.language_filter,
        min_audio_duration=args.min_audio_duration,
        max_audio_duration=args.max_audio_duration,
        max_text_tokens=args.max_text_tokens,
    )
    if len(full_dataset) == 0:
        raise RuntimeError("No valid SFT samples found.")

    val_size = min(max(args.val_data_size, 0), max(len(full_dataset) - 1, 0))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )
    accelerator.print(f"[Data] Total: {len(full_dataset)} -> Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_device,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_sft_samples,
        pin_memory=True,
    )
    val_loader = None
    if val_size > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size_per_device,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_sft_samples,
            pin_memory=True,
        )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accumulation)
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * num_update_steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    if val_loader is None:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
    else:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )

    global_step = 0
    start_epoch = 0
    resume_batches_to_skip = 0
    resume_path = resolve_resume_path(output_dir, args.resume)
    if resume_path and os.path.exists(resume_path):
        accelerator.print(f"[Info] Resuming from {resume_path}")
        accelerator.load_state(resume_path)
        try:
            global_step = int(Path(resume_path).name.split("-")[-1])
            start_epoch = global_step // max(num_update_steps_per_epoch, 1)
            resume_step_in_epoch = global_step % max(num_update_steps_per_epoch, 1)
            resume_batches_to_skip = resume_step_in_epoch * args.grad_accumulation
        except ValueError:
            pass
    elif args.resume:
        accelerator.print(f"[Warn] Resume path {args.resume} not found. Starting from scratch.")

    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_main_process)
    progress_bar.update(global_step)
    model.train()

    for epoch in range(start_epoch, args.epochs):
        active_train_loader = train_loader
        skipped_batches = 0
        if epoch == start_epoch and resume_batches_to_skip > 0:
            active_train_loader = accelerator.skip_first_batches(train_loader, resume_batches_to_skip)
            skipped_batches = resume_batches_to_skip

        for batch_idx, samples in enumerate(active_train_loader):
            logical_batch_idx = batch_idx + skipped_batches
            policy_for_prep = accelerator.unwrap_model(model).model
            batch = prepare_sft_batch(
                samples,
                feature_extractor,
                policy_for_prep,
                accelerator.device,
                ref_dropout=args.ref_dropout,
                emo_dropout=args.emo_dropout,
            )
            if batch is None:
                continue

            with accelerator.accumulate(model):
                mel_loss, metrics = model(batch)
                accelerator.backward(mel_loss)

                if accelerator.sync_gradients:
                    if args.grad_clip > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if global_step % args.log_interval == 0:
                    lr = scheduler.get_last_lr()[0]
                    accelerator.log(
                        {
                            "train/mel_loss": mel_loss.item(),
                            "train/tokens": metrics["tokens"].item(),
                            "train/lr": lr,
                            "train/epoch": epoch + logical_batch_idx / max(len(train_loader), 1),
                        },
                        step=global_step,
                    )
                    accelerator.print(
                        f"[Train] epoch={epoch + 1} step={global_step} "
                        f"loss={mel_loss.item():.4f} lr={lr:.2e}"
                    )

                if (
                    val_loader is not None
                    and args.val_interval > 0
                    and global_step % args.val_interval == 0
                ):
                    val_metrics = evaluate(model, val_loader, feature_extractor, accelerator)
                    accelerator.log({"val/mel_loss": val_metrics["mel_loss"]}, step=global_step)
                    accelerator.print(
                        f"[Val] epoch={epoch + 1} step={global_step} "
                        f"mel_loss={val_metrics['mel_loss']:.4f}"
                    )

                is_regular_save = args.save_every > 0 and global_step % args.save_every == 0
                is_major_save = (
                    args.major_save_every > 0 and global_step % args.major_save_every == 0
                )
                if is_regular_save or is_major_save:
                    accelerator.wait_for_everyone()
                    save_path = output_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(save_path)
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(model)
                        torch.save(
                            {"model": unwrapped.model.state_dict()},
                            output_dir / f"model_step{global_step}.pth",
                        )
                        print(f"[Checkpoint] Saved checkpoint to {save_path}")
                        rotate_checkpoints(
                            output_dir,
                            keep_last=args.keep_last,
                            major_save_every=args.major_save_every,
                            is_main=True,
                        )

            if args.max_steps and global_step >= args.max_steps:
                break

        if args.max_steps and global_step >= args.max_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = output_dir / "checkpoint-final"
        accelerator.save_state(save_path)
        unwrapped = accelerator.unwrap_model(model)
        torch.save({"model": unwrapped.model.state_dict()}, output_dir / "model_final.pth")

    accelerator.end_training()
    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
