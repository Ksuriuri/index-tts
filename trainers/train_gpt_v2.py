import argparse
import json
import os
import pickle
import random
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.front import TextNormalizer, TextTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune IndexTTS2 GPT on Japanese data.")
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--val-data-path",
        type=str,
        required=True,
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("checkpoints/IndexTTS-2-vLLM/jp_bpe.model"), help="SentencePiece model path.")
    parser.add_argument("--config", type=Path, default=Path("checkpoints/IndexTTS-2-vLLM/config.yaml"), help="Model config YAML.")
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/IndexTTS-2-vLLM/gpt.pth"), help="Base GPT checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("trained_ckpts"), help="Directory for checkpoints/logs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size per optimisation step.")
    parser.add_argument("--grad-accumulation", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="LR warmup steps.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional max optimiser steps (0 = unlimited).")
    parser.add_argument("--log-interval", type=int, default=100, help="Steps between training log entries.")
    parser.add_argument("--val-interval", type=int, default=100, help="Validation frequency in steps.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping value.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP.")
    parser.add_argument("--save_every", type=int, default=1000, help="save checkpoint every N steps")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from, or 'auto'.")
    parser.add_argument(
        "--use-duration-control",
        action="store_true",
        help="Train GPT with duration embeddings derived from target semantic lengths.",
    )
    parser.add_argument(
        "--duration-dropout",
        type=float,
        default=0.3,
        help="Probability of zeroing duration embeddings when --use-duration-control is enabled.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class Sample:
    text_ids: torch.Tensor  # torch.int32, [text_len]
    codes: torch.Tensor  # torch.int32, [code_len]
    text_len: int
    code_len: int
    condition: torch.Tensor  # torch.float32, [32, dim]
    emo_vec: torch.Tensor  # torch.float32, [1, dim]


class JapaneseGPTDataset(Dataset):
    def __init__(self, data_path: str):
        self.samples: List[Sample] = []
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Pickle file not found at: {data_path}")
            
        print(f"Loading dataset from {data_path} ...")
        try:
            with open(data_path, 'rb') as f:
                # 加载之前保存的 List[Sample]
                self.samples = pickle.load(f)
            print(f"Loaded {len(self.samples)} samples successfully.")
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            raise e

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "text_ids": sample.text_ids,
            "codes": sample.codes.squeeze(0),
            "condition": sample.condition,  # [cond_len, dim]
            "emo_vec": sample.emo_vec,
            "text_len": torch.tensor(sample.text_len, dtype=torch.long),
            "code_len": torch.tensor(sample.code_len, dtype=torch.long),
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    text_tensors = [item["text_ids"] for item in batch]
    code_tensors = [item["codes"] for item in batch]
    condition_tensors = [item["condition"] for item in batch]
    emo_tensors = [item["emo_vec"] for item in batch]

    text_padded = pad_sequence(text_tensors, batch_first=True, padding_value=0)
    code_padded = pad_sequence(code_tensors, batch_first=True, padding_value=0)
    condition_stacked = torch.cat(condition_tensors, dim=0)
    emo_stacked = torch.cat(emo_tensors, dim=0)

    text_lengths = torch.stack([item["text_len"] for item in batch])
    code_lengths = torch.stack([item["code_len"] for item in batch])

    return {
        "text_ids": text_padded,
        "codes": code_padded,
        "condition": condition_stacked,
        "emo_vec": emo_stacked,
        "text_lengths": text_lengths,
        "code_lengths": code_lengths,
    }


def compute_losses(
    model: UnifiedVoice,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_duration_control: bool = False,
    duration_dropout: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    condition = batch["condition"].to(device)
    text_ids = batch["text_ids"].to(device)
    codes = batch["codes"].to(device)
    emo_vec = batch["emo_vec"].to(device)
    text_lengths = batch["text_lengths"].to(device)
    code_lengths = batch["code_lengths"].to(device)

    batch_size = text_ids.size(0)
    use_speed = torch.zeros(batch_size, dtype=torch.long, device=device)

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
        if duration_dropout > 0.0:
            drop_mask = torch.rand(code_lengths.size(0), device=device) < duration_dropout
            if drop_mask.any():
                duration_ctrl = torch.where(drop_mask.unsqueeze(1), duration_free, duration_ctrl)
    else:
        duration_ctrl = model.speed_emb(torch.ones_like(use_speed))
    conds = torch.cat(
        (condition + emo_vec.unsqueeze(1), duration_ctrl.unsqueeze(1), duration_free.unsqueeze(1)),
        dim=1,
    )

    text_emb = model.text_embedding(text_inputs) + model.text_pos_embedding(text_inputs)
    mel_emb = model.mel_embedding(mel_inputs) + model.mel_pos_embedding(mel_inputs)

    text_logits, mel_logits = model.get_logits(conds, text_emb, model.text_head, mel_emb, model.mel_head)

    mel_mask = (
        torch.arange(mel_targets.size(1), device=device).unsqueeze(0)
        < (code_lengths + 1).unsqueeze(1)
    )

    mel_ce = F.cross_entropy(mel_logits, mel_targets, reduction="none")

    mel_loss = (mel_ce * mel_mask).sum() / mel_mask.sum().clamp_min(1)

    metrics = {}
    with torch.no_grad():
        mel_logits_flat = mel_logits.permute(0, 2, 1).reshape(-1, mel_logits.size(1))
        mel_targets_flat = mel_targets.reshape(-1)
        mel_mask_flat = mel_mask.reshape(-1)
        if mel_mask_flat.any():
            valid_logits = mel_logits_flat[mel_mask_flat]
            valid_targets = mel_targets_flat[mel_mask_flat]
            top1 = (valid_logits.argmax(dim=-1) == valid_targets).float().mean().item()
        else:
            top1 = 0.0
        metrics["mel_top1"] = top1

    return mel_loss, metrics


def load_tokenizer(tokenizer_path: Path) -> TextTokenizer:
    normalizer = TextNormalizer()
    normalizer.load()
    tokenizer = TextTokenizer(str(tokenizer_path), normalizer)
    return tokenizer


def build_model(cfg_path: Path, tokenizer: TextTokenizer, base_checkpoint: Path, device: torch.device) -> UnifiedVoice:
    cfg = OmegaConf.load(cfg_path)
    vocab_size = tokenizer.vocab_size
    if cfg.gpt.number_text_tokens != vocab_size:
        cfg.gpt.number_text_tokens = vocab_size

    model = UnifiedVoice(**cfg.gpt)
    checkpoint = torch.load(base_checkpoint, map_location="cpu")
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
    if missing:
        print(f"[Warn] Missing keys during load: {missing}")
    if unexpected:
        print(f"[Warn] Unexpected keys during load: {unexpected}")

    return model.to(device)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "step": step,
    }
    torch.save(state, path)


def evaluate(
    model: UnifiedVoice,
    loader: DataLoader,
    device: torch.device,
    use_duration_control: bool = False,
    duration_dropout: float = 0.3,
) -> Dict[str, float]:
    model.eval()
    totals = {"mel_loss": 0.0, "mel_top1": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            mel_loss, metrics = compute_losses(
                model,
                batch,
                device,
                use_duration_control=use_duration_control,
                duration_dropout=duration_dropout,
            )
            bsz = batch["text_ids"].size(0)
            totals["mel_loss"] += mel_loss.item() * bsz
            totals["mel_top1"] += metrics["mel_top1"] * bsz
            count += bsz
    model.train()
    if count == 0:
        return {k: 0.0 for k in totals}
    return {k: v / count for k, v in totals.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_root = output_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if os.environ.get("INDEXTTS_RUN_NAME") is None
        else os.environ["INDEXTTS_RUN_NAME"]
    )
    log_dir = log_root / run_name
    writer = SummaryWriter(log_dir=str(log_dir))

    tokenizer = load_tokenizer(args.tokenizer)
    model = build_model(args.config, tokenizer, args.base_checkpoint, device)

    train_dataset = JapaneseGPTDataset(args.train_data_path)
    val_dataset = JapaneseGPTDataset(args.val_data_path)

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=use_cuda,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * max(1, len(train_loader)) // max(1, args.grad_accumulation)
    total_steps = max(total_steps, 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    global_step = 0
    start_epoch = 0
    last_saved_step: int | None = None

    if args.resume:
        resume_path = args.resume
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("step", 0)
        last_saved_step = checkpoint.get("step")
        print(f"[Info] Resumed from {resume_path} at epoch {start_epoch}, step {global_step}.")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=use_amp):
                mel_loss, metrics = compute_losses(
                    model,
                    batch,
                    device,
                    use_duration_control=args.use_duration_control,
                    duration_dropout=args.duration_dropout,
                )
                loss = mel_loss
            if use_amp:
                scaler.scale(loss / args.grad_accumulation).backward()
            else:
                (loss / args.grad_accumulation).backward()

            if (batch_idx + 1) % args.grad_accumulation == 0:
                if args.grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % args.log_interval == 0:
                    writer.add_scalar("train/mel_loss", mel_loss.item(), global_step)
                    writer.add_scalar("train/mel_top1", metrics["mel_top1"], global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    print(
                        f"[Train] epoch={epoch + 1} step={global_step} "
                        f"mel_top1={metrics['mel_top1']:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                    )

                if args.val_interval > 0 and global_step > 0 and global_step % args.val_interval == 0:
                    val_metrics = evaluate(
                        model,
                        val_loader,
                        device,
                        use_duration_control=args.use_duration_control,
                        duration_dropout=args.duration_dropout,
                    )
                    writer.add_scalar("val/text_loss", val_metrics["text_loss"], global_step)
                    writer.add_scalar("val/mel_loss", val_metrics["mel_loss"], global_step)
                    writer.add_scalar("val/mel_top1", val_metrics["mel_top1"], global_step)
                    print(
                        f"[Val] epoch={epoch + 1} step={global_step} "
                        f"text_loss={val_metrics['text_loss']:.4f} mel_loss={val_metrics['mel_loss']:.4f} "
                        f"mel_top1={val_metrics['mel_top1']:.4f}"
                    )

                if global_step % args.save_every == 0:
                    ckpt_path = output_dir / f"model_step{global_step}.pth"
                    save_checkpoint(
                        ckpt_path,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        global_step,
                    )
                    last_saved_step = global_step

            if args.max_steps and global_step >= args.max_steps:
                break

        if args.max_steps and global_step >= args.max_steps:
            break


    if global_step > 0 and last_saved_step != global_step:
        ckpt_path = output_dir / f"model_step{global_step}.pth"
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
        )

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()