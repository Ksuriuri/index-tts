import argparse
import json
import math
import os
import random
import datetime
from datasets import load_from_disk, concatenate_datasets
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple
from loguru import logger
from tqdm.auto import tqdm
import shutil

# 原始路径设置
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf

# 新增库
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import wandb

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.front import TextNormalizer, TextTokenizer
from trainers.utils import ProcessedData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune IndexTTS2 GPT on Japanese data.")
    parser.add_argument(
        "--train-data-dirs", 
        type=str, 
        nargs='+',  # 接收一个或多个值
        required=True, 
        help="Path to the arrow dataset directories (containing part_*), can specify multiple"
    )
    parser.add_argument("--val-data-size", type=int, default=128, help="Validation data size.")
    parser.add_argument("--tokenizer", type=Path, default=Path("checkpoints/IndexTTS-2-vLLM/jp_bpe.model"), help="SentencePiece model path.")
    parser.add_argument("--config", type=Path, default=Path("checkpoints/IndexTTS-2-vLLM/config.yaml"), help="Model config YAML.")
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/IndexTTS-2-vLLM/gpt.pth"), help="Base GPT checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("trained_ckpts"), help="Directory for checkpoints/logs.")
    parser.add_argument("--batch-size-per-device", type=int, default=4, help="batch size per device.")
    parser.add_argument("--grad-accumulation", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="LR warmup steps.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional max optimiser steps (0 = unlimited).")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between training log entries.")
    parser.add_argument("--val-interval", type=int, default=100, help="Validation frequency in steps.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--keep-last", type=int, default=2, help="Keep last N checkpoints.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping value.")
    parser.add_argument("--save_every", type=int, default=1000, help="save checkpoint every N steps")
    parser.add_argument("--major-save-every", type=int, default=25000, help="Save a permanent checkpoint every N steps (never deleted).")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint directory to resume from (accelerate style), or empty.")
    parser.add_argument("--use-duration-control", action="store_true", help="Train GPT with duration embeddings.")
    parser.add_argument("--duration-dropout", type=float, default=0.3, help="Probability of zeroing duration embeddings.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    # WandB configs
    parser.add_argument("--wandb-project", type=str, default="indextts-finetune", help="WandB project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity (team/user)")

    return parser.parse_args()


class ArrowJapaneseGPTDataset(Dataset):
    def __init__(self, arrow_root_dirs: List[str]):
        """
        Args:
            arrow_root_dir: 包含 xxx_part_0, xxx_part_1... 的根目录路径
        """
        all_shard_paths = []
        for arrow_root_dir in arrow_root_dirs:
            arrow_root_dir = Path(arrow_root_dir)
            
            # 1. 扫描所有分片目录
            print(f"[Dataset] Scanning shards in {arrow_root_dir} ...")
            shard_paths = sorted([
                d for d in arrow_root_dir.iterdir() 
                if d.is_dir() and "_part_" in d.name
            ], key=lambda x: int(x.name.split("_")[-1])) # 按 part_后面的数字排序

            if not shard_paths:
                raise ValueError(f"No 'part_*' directories found in {arrow_root_dir}")
            
            all_shard_paths.extend(shard_paths)

        # 2. 加载所有分片 (load_from_disk 是懒加载，速度很快)
        # 注意：这里加载的是 Dataset 对象，还没有读取具体数据
        datasets = [load_from_disk(str(p)) for p in all_shard_paths]
        
        # 3. 逻辑合并 (零拷贝，仅仅是索引合并)
        self.dataset = concatenate_datasets(datasets)
        
        print(f"[Dataset] Loaded {len(datasets)} shards. Total samples: {len(self.dataset)}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Arrow dataset 返回的是字典，里面的值通常是 Python list 或 numpy array
        item = self.dataset[idx]
        
        return {
            "text_ids": torch.tensor(item["text_ids"], dtype=torch.long),
            "codes": torch.tensor(item["codes"], dtype=torch.long),
            "condition": torch.tensor(item["condition"], dtype=torch.float32),
            "emo_vec": torch.tensor(item["emo_vec"], dtype=torch.float32),
            "text_len": torch.tensor(item["text_len"], dtype=torch.long),
            "code_len": torch.tensor(item["code_len"], dtype=torch.long),
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    text_tensors = [item["text_ids"] for item in batch]
    code_tensors = [item["codes"] for item in batch]
    condition_tensors = [item["condition"] for item in batch]
    emo_tensors = [item["emo_vec"] for item in batch]

    text_padded = pad_sequence(text_tensors, batch_first=True, padding_value=0)
    code_padded = pad_sequence(code_tensors, batch_first=True, padding_value=0)
    condition_stacked = torch.stack(condition_tensors, dim=0)
    emo_stacked = torch.stack(emo_tensors, dim=0)

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


class GPTLossWrapper(nn.Module):
    def __init__(self, model: UnifiedVoice, use_duration_control: bool = False, duration_dropout: float = 0.3):
        super().__init__()
        self.model = model
        self.use_duration_control = use_duration_control
        self.duration_dropout = duration_dropout

    def forward(self, batch: Dict[str, torch.Tensor]):
        target_device = batch["text_ids"].device

        condition = batch["condition"]
        text_ids = batch["text_ids"]
        codes = batch["codes"]
        emo_vec = batch["emo_vec"]
        text_lengths = batch["text_lengths"]
        code_lengths = batch["code_lengths"]

        batch_size = text_ids.size(0)
        use_speed = torch.zeros(batch_size, dtype=torch.long, device=target_device)

        text_inputs = self.model.set_text_padding(text_ids.clone(), text_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.model.stop_text_token)
        text_inputs, text_targets = self.model.build_aligned_inputs_and_targets(
            text_inputs, self.model.start_text_token, self.model.stop_text_token
        )

        mel_inputs = self.model.set_mel_padding(codes.clone(), code_lengths)
        mel_inputs = F.pad(mel_inputs, (0, 1), value=self.model.stop_mel_token)
        mel_inputs, mel_targets = self.model.build_aligned_inputs_and_targets(
            mel_inputs, self.model.start_mel_token, self.model.stop_mel_token
        )

        duration_free = self.model.speed_emb(torch.zeros_like(use_speed))
        if self.use_duration_control:
            duration_ctrl = self.model.get_duration_embeddings(code_lengths)
            if self.duration_dropout > 0.0:
                drop_mask = torch.rand(code_lengths.size(0), device=target_device) < self.duration_dropout
                if drop_mask.any():
                    duration_ctrl = torch.where(drop_mask.unsqueeze(1), duration_free, duration_ctrl)
        else:
            duration_ctrl = self.model.speed_emb(torch.ones_like(use_speed))
        conds = torch.cat(
            (condition + emo_vec.unsqueeze(1), duration_ctrl.unsqueeze(1), duration_free.unsqueeze(1)),
            dim=1,
        )

        text_emb = self.model.text_embedding(text_inputs) + self.model.text_pos_embedding(text_inputs)
        mel_emb = self.model.mel_embedding(mel_inputs) + self.model.mel_pos_embedding(mel_inputs)

        text_logits, mel_logits = self.model.get_logits(conds, text_emb, self.model.text_head, mel_emb, self.model.mel_head)

        mel_mask = (
            torch.arange(mel_targets.size(1), device=target_device).unsqueeze(0)
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
                top1 = (valid_logits.argmax(dim=-1) == valid_targets).float().mean()
            else:
                top1 = torch.tensor(0.0, device=target_device)
            metrics["mel_top1"] = top1

        return mel_loss, metrics


def load_tokenizer(tokenizer_path: Path) -> TextTokenizer:
    normalizer = TextNormalizer()
    normalizer.load()
    tokenizer = TextTokenizer(str(tokenizer_path), normalizer)
    return tokenizer


def build_model(cfg_path: Path, tokenizer: TextTokenizer, base_checkpoint: Path) -> UnifiedVoice:
    cfg = OmegaConf.load(cfg_path)
    vocab_size = tokenizer.vocab_size
    if cfg.gpt.number_text_tokens != vocab_size:
        cfg.gpt.number_text_tokens = vocab_size

    model = UnifiedVoice(**cfg.gpt, checkpointing=False)
    
    # 只有主进程打印加载信息，或者让它在所有进程加载（但通常 checkpoint 是只读的，并发读没问题）
    print(f"Loading checkpoint from {base_checkpoint}")
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

    # Resize embeddings logic
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

    return model # accelerator will handle .to(device)


def evaluate(
    model: UnifiedVoice,
    loader: DataLoader,
    accelerator: Accelerator,
) -> Dict[str, float]:
    model.eval()
    totals = {"mel_loss": 0.0, "mel_top1": 0.0}
    count = 0
    
    for batch in loader:
        with torch.no_grad():
            mel_loss, metrics = model(batch)
            
            # 使用 gather 收集多卡上的结果以计算准确的平均值
            # 注意：loss 已经是 scalar，metrics['mel_top1'] 也是 scalar tensor
            gathered_loss = accelerator.gather(mel_loss.repeat(batch["text_ids"].size(0)))
            gathered_top1 = accelerator.gather(metrics["mel_top1"].repeat(batch["text_ids"].size(0)))
            
            # gather 后得到的 tensor 长度为 batch_size * num_processes
            totals["mel_loss"] += gathered_loss.sum().item()
            totals["mel_top1"] += gathered_top1.sum().item()
            count += gathered_loss.numel()

    model.train()
    
    if count == 0:
        return {k: 0.0 for k in totals}
    
    return {k: v / count for k, v in totals.items()}


def rotate_checkpoints(output_dir: Path, keep_last: int, major_save_every: int, accelerator: Accelerator):
    """
    保留最新的 keep_last 个检查点。
    - 忽略（不删除）能被 major_save_every 整除的节点。
    - 删除旧节点时，同时删除 'checkpoint-XXX' 文件夹和 'model_stepXXX.pth' 文件。
    """
    # 只在主进程执行删除操作
    if not accelerator.is_main_process:
        return

    # 1. 找到所有 checkpoint-X 文件夹
    glob_checkpoints = list(output_dir.glob("checkpoint-*"))
    checkpoints = []
    
    for path in glob_checkpoints:
        if not path.is_dir():
            continue
        try:
            # 解析步数，例如 checkpoint-2000 -> 2000
            step = int(path.name.split("-")[-1])
            checkpoints.append((step, path))
        except ValueError:
            continue

    # 按步数从小到大排序
    checkpoints.sort(key=lambda x: x[0])

    # 2. 分离 "普通节点" 和 "重大节点"
    regular_checkpoints = []
    
    for step, path in checkpoints:
        if major_save_every > 0 and step % major_save_every == 0:
            # 这是一个重大节点 (比如 25000)，跳过，不放入待删除列表
            continue
        else:
            regular_checkpoints.append((step, path))

    # 3. 如果普通节点超过了 keep_last，删除最旧的
    if len(regular_checkpoints) > keep_last:
        # 计算需要删除的数量
        num_to_delete = len(regular_checkpoints) - keep_last
        # 获取要删除的列表（前 num_to_delete 个就是最旧的）
        to_delete = regular_checkpoints[:num_to_delete]
        
        for step, folder_path in to_delete:
            print(f"[Checkpoint] Rotate: Deleting old step {step}...")
            
            # --- 删除文件夹 checkpoint-XXX ---
            try:
                shutil.rmtree(folder_path)
                print(f"  - Deleted dir: {folder_path.name}")
            except OSError as e:
                print(f"  - [Error] Failed to delete dir {folder_path}: {e}")

            # --- 删除对应的 .pth 文件 model_stepXXX.pth ---
            pth_path = output_dir / f"model_step{step}.pth"
            if pth_path.exists():
                try:
                    os.remove(pth_path) # 或者 pth_path.unlink()
                    print(f"  - Deleted file: {pth_path.name}")
                except OSError as e:
                    print(f"  - [Error] Failed to delete file {pth_path}: {e}")


def main() -> None:
    args = parse_args()

    # 1. Initialize Accelerator
    # log_with="wandb" 会自动处理 wandb.init
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accumulation,
        log_with="wandb",
        # kwargs_handlers=[ddp_kwargs]
    )
    
    # 2. Set seed
    set_seed(args.seed)

    # 3. WandB Setup (via Accelerator)
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.wandb_run_name, "entity": args.wandb_entity}}
        )

    output_dir = args.output_dir.resolve()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Load Data & Model
    tokenizer = load_tokenizer(args.tokenizer)
    model = build_model(args.config, tokenizer, args.base_checkpoint)

    for param in model.parameters():
        param.requires_grad = True

    freeze_modules = [
        model.conditioning_encoder,
        model.perceiver_encoder,
        model.emo_conditioning_encoder,
        model.emo_perceiver_encoder,
        model.emo_layer,
        model.emovec_layer,
        model.speed_emb,
        model.text_head,
    ]

    if accelerator.is_main_process:
        print("Freezing the following modules:")

    for module in freeze_modules:
        if accelerator.is_main_process:
            print(f" - Freezing {module.__class__.__name__}")
        for param in module.parameters():
            param.requires_grad = False

    model = GPTLossWrapper(
        model, 
        use_duration_control=args.use_duration_control,
        duration_dropout=args.duration_dropout
    )

    full_dataset = ArrowJapaneseGPTDataset(args.train_data_dirs)
    total_size = len(full_dataset)
    train_size = total_size - args.val_data_size
    
    # 使用固定种子进行切分，确保每次运行验证集都是同一批数据
    # 即使 args.seed 不同，我们也希望数据划分相对稳定，或者你可以直接用 args.seed
    generator = torch.Generator().manual_seed(args.seed)
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, args.val_data_size], 
        generator=generator
    )
    
    if accelerator.is_main_process:
        print(f"[Data] Total: {total_size} -> Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Accelerate 会自动处理 DataLoader 的 sampler (分布式切分)，这里通常不需要设置 shuffle (虽然设置了也没事)
    # 也不需要 pin_memory=True，Accelerate 会优化
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_device,
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_device,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True
    )

    # optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # total_steps = args.max_steps if args.max_steps > 0 else args.epochs * len(train_loader) // max(1, args.grad_accumulation)
    num_batches_per_epoch = len(train_loader) // accelerator.num_processes
    num_update_steps_per_epoch = math.ceil(num_batches_per_epoch / args.grad_accumulation)
    total_steps = args.epochs * num_update_steps_per_epoch

    if accelerator.is_main_process:
        print(f"Total training steps: {total_steps} (Steps per epoch: {num_update_steps_per_epoch})")
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    # 5. Prepare with Accelerator
    # 这一步会自动处理 device placement, DDP wrapping, mixed precision 等
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # 6. Resume from checkpoint (if needed)
    global_step = 0
    start_epoch = 0
    if args.resume:
        if args.resume == "auto":
            # 简单的自动查找逻辑
            dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
            if dirs:
                # 找最新的
                dirs.sort(key=lambda x: os.path.getmtime(x))
                resume_path = str(dirs[-1])
            else:
                resume_path = None
        else:
            resume_path = args.resume

        if resume_path and os.path.exists(resume_path):
            accelerator.print(f"[Info] Resuming from {resume_path}")
            accelerator.load_state(resume_path)
            # 计算 global_step 和 start_epoch 的大致位置 (非精确，仅用于显示)
            # 如果要精确控制，需要单独保存 step 到文件或从路径名解析
            try:
                # 假设文件夹名字是 checkpoint-{step}
                step_val = int(Path(resume_path).name.split("-")[-1])
                global_step = step_val
                start_epoch = global_step // (len(train_loader) // args.grad_accumulation)
            except:
                pass
        else:
            accelerator.print(f"[Warn] Resume path {args.resume} not found or invalid. Starting from scratch.")

    model.train()
    
    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_main_process)
    progress_bar.update(global_step)

    for epoch in range(start_epoch, args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            # Gradient Accumulation Context
            with accelerator.accumulate(model):
                mel_loss, metrics = model(batch)
                
                loss = mel_loss
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    if args.grad_clip > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # 只有在梯度更新发生后才增加 global_step 和记录日志
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if global_step % args.log_interval == 0:
                    lr = scheduler.get_last_lr()[0]
                    # WandB logging
                    accelerator.log(
                        {
                            "train/mel_loss": mel_loss.item(),
                            "train/mel_top1": metrics["mel_top1"].item(),
                            "train/lr": lr,
                            "train/epoch": epoch + (batch_idx / len(train_loader)),
                        },
                        step=global_step,
                    )
                    
                    accelerator.print(
                        f"[Train] epoch={epoch + 1} step={global_step} "
                        f"loss={mel_loss.item():.4f} top1={metrics['mel_top1']:.4f} lr={lr:.2e}"
                    )

                # Validation
                if args.val_interval > 0 and global_step > 0 and global_step % args.val_interval == 0:
                    val_metrics = evaluate(
                        model,
                        val_loader,
                        accelerator,
                    )
                    accelerator.log(
                        {
                            "val/mel_loss": val_metrics["mel_loss"],
                            "val/mel_top1": val_metrics["mel_top1"],
                        },
                        step=global_step,
                    )
                    accelerator.print(
                        f"[Val] epoch={epoch + 1} step={global_step} "
                        f"mel_loss={val_metrics['mel_loss']:.4f} top1={val_metrics['mel_top1']:.4f}"
                    )

                # Saving Checkpoint
                is_regular_save = (global_step % args.save_every == 0)
                is_major_save = (args.major_save_every > 0 and global_step % args.major_save_every == 0)
                if (is_regular_save or is_major_save) and global_step > 0:
                    # 确保所有进程同步
                    accelerator.wait_for_everyone()
                    
                    # 1. 保存完整状态 (checkpoint-STEP)
                    save_path = output_dir / f"checkpoint-{global_step}"
                    # accelerate 的 save_state 会自动处理主进程写文件
                    accelerator.save_state(save_path)
                    
                    # 2. 保存纯权重 & 执行轮换清理 (只在主进程)
                    if accelerator.is_main_process:
                        # 保存 .pth 模型文件
                        unwrapped_wrapper = accelerator.unwrap_model(model)
                        real_model = unwrapped_wrapper.model 
                        weight_path = output_dir / f"model_step{global_step}.pth"
                        torch.save({"model": real_model.state_dict()}, weight_path)
                        print(f"[Checkpoint] Saved checkpoint to {save_path}")

                        # 执行轮换：保留4个最新的，除非是25000的倍数
                        rotate_checkpoints(
                            output_dir=output_dir,
                            keep_last=args.keep_last,
                            major_save_every=args.major_save_every,
                            accelerator=accelerator
                        )
                    
            if args.max_steps and global_step >= args.max_steps:
                break
        
        if args.max_steps and global_step >= args.max_steps:
            break

    # Final Save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = output_dir / f"checkpoint-final"
        accelerator.save_state(save_path)
        
        unwrapped_wrapper = accelerator.unwrap_model(model)
        real_model = unwrapped_wrapper.model
        torch.save({"model": real_model.state_dict()}, output_dir / "model_final.pth")
    
    accelerator.end_training()
    print("Training complete.")


if __name__ == "__main__":
    main()