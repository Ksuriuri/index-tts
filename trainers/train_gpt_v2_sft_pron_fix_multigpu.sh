#!/usr/bin/env bash
set -e

accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision fp16 \
    trainers/train_gpt_v2_sft_jsonl_multigpu.py \
    --config checkpoints/IndexTTS-2-vLLM/config.yaml \
    --tokenizer checkpoints/IndexTTS-2-vLLM/bpe.model \
    --base-checkpoint checkpoints/IndexTTS-2-vLLM/gpt.pth \
    --model-dir checkpoints/IndexTTS-2-vLLM \
    --metadata /mnt/data_3t_1/datasets/raw_data/pron_fix/metadata_train.jsonl \
    --audio-root /mnt/data_3t_1/datasets/raw_data/pron_fix \
    --language-filter zh \
    --min-audio-duration 0.5 \
    --max-audio-duration 36 \
    --batch-size-per-device 2 \
    --grad-accumulation 1 \
    --gradient-checkpointing \
    --num-workers 2 \
    --epochs 3 \
    --learning-rate 4e-5 \
    --weight-decay 0.01 \
    --warmup-steps 50 \
    --ref-dropout 0.1 \
    --log-interval 10 \
    --val-interval 200 \
    --save-every 2500 \
    --major-save-every 25000 \
    --keep-last 4 \
    --output-dir ./trained_ckpts/sft_pron_fix \
    --resume auto \
    --wandb-project "IndexTTS2-SFT" \
    --wandb-run-name "pron_fix_sft"
    # --use-duration-control --duration-dropout 0.3 \
    # --no-emo-vec \
