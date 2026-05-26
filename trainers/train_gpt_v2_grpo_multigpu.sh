#!/usr/bin/env bash
set -e

accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision fp16 \
    trainers/train_gpt_v2_grpo_multigpu.py \
    --config checkpoints/IndexTTS-2-vLLM/config.yaml \
    --tokenizer checkpoints/IndexTTS-2-vLLM/bpe.model \
    --base-checkpoint checkpoints/IndexTTS-2-vLLM/gpt.pth \
    --model-dir checkpoints/IndexTTS-2-vLLM \
    --metadata /mnt/data_3t_1/datasets/raw_data/noiz-v2/multigen/metadata_v2_merged_04-05.jsonl \
    --audio-root /mnt/data_3t_1/datasets/raw_data/noiz-v2/multigen \
    --ref-audio-root /mnt/data_3t_1/datasets/raw_data/noiz-v2/ref_audios \
    --ref-audio-suffix .flac \
    --max-group-size 8 \
    --max-samples-per-batch 16 \
    --max-audio-duration 36 \
    --max-ref-duration 36 \
    --groups-per-device 1 \
    --grad-accumulation 1 \
    --gradient-checkpointing \
    --num-workers 2 \
    --epochs 3 \
    --learning-rate 2e-6 \
    --weight-decay 0.0 \
    --warmup-steps 50 \
    --clip-eps 0.2 \
    --kl-coeff 0.04 \
    --kl-estimator k3 \
    --adv-norm global_batch \
    --entropy-coeff 0.0 \
    --log-interval 10 \
    --save-every 1000 \
    --major-save-every 10000 \
    --keep-last 2 \
    --output-dir ./trained_ckpts/grpo \
    --wandb-project "IndexTTS2-GRPO" \
    --wandb-run-name "noiz_v2_multigen_grpo"
    # --use-duration-control --duration-dropout 0.3 \
    # --resume auto \
