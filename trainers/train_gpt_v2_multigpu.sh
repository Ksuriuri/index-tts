accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision fp16 \
    trainers/train_gpt_v2_multigpu.py \
    --train-data-path /mnt/data_3t_2/datasets/indextts_train_data/Galgame-VisualNovel-Reupload \
    --val-data-size 128 \
    --data-cache-size 512 \
    --output-dir ./trained_ckpts \
    --batch-size-per-device 3 \
    --grad-accumulation 1 \
    --epochs 1 \
    --learning-rate 2e-5 \
    --log-interval 10 \
    --val-interval 100 \
    --save_every 1000 \
    --use-duration-control \
    --duration-dropout 0.3 \
    --wandb-project "IndexTTS2-jp" \
    --wandb-run-name "20251210-test"