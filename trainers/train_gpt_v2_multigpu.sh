accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision fp16 \
    trainers/train_gpt_v2_multigpu.py \
    --config checkpoints/IndexTTS-2-vLLM/config.yaml \
    --tokenizer checkpoints/IndexTTS-2-vLLM/jp_bpe.model \
    --base-checkpoint checkpoints/IndexTTS-2-vLLM/gpt.pth \
    --train-data-dir \
    /mnt/data_3t_2/datasets/indextts_train_data_v2/final_train_data/train_data_v2_260107 \
    --val-data-size 128 \
    --output-dir ./trained_ckpts \
    --batch-size-per-device 4 \
    --grad-accumulation 1 \
    --num-workers 2 \
    --epochs 1 \
    --learning-rate 4e-5 \
    --log-interval 10 \
    --val-interval 200 \
    --save_every 2000 \
    --keep-last 4 \
    --major-save-every 25000 \
    --use-duration-control \
    --duration-dropout 0.3 \
    --wandb-project "IndexTTS2-jp" \
    --wandb-run-name "20260108-test"