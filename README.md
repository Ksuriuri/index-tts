## 公网推理

conda 环境：`index-tts`

### 1. 启动 webui
```bash
python webui.py
```
### 2. 快速创建公网访问链接
```bash
cloudflared tunnel --url http://127.0.0.1:7860
```


## 数据预处理

### v2版本

conda 环境：`index-tts`

1. 使用 `trainers/data_preprocess/preprocess_xxx.py` 预处理原始数据，主要执行 ASR 并计算 CER
2. 使用 `trainers/data_preprocess/speaker_diarization.py`: 生成说话人日志
3. 使用 `trainers/data_preprocess/gen_indextts_emb_xxx.py`: 预处理训练数据


## CV3-Eval 评测

### 生成音频

conda 环境：`cv3-eval`

```bash
python CV3-Eval.py --gpus 1,2,3,4,5,6,7
```

### 评测

conda 环境：`cv3-eval`

```bash
CUDA_VISIBLE_DEVICES=2 bash run_infer_cv3_eval.sh
python scripts/eval_speaker_similarity.py
```


## 模型训练

conda 环境：`index-tts`

```bash
bash trainers/train_gpt_v2_multigpu.sh
```
