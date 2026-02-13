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

### 1. 转为标准格式
- 使用 `trainers/data_preprocess/preprocess_xxx.py` 预处理原始数据，主要执行 ASR 并计算 CER（保存为 .parquet 格式）
- 其中，`.parquet` 文件中每行的内容如下：
```python
{
    "audio": 音频文件的二进制数据流,
    "text": 数据集原文本,
    "speaker": 说话人ID,
    "whisper_large_v3": {
        "text": ASR文本,
        "cer": float,
        "language": str,
        "segments": [{"text": str, "start": float, "end": float}, ...]
    }
}
```

### 2. 计算发音级别的 CER
- 使用 `trainers/data_preprocess/pron_cer_calc_xxx.py` 计算发音级别的 CER，追加到 .parquet 文件中
- 具体来说，在 `whisper_large_v3` 中添加 `pron_CER` 字段，float()

### 3. 生成说话人日志
- 使用 `trainers/data_preprocess/speaker_diarization.py`: 生成说话人日志，追加到 .parquet 文件中
- 具体来说，添加了 `speaker_diarization` 字段：`[{"speaker": str, "start": float, "end": float}, ...]`

### 3.5. 为无说话人标签数据生成参考音频
- 使用 `trainers/data_preprocess/speaker_synthesis_indextts_jp.py`: 为无说话人标签数据生成参考音频，保存为同目录结构且同名的 .parquet 文件中仅保存 `audio` 和 `text` 字段
- 当前过滤条件：1. 原音频大于6秒且小于36秒；2. pron_CE小于0.5；3. 音频中仅有一个说话人
- 生成文本：提取原文件中长度小于100且大于10的文本组成数组，从中随机选择

### 4. 预处理训练数据
使用 `trainers/data_preprocess/gen_indextts_emb_xxx.py`: 预处理训练数据：生成 embedding 和 token id（.pkl 格式）
- 其中，`.pkl` 文件与原 `.parquet` 一一对应，内容为 `List[Dict[str, Any]]`，具体是 `[{"index": int, "data": ProcessedData}]`，`ProcessedData` 参考 `trainers.utils`
- 对于无说话人标签的数据，将语音间静音大于 MIN_SILLENCE_DURATION 的片段进行切分，`index` 指向切分前的 `index`

### 5. 生成最终训练数据集
#### 不使用生成数据
- 使用 `trainers/data_preprocess/convert_to_arrow_multi.py` 生成最终训练数据集（.arrow 格式）
- 根据 CER 或者 pron_CER 过滤；根据 `speaker_diarization` 过滤说话人数量不等于1的数据；根据 whisper 的 segments 字段最后一个 end 过滤末尾过长的数据
- 对于原 `.parquet` 文件中 `speaker` 字段为 None 的数据，将 `speaker` 字段设置为 `{source_name}_idx_{index}`，这样便能标记同一说话人


## CV3-Eval 评测

### 生成音频

conda 环境：`cv3-eval`

```bash
python CV3-Eval.py --gpus 0,1,2,3,4,5,6,7
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
