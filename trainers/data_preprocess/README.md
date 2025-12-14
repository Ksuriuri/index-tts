
# 预处理脚本
## 处理 .parquet 类文件
### gpt_data_process_parquet.py
主要用于并行处理 Galgame-VisualNovel-Reupload 数据集（.parquet 子文件）为训练数据的 pickle 文件，保持原数据集的文件结构（将 .parquet 文件处理后保存为 .pkl 文件）。其中每个 .pkl 文件内容为 `List[ProcessedData]`， ProcessedData 的定义在 `trainers/utils.py` 中。

### convert_to_arrow.py
将上述输出的 .pkl 文件转换为训练使用的 .arrow 文件，方便训练高效读取。

## 处理 .wav + .lab 类文件
### gpt_data_process_wav_lab.py
主要用于并行处理原神星铁鸣潮等二次元游戏数据集（每个数据集里有多个角色文件夹，其中有多个.wav音频文件以及其对应的文本.lab文件）为训练数据的 pickle 文件，处理后将每个角色的数据分别保存为 角色名.pkl 文件。其中每个 .pkl 文件内容为 `List[ProcessedData]`， ProcessedData 的定义在 `trainers/utils.py` 中。

### convert_to_arrow_with_spk_pkl.py
将上述输出的 .pkl 文件转换为训练使用的 .arrow 文件，方便训练高效读取。另外，将同一角色的 condition 进行打乱，避免模型从 condition 中学习目标音频，导致推理时参考音频泄露。

## 处理 .tar 类文件
### gpt_data_process_tar.py
主要用于并行处理 Eimilia 数据集（.tar 子文件）为训练数据的 pickle 文件，保持原数据集的文件结构（将 .tar 文件处理后保存为 .pkl 文件）。其中每个 .pkl 文件内容为 `Dict[str, List[ProcessedData]]`， key 是 speaker id， value 是该 speaker 对应的 `List[ProcessedData]` ， ProcessedData 的定义在 `trainers/utils.py` 中。

### convert_to_arrow_with_spk_dict.py
将上述输出的 .pkl 文件转换为训练使用的 .arrow 文件，方便训练高效读取。另外，将同一 speaker 的 condition 进行打乱，避免模型从 condition 中学习目标音频，导致推理时参考音频泄露。

# 统计信息
## count_num_and_duration.py
用于统计一个文件夹下所有 .pkl 文件中的样本数量和时长。并输出一个统计图。

# 其他
## download_hf.py
从 HuggingFace 下载数据集。

## pkl_data_split.py
用于将一个 .pkl 文件中的数据集进行切分，并保存为多个 .pkl 文件，便于训练时随机读取数据。切分后的文件名会在原 xxx.pkl 文件的基础上加上 `-split-pkl-id`，如 xxx-split-pkl-0.pkl xxx-split-pkl-1.pkl ...