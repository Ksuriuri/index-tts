## gpt_data_process_parquet.py
主要用于并行处理 Galgame-VisualNovel-Reupload 数据集（.parquet子文件）为训练数据的 pickle 文件，保持原数据集的文件结构。

## count_num_and_duration.py
用于统计一个文件夹下所有 .pkl 文件中的样本数量和时长。

## pkl_data_split.py
用于将一个 .pkl 文件中的数据集进行切分，并保存为多个 .pkl 文件，便于训练时随机读取数据。切分后的文件名会在原 xxx.pkl 文件的基础上加上 `-split-pkl-id`，如 xxx-split-pkl-0.pkl xxx-split-pkl-1.pkl ...

## pkl 文件数据结构
每个 .pkl 文件里面都是 ProcessedData 的列表，ProcessedData的定义在 `trainers/utils.py` 中。