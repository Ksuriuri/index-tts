import os
import pickle
import sys
import random
import traceback
import numpy as np
import pandas as pd
from typing import Dict, Union, List, Any
from datasets import Dataset
from tqdm import tqdm
from collections import defaultdict

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from trainers.utils import ProcessedData

# ================= 配置区域 =================
# 1. 修改为列表，支持多个源文件夹
SOURCE_NAMES = [
    "Emilia_JA",
    "Emilia-YODAS_JA",
    "Gacha_games_jp",
    "Galgame-VisualNovel-Reupload",
    "Japanese-Eroge-Voice"
]

# 基础数据路径
DATA_ROOT = "/mnt/data_3t_2/datasets/indextts_train_data_v2"
# 预处理 Parquet 文件的根目录
PREPROCESS_ROOT = "/mnt/data_3t_1/datasets/preprocess"
# 输出目录
TARGET_DIR = f"{DATA_ROOT}/final_train_data/train_data_v2_260107"

SHARD_SIZE = 40000  # 每个分片包含的数据量
MIN_DURATION = 0
MAX_DURATION = 36
MIN_TEXT_TOKENS = 1
MAX_TEXT_TOKENS = 600
CER_THRESHOLD = 0.35  # CER 过滤阈值

def get_parquet_path(pkl_path: str, source_name: str) -> str:
    try:
        # 分割路径
        path_parts = pkl_path.split(os.sep)
        # 找到 source_name 的索引
        idx = path_parts.index(source_name)
        # 提取 source_name 之后的部分
        rel_path = os.sep.join(path_parts[idx+1:])
        # 替换后缀
        rel_path = rel_path.replace('.pkl', '.parquet')
        # 拼接新的根目录
        parquet_path = os.path.join(PREPROCESS_ROOT, source_name, rel_path)
        return parquet_path
    except ValueError:
        print(f"警告: 路径中未找到 source_name ({source_name}): {pkl_path}")
        return None

def get_all_pkl_files(directory):
    """递归查找所有符合条件的pkl文件"""
    pkl_files = []
    # 使用 os.walk 递归遍历
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))
    return pkl_files


def main():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    shard_count = 0

    # 1. 遍历多个 Source
    data_num_dict = {}
    for source_name in SOURCE_NAMES:
        data_num_dict[source_name] = 0
        source_dir = os.path.join(DATA_ROOT, source_name)
        if not os.path.exists(source_dir):
            print(f"跳过不存在的目录: {source_dir}")
            continue

        files = get_all_pkl_files(source_dir)
        print(f"开始处理 {source_name}, 共 {len(files)} 个文件")

        pbar = tqdm(files, desc=f"Processing {source_name}")
        for file_path in pbar:
            cer_skip_num = 0
            diarization_skip_num = 0
            try:
                # 2. 读取对应的 Parquet 文件
                parquet_path = get_parquet_path(file_path, source_name)
                if not parquet_path or not os.path.exists(parquet_path):
                    # 如果找不到对应的 parquet，根据需求决定是跳过还是报错。
                    # 这里假设必须有 meta 信息才能继续
                    print(f"\n缺少 Parquet 文件: {parquet_path}, 跳过 {file_path}")
                    continue
                
                # 读取 Parquet (只读取需要的列以加速，如果文件很大)
                # 必须读取: whisper_large_v3, speaker
                try:
                    df_meta = pd.read_parquet(parquet_path, columns=["whisper_large_v3", "speaker", "speaker_diarization"])
                except Exception as e:
                    print(f"\nParquet 读取失败 {parquet_path}: {e}")
                    continue

                with open(file_path, "rb") as f:
                    data_list: list[dict] = pickle.load(f)

                # 处理 pkl 中的数据项
                if not data_list:
                    continue

                for item in data_list:
                    processed_data: ProcessedData = item["data"]
                    parquet_idx = item["index"]
                        
                    row = df_meta.iloc[parquet_idx]
                    
                    # ================= 2. CER 过滤逻辑 =================
                    whisper_info = row["whisper_large_v3"]
                    cer = whisper_info.get("cer", 1.0) # 默认 1.0 (Bad)
                    if cer > CER_THRESHOLD: 
                        cer_skip_num += 1
                        continue
                    
                    # 过滤说话人不为 1 的数据
                    unique_speakers = set(seg['speaker'] for seg in row["speaker_diarization"])
                    num_speakers = len(unique_speakers)
                    if num_speakers != 1:
                        diarization_skip_num += 1
                        continue

                    # 基础长度/时长过滤
                    if processed_data.duration < MIN_DURATION or processed_data.duration > MAX_DURATION or \
                       processed_data.text_len < MIN_TEXT_TOKENS or processed_data.text_len > MAX_TEXT_TOKENS:
                        continue

                    data_num_dict[source_name] += 1

            except Exception as e:
                print(f"\n处理文件 {file_path} 时出错: {traceback.format_exc()}")
            
            print(f"{file_path} 处理完成，跳过 cer: {cer_skip_num}, diarization: {diarization_skip_num}")

    print(data_num_dict)

if __name__ == "__main__":
    main()