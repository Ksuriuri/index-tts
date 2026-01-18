import os
import pickle
import sys
import traceback
import numpy as np
import pandas as pd
from typing import Dict, Union, List, Any
from tqdm import tqdm
from collections import defaultdict
from tqdm.contrib.concurrent import process_map  # 核心并行库
from functools import partial

# 假设原始项目结构保留
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from trainers.utils import ProcessedData

# ================= 配置区域 =================
SOURCE_NAMES = [
    "Emilia_JA",
    "Emilia-YODAS_JA",
    "Gacha_games_jp",
    "Galgame-VisualNovel-Reupload",
    "Japanese-Eroge-Voice"
]

END_SILENCE_FILTER_NAMES = [
    "Emilia_JA",
    "Emilia-YODAS_JA",
    "Gacha_games_jp",
]

DATA_ROOT = "/mnt/data_3t_2/datasets/indextts_train_data_v2"
PREPROCESS_ROOT = "/mnt/data_3t_1/datasets/preprocess"

SHARD_SIZE = 40000 
MIN_DURATION = 0
MAX_DURATION = 36
MIN_TEXT_TOKENS = 1
MAX_TEXT_TOKENS = 600
CER_THRESHOLD = 0.10
CER_TYPE = "cer"
# CER_TYPE = "pron_CER"
END_SILENCE_MIN = 0.0
END_SILENCE_MAX = 0.7

# 并行相关配置
MAX_WORKERS = 4  # os.cpu_count()  # 使用所有 CPU 核心，也可以手动指定如 16

def get_parquet_path(pkl_path: str, source_name: str) -> str:
    try:
        path_parts = pkl_path.split(os.sep)
        if source_name not in path_parts:
             return None
        idx = path_parts.index(source_name)
        rel_path = os.sep.join(path_parts[idx+1:])
        rel_path = rel_path.replace('.pkl', '.parquet')
        parquet_path = os.path.join(PREPROCESS_ROOT, source_name, rel_path)
        return parquet_path
    except ValueError:
        return None

def get_all_pkl_files(directory):
    """递归查找所有符合条件的pkl文件"""
    pkl_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))
    return pkl_files

def process_single_file(args):
    """
    处理单个文件的逻辑，用于并行调用
    Args:
        args: tuple (file_path, source_name)
    Returns:
        dict: 包含统计信息的字典
    """
    file_path, source_name = args
    
    # 统计变量初始化
    stats = {
        "source_name": source_name,
        "valid_count": 0,
        "valid_duration": 0.0,
        "cer_skip": 0,
        "silence_skip": 0,
        "diarization_skip": 0,
        "error": None
    }

    try:
        # 1. 获取 Parquet 路径
        parquet_path = get_parquet_path(file_path, source_name)
        if not parquet_path or not os.path.exists(parquet_path):
            # 这里的 print 在多进程中可能会乱序，尽量减少
            return stats 

        # 2. 读取 Parquet
        try:
            df_meta = pd.read_parquet(parquet_path, columns=["whisper_large_v3", "speaker", "speaker_diarization"])
        except Exception as e:
            stats["error"] = f"Parquet Error: {e}"
            return stats

        # 3. 读取 Pickle
        with open(file_path, "rb") as f:
            data_list: list[dict] = pickle.load(f)

        if not data_list:
            return stats

        # 4. 遍历数据项
        for item in data_list:
            processed_data: ProcessedData = item["data"]
            parquet_idx = item["index"]
            
            # 确保索引不越界
            if parquet_idx >= len(df_meta):
                continue
                
            row = df_meta.iloc[parquet_idx]
            
            # --- 过滤逻辑开始 ---
            
            # A. CER 过滤
            whisper_info = row["whisper_large_v3"]
            cer = whisper_info.get(CER_TYPE, 1.0)
            if cer > CER_THRESHOLD:
                stats["cer_skip"] += 1
                continue
            
            # B. 尾部静音过滤
            if source_name in END_SILENCE_FILTER_NAMES:
                total_duration = processed_data.duration
                segments = list(whisper_info.get("segments", []))
                skip_flag = True
                if segments:
                    last_seg_end = segments[-1]["end"]
                    tail_gap = total_duration - last_seg_end
                    if END_SILENCE_MIN <= tail_gap <= END_SILENCE_MAX:
                        skip_flag = False
                if skip_flag:
                    stats["silence_skip"] += 1
                    continue
            
            # C. 说话人 Diarization 过滤
            unique_speakers = set(seg['speaker'] for seg in row["speaker_diarization"])
            if len(unique_speakers) != 1:
                stats["diarization_skip"] += 1
                continue

            # D. 基础长度/时长过滤
            if (processed_data.duration < MIN_DURATION or 
                processed_data.duration > MAX_DURATION or 
                processed_data.text_len < MIN_TEXT_TOKENS or 
                processed_data.text_len > MAX_TEXT_TOKENS):
                continue

            # --- 统计有效数据 ---
            stats["valid_count"] += 1
            stats["valid_duration"] += processed_data.duration

    except Exception as e:
        stats["error"] = f"Process Error: {traceback.format_exc()}"
    
    return stats

def main():
    # 1. 收集所有任务
    all_tasks = []
    print("正在扫描文件列表...")
    
    for source_name in SOURCE_NAMES:
        source_dir = os.path.join(DATA_ROOT, source_name)
        if not os.path.exists(source_dir):
            print(f"跳过不存在的目录: {source_dir}")
            continue

        files = get_all_pkl_files(source_dir)
        print(f"Source: {source_name}, Files: {len(files)}")
        
        # 将任务打包成 tuple: (file_path, source_name)
        for f in files:
            all_tasks.append((f, source_name))

    print(f"总任务数: {len(all_tasks)}")
    
    if not all_tasks:
        return

    # 2. 并行处理
    # chunksize=1 表示每个进程每次领1个任务。
    # 如果文件非常多且处理极快，可以适当调大 chunksize (例如 10) 以减少通信开销
    results = process_map(
        process_single_file, 
        all_tasks, 
        max_workers=MAX_WORKERS, 
        chunksize=1, 
        desc="Parallel Processing"
    )

    # 3. 聚合结果
    final_num_dict = defaultdict(int)
    final_duration_dict = defaultdict(float)
    
    # 错误日志聚合
    error_logs = []

    print("\n正在聚合统计结果...")
    for res in results:
        src = res["source_name"]
        
        # 记录错误
        if res["error"]:
            error_logs.append(res["error"])
            
        final_num_dict[src] += res["valid_count"]
        final_duration_dict[src] += res["valid_duration"]

    # 4. 输出报告
    print("\n" + "="*30)
    print("处理完成报告")
    print("="*30)
    
    if error_logs:
        print(f"警告: 出现了 {len(error_logs)} 个文件处理错误 (仅显示前5个):")
        for err in error_logs[:5]:
            print(err)
        print("-" * 20)

    print(f"{'Source Name':<30} | {'Count':<10} | {'Hours':<10}")
    print("-" * 56)
    
    total_duration_all = 0
    total_count_all = 0
    
    for source_name in SOURCE_NAMES:
        count = final_num_dict[source_name]
        dur_hours = final_duration_dict[source_name] / 3600
        total_count_all += count
        total_duration_all += final_duration_dict[source_name]
        
        print(f"{source_name:<30} | {count:<10} | {dur_hours:.2f} h")
    
    print("-" * 56)
    print(f"{'TOTAL':<30} | {total_count_all:<10} | {(total_duration_all / 3600):.2f} h")

if __name__ == "__main__":
    # Windows/macOS 下 multiprocessing 需要这个保护，Linux 下建议也保留
    main()