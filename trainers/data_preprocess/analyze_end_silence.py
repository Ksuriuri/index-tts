import os
import sys
import pickle
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Dict

# 原始路径设置
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

PREPROCESS_ROOT = "/mnt/data_3t_1/datasets/preprocess"
DATA_ROOT = "/mnt/data_3t_2/datasets/indextts_train_data_v2"
OUTPUT_DIR = "./outputs"

def get_parquet_path(pkl_path: str, source_name: str) -> str:
    try:
        path_parts = pkl_path.split(os.sep)
        # 找到 source_name 在路径中的位置
        if source_name in path_parts:
            idx = path_parts.index(source_name)
            rel_path = os.sep.join(path_parts[idx+1:])
            rel_path = rel_path.replace('.pkl', '.parquet')
            parquet_path = os.path.join(PREPROCESS_ROOT, source_name, rel_path)
            return parquet_path
        return None
    except ValueError:
        return None

def get_all_pkl_files(directory):
    pkl_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))
    return pkl_files

# ================= 核心统计逻辑 =================

def process_source_data(source_name):
    source_dir = os.path.join(DATA_ROOT, source_name)
    if not os.path.exists(source_dir):
        print(f"[Warn] Source dir not found: {source_dir}")
        return []

    pkl_files = get_all_pkl_files(source_dir)
    tail_durations = []
    
    print(f"--> Analyze {source_name}: Found {len(pkl_files)} pkl files")

    for pkl_path in tqdm(pkl_files, desc=f"Reading {source_name}"):
        try:
            # 1. 获取 Parquet 路径并读取 Meta
            parquet_path = get_parquet_path(pkl_path, source_name)
            if not parquet_path or not os.path.exists(parquet_path):
                continue
            
            # 仅读取 whisper 列以加速
            df_meta = pd.read_parquet(parquet_path, columns=["whisper_large_v3"])
            
            # 2. 读取 Pickle 数据
            # data_list = safe_load_pickle(pkl_path)
            with open(pkl_path, "rb") as f:
                data_list = pickle.load(f)
            if not data_list:
                continue

            # 3. 遍历列表进行匹配计算
            for item in data_list:
                idx = item["index"]
                processed_obj = item["data"]
                
                if idx >= len(df_meta):
                    continue
                
                whisper_info = df_meta.iloc[idx]["whisper_large_v3"]
                
                # 计算逻辑
                total_duration = processed_obj.duration
                segments = list(whisper_info.get("segments", []))
                
                if segments:
                    last_seg_end = segments[-1]["end"]
                    # 核心指标：总时长 - 最后一段文字结束时间
                    tail_gap = total_duration - last_seg_end
                    tail_durations.append(tail_gap)
                else:
                    # 如果 whisper 没有识别出任何段落，tail_gap 理论上等于 total_duration
                    # 但为了统计准确性，您可以选择记录或跳过。这里选择记录。
                    tail_durations.append(total_duration)

        except Exception as e:
            print(f"Error processing {pkl_path}: {traceback.format_exc()}")
            
    return tail_durations

def plot_distribution(data_dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 绘制合并的 Boxplot (查看异常值)
    plt.figure(figsize=(12, 6))
    plot_data = []
    labels = []
    for name, values in data_dict.items():
        if values:
            plot_data.append(values)
            labels.append(name)
    
    if plot_data:
        plt.boxplot(plot_data, labels=labels, vert=False, patch_artist=True)
        plt.title("Distribution of Audio Tail Silence (Duration - Whisper End)")
        plt.xlabel("Seconds")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "tail_silence_boxplot.png"))
        plt.close()

    # 2. 绘制每个 Source 的 Histogram
    for name, values in data_dict.items():
        if not values:
            continue
        
        values = np.array(values)
        
        # 统计描述
        desc = (
            f"Mean: {np.mean(values):.3f}s\n"
            f"Median: {np.median(values):.3f}s\n"
            f"Max: {np.max(values):.3f}s\n"
            f"Min: {np.min(values):.3f}s\n"
            f"<0s count: {np.sum(values < 0)} ({np.mean(values < 0)*100:.2f}%)"
        )
        
        plt.figure(figsize=(10, 6))
        # 过滤掉极端异常值以便绘图好看 (例如 > 3秒的，或者根据实际情况调整)
        # 这里只为了绘图截断，统计数据保留全量
        plot_vals = values[values < 5.0] 
        plot_vals = plot_vals[plot_vals > -1.0]

        sns.histplot(plot_vals, bins=100, kde=True)
        plt.title(f"Tail Silence Distribution: {name}")
        plt.xlabel("Tail Duration (seconds)")
        plt.axvline(0, color='r', linestyle='--', alpha=0.5)
        
        # 在图上添加统计文本
        plt.gca().text(0.95, 0.95, desc, transform=plt.gca().transAxes, 
                       verticalalignment='top', horizontalalignment='right', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"end_silence_{name}.png"))
        plt.close()

# ================= 主程序 =================

def main():
    print("Start analyzing tail durations...")
    results = {}
    
    stats_summary = []

    for source in SOURCE_NAMES:
        durations = process_source_data(source)
        results[source] = durations
        
        if durations:
            arr = np.array(durations)
            # stats_summary.append({
            #     "source": source,
            #     "count": len(arr),
            #     "mean": np.mean(arr),
            #     "std": np.std(arr),
            #     "min": np.min(arr),
            #     "p25": np.percentile(arr, 25),
            #     "p50": np.percentile(arr, 50),
            #     "p75": np.percentile(arr, 75),
            #     "p95": np.percentile(arr, 95),
            #     "p99": np.percentile(arr, 99),
            #     "max": np.max(arr),
            #     "negative_count": np.sum(arr < 0)
            # })
            print(f"[{source}] Mean: {np.mean(arr):.4f}s, P95: {np.percentile(arr, 95):.4f}s")
        else:
            print(f"[{source}] No data found.")

    # # 保存统计 CSV
    # if stats_summary:
    #     df_stats = pd.DataFrame(stats_summary)
    #     csv_path = os.path.join(OUTPUT_DIR, "tail_silence_stats.csv")
    #     df_stats.to_csv(csv_path, index=False)
    #     print(f"\nStats saved to {csv_path}")

    # 绘图
    plot_distribution(results)
    print(f"Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()