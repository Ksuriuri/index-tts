import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import numpy as np

# --- 配置 (保持与你之前的配置一致) ---
# DATASET_NAME = "Galgame-VisualNovel-Reupload"
DATASET_NAME = "Gacha_games_jp"
# DATASET_NAME = "Emilia_JA"
# DATASET_NAME = "Emilia-YODAS_JA"
# DATASET_NAME = "Japanese-Eroge-Voice"
DATASET_DIR = f"/mnt/data_3t_1/datasets/preprocess/{DATASET_NAME}"
OUTPUT_IMG = f"outputs/speaker_distribution_{DATASET_NAME}.png"

def analyze_dataset():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Directory {DATASET_DIR} does not exist.")
        return

    parquet_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.parquet")))
    print(f"Found {len(parquet_files)} parquet files in {DATASET_DIR}")

    # 用于存储统计数据
    # key: 说话人数量 (0, 1, 2...), value: 文件数量
    speaker_count_per_file = []
    
    # 额外的统计：总共有多少个音频片段
    total_segments = 0
    total_audio_files = 0
    files_missing_diarization_col = 0

    print("Starting analysis...")
    
    for f in tqdm(parquet_files, desc="Analyzing files"):
        try:
            # 只读取需要的列以加快速度
            try:
                df = pd.read_parquet(f, columns=["speaker_diarization"])
            except Exception:
                # 如果没有这一列，可能还没处理完，读取全部看看列是否存在
                df = pd.read_parquet(f)
                if "speaker_diarization" not in df.columns:
                    files_missing_diarization_col += 1
                    continue
            
            total_audio_files += len(df)

            # 遍历每一行
            for diarization_data in df["speaker_diarization"]:
                if diarization_data is None or len(diarization_data) == 0:
                    speaker_count_per_file.append(0)
                    continue

                # diarization_data 是一个列表，形如: 
                # [{'start': 0.1, 'end': 1.5, 'speaker': 'SPEAKER_00'}, ...]
                
                # 统计这段音频里有多少个唯一的 speaker
                unique_speakers = set(seg['speaker'] for seg in diarization_data)
                num_speakers = len(unique_speakers)
                
                speaker_count_per_file.append(num_speakers)
                total_segments += len(diarization_data)

        except Exception as e:
            print(f"Error processing file {f}: {e}")

    # --- 汇总结果 ---
    if not speaker_count_per_file:
        print("No diarization data found.")
        return

    counts = Counter(speaker_count_per_file)
    total_processed = len(speaker_count_per_file)
    max_speakers = max(counts.keys()) if counts else 0

    print("\n" + "="*50)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Total Audio Rows Processed: {total_processed}")
    print(f"Total Files Missing Column: {files_missing_diarization_col}")
    print(f"Total Speech Segments Detected: {total_segments}")
    print("="*50)
    print(f"{'# Speakers':<15} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 42)

    # 按说话人数量排序输出 (0, 1, 2, ...)
    sorted_keys = sorted(counts.keys())
    for k in sorted_keys:
        count = counts[k]
        percent = (count / total_processed) * 100
        print(f"{k:<15} | {count:<10} | {percent:.2f}%")
    print("="*50)

    # --- 绘图 (柱状图) ---
    try:
        plt.figure(figsize=(10, 6))
        # 过滤掉极其罕见的长尾数据以便绘图清晰（比如某条音频有10个说话人，只出现一次）
        # 这里全画，但如果 max_speakers 很大，可以限制 x 轴
        
        x = sorted_keys
        y = [counts[k] for k in x]
        
        bars = plt.bar(x, y, color='skyblue', edgecolor='black')
        
        plt.xlabel('Number of Unique Speakers per Audio File')
        plt.ylabel('Count of Audio Files')
        plt.title(f'Speaker Distribution in {DATASET_NAME}\n(Total: {total_processed})')
        plt.xticks(x)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 在柱子上标数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}',
                     ha='center', va='bottom')

        plt.savefig(OUTPUT_IMG)
        print(f"\nDistribution chart saved to: {os.path.abspath(OUTPUT_IMG)}")
        # plt.show() # 如果在 Jupyter 中运行取消注释
    except Exception as e:
        print(f"Could not save plot: {e}")

if __name__ == "__main__":
    analyze_dataset()