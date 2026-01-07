import os
import glob
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- 配置 ---
# DATASET_NAME = "Galgame-VisualNovel-Reupload"
# DATASET_NAME = "Gacha_games_jp"
DATASET_NAME = "Emilia_JA"
# DATASET_NAME = "Emilia-YODAS_JA"
# DATASET_NAME = "Japanese-Eroge-Voice"
DATASET_DIR = f"/mnt/data_3t_1/datasets/preprocess/{DATASET_NAME}"

# 修改输出文件名以区分统计内容
PLOT_OUTPUT_PATH = f"outputs/segment_count_distribution_{DATASET_NAME}.png" 

def calculate_segment_count(segments_list, silence_threshold=0.5):
    """
    根据静音阈值计算实际的语音段数。
    逻辑：如果两段语音之间的间隔小于阈值，视为同一段；
          如果间隔 >= 阈值，计数 +1。
    """
    if len(segments_list) == 0 or not isinstance(segments_list, list):
        return 0
    
    # 确保按开始时间排序
    segments_list.sort(key=lambda x: x.get('start', 0))
    
    count = 1
    # 记录当前合并段的结束时间
    last_end = segments_list[0].get('end', 0)
    
    for i in range(1, len(segments_list)):
        current_start = segments_list[i].get('start', 0)
        current_end = segments_list[i].get('end', 0)
        
        # 计算静音间隙
        gap = current_start - last_end
        
        if gap >= silence_threshold:
            # 发现足够长的静音，视为新的一段
            count += 1
            last_end = current_end
        else:
            # 静音太短，视为连接在一起，更新结束时间（取较晚的结束时间）
            last_end = max(last_end, current_end)
            
    return count

def load_segment_data(data_dir):
    """
    遍历目录下所有的 parquet 文件，提取 Segments 计数
    """
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "part_*.parquet")))
    
    if not parquet_files:
        print(f"Error: No parquet files found in {data_dir}")
        return []

    all_counts = []
    
    print(f"Found {len(parquet_files)} files. Loading and analyzing segments...")
    
    for file_path in tqdm(parquet_files, unit="file"):
        try:
            df = pd.read_parquet(file_path, columns=["whisper_large_v3"])
            
            # 提取 segments 列表并计算段数
            # 每一行 x 是一个 dict，里面包含 'segments' 列表
            counts = []
            for x in df['whisper_large_v3']:
                if x is not None and isinstance(x, dict):
                    segs = list(x['segments'])
                    cnt = calculate_segment_count(segs, silence_threshold=0.5)
                    if cnt == 0:
                        print(f"Error: No segments found in {segs}, cer: {x['cer']}")
                    counts.append(cnt)
                else:
                    counts.append(0)
            
            all_counts.extend(counts)
            
        except Exception as e:
            print(f"Error reading {file_path}: {traceback.format_exc()}")
            continue

    return np.array(all_counts, dtype=np.int32)

def print_statistics(counts):
    """
    打印统计信息
    """
    if len(counts) == 0:
        print("No data to analyze.")
        return

    print("\n" + "="*30)
    print("   Segment Count Statistics   ")
    print("   (Gap >= 0.5s = Split)      ")
    print("="*30)
    print(f"Total Samples: {len(counts)}")
    print(f"Mean Segments: {np.mean(counts):.4f}")
    print(f"Max Segments:  {np.max(counts)}")
    print("-" * 30)
    
    # 统计具体段数的分布比例
    unique, frequency = np.unique(counts, return_counts=True)
    total = len(counts)
    
    print("Distribution:")
    # 只打印前10个最常见的段数情况，或者打印 0-5 段的比例
    for val, freq in zip(unique, frequency):
        if val <= 5 or freq / total > 0.001: # 打印 <= 5段 或者 占比超过 0.1% 的情况
            print(f"  {val} Segments: {freq:7d} ({freq/total*100:.2f}%)")
    
    if np.max(counts) > 5:
        over_5 = np.sum(counts > 5)
        print(f"  >5 Segments: {over_5:7d} ({over_5/total*100:.2f}%)")
        
    print("="*30)

def plot_distribution(counts, output_path):
    """
    绘制分布图 (离散柱状图)
    """
    if len(counts) == 0:
        return

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # 为了图表可读性，如果长尾太长，截断显示（比如只显示到 99 分位数的最大值 + 1）
    p99 = np.percentile(counts, 99)
    max_plot_val = max(5, int(p99) + 2) # 至少显示到5
    
    plot_data = counts[counts <= max_plot_val]
    outlier_count = len(counts) - len(plot_data)
    
    # 使用 discrete=True 来绘制整数分布
    ax = sns.histplot(plot_data, discrete=True, color="mediumpurple", edgecolor="black", alpha=0.8)

    plt.title(f'Audio Segments Distribution (Gap >= 0.5s)\nDATASET: {DATASET_NAME}', fontsize=14)
    plt.xlabel('Number of Segments', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # 设置 X 轴刻度为整数
    plt.xticks(range(0, max_plot_val + 1))
    
    # 在柱子上标数值
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'bottom', 
                        fontsize=9, xytext=(0, 2), 
                        textcoords='offset points')

    if outlier_count > 0:
        plt.figtext(0.5, 0.01, f"Note: {outlier_count} samples with > {max_plot_val} segments excluded from plot.", 
                    ha="center", fontsize=10, color="gray")

    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {os.path.abspath(output_path)}")

def main():
    # 1. 加载数据 (统计段数)
    counts = load_segment_data(DATASET_DIR)
    
    # 2. 打印统计
    if len(counts) > 0:
        print_statistics(counts)
        
        # 3. 绘图
        plot_distribution(counts, PLOT_OUTPUT_PATH)
    else:
        print("No data found.")

if __name__ == "__main__":
    main()