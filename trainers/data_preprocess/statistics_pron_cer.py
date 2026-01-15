import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- 配置 ---
DATASET_NAME = "Galgame-VisualNovel-Reupload"
# DATASET_NAME = "Gacha_games_jp"
# DATASET_NAME = "Emilia_JA"
# DATASET_NAME = "Emilia-YODAS_JA"
# DATASET_NAME = "Japanese-Eroge-Voice"
DATASET_DIR = f"/mnt/data_3t_1/datasets/preprocess/{DATASET_NAME}"

PLOT_OUTPUT_PATH = f"outputs/pron_cer_distribution_{DATASET_NAME}.png"  # 输出图片路径

def load_pron_cer_data(data_dir):
    """
    遍历目录下所有的 parquet 文件，提取 pron_CER 数据
    """
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "part_*.parquet")))
    
    if not parquet_files:
        print(f"Error: No parquet files found in {data_dir}")
        return []

    all_pron_cers = []
    
    print(f"Found {len(parquet_files)} files. Loading data...")
    
    for file_path in tqdm(parquet_files, unit="file"):
        try:
            # 只读取需要的列以节省内存
            df = pd.read_parquet(file_path, columns=["whisper_large_v3"])
            
            # 数据是以 dict 形式存储在 whisper_large_v3 列中的
            # 使用列表推导式提取 pron_cer，速度通常比 apply 快
            # 注意处理可能存在的 None 或异常数据
            pron_cers = [
                x.get('pron_CER', 0.0) 
                for x in df['whisper_large_v3'] 
                if x is not None and isinstance(x, dict)
            ]
            
            all_pron_cers.extend(pron_cers)
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    return np.array(all_pron_cers, dtype=np.float32)

def print_statistics(pron_cers):
    """
    打印统计信息
    """
    if len(pron_cers) == 0:
        print("No data to analyze.")
        return

    print("\n" + "="*30)
    print("      pron_CER Statistics      ")
    print("="*30)
    print(f"Total Samples: {len(pron_cers)}")
    print(f"Mean pron_CER:      {np.mean(pron_cers):.4f}")
    print(f"Median pron_CER:    {np.median(pron_cers):.4f}")
    print(f"Std Dev:       {np.std(pron_cers):.4f}")
    print(f"Min pron_CER:       {np.min(pron_cers):.4f}")
    print(f"Max pron_CER:       {np.max(pron_cers):.4f}")
    print("-" * 30)
    
    # 分位数
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    perc_vals = np.percentile(pron_cers, percentiles)
    
    for p, val in zip(percentiles, perc_vals):
        print(f"P{p:02d}:           {val:.4f}")
    
    # 常用阈值统计
    print("-" * 30)
    print(f"Samples < 0.1 (10%): {np.sum(pron_cers < 0.1) / len(pron_cers) * 100:.2f}%")
    print(f"Samples < 0.2 (20%): {np.sum(pron_cers < 0.2) / len(pron_cers) * 100:.2f}%")
    print(f"Samples = 0.0 (Perfect): {np.sum(pron_cers == 0) / len(pron_cers) * 100:.2f}%")
    print("="*30)

def plot_distribution(pron_cers, output_path):
    """
    绘制分布图
    """
    if len(pron_cers) == 0:
        return

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # 过滤掉极端的异常值用于绘图 (比如 pron_CER > 1.5 的通常是由于幻觉导致的极长文本，会压缩图表)
    # 统计数据仍然包含它们，但图表为了美观截断一下
    plot_data = pron_cers[pron_cers <= 1.5]
    outlier_count = len(pron_cers) - len(plot_data)
    
    # 绘制直方图和 KDE (核密度估计)
    sns.histplot(plot_data, bins=100, kde=True, color="skyblue", edgecolor="black", alpha=0.7)

    plt.title(f'pron_CER Distribution (Whisper Large V3)\nExcluded {outlier_count} samples > 1.5 for visualization', fontsize=14)
    plt.xlabel('Character Error Rate (pron_CER)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xlim(0, 1.0) # 重点关注 0-1 范围
    
    # 添加平均值线
    mean_val = np.mean(pron_cers)
    plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.3f}')
    
    # 添加中位数线
    median_val = np.median(pron_cers)
    plt.axvline(median_val, color='g', linestyle='-', label=f'Median: {median_val:.3f}')

    plt.legend()
    plt.tight_layout()
    
    # 保存
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {os.path.abspath(output_path)}")
    # plt.show() # 如果在 Jupyter 或本地运行可以取消注释

def main():
    # 1. 加载数据
    pron_cers = load_pron_cer_data(DATASET_DIR)
    
    # 2. 打印统计
    if len(pron_cers) > 0:
        print_statistics(pron_cers)
        
        # 3. 绘图
        plot_distribution(pron_cers, PLOT_OUTPUT_PATH)
    else:
        print("No pron_CER data found.")

if __name__ == "__main__":
    main()