import os
import pickle
import sys
import numpy as np

# 原始路径设置
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from typing import Dict, Union
from concurrent.futures import ProcessPoolExecutor
import time
from tqdm import tqdm
from trainers.utils import ProcessedData


def process_single_file(file_path):
    """
    返回: (处理是否成功(int 0/1), 该文件包含的总时长(float))
    """
    file_duration = 0.0
    try:
        with open(file_path, 'rb') as f:
            data_list = pickle.load(f)
        
        if isinstance(data_list, list):
            # for item in tqdm(data_list, desc="Processing file"):
            for item in data_list:
                if isinstance(item, ProcessedData):
                    file_duration += item.duration
            
        return 1, file_duration  # 成功计数1，返回时长
        
    except Exception as e:
        print(f"\nError processing {file_path}: {e}")
        return 0, 0.0

# --- 主逻辑 ---
def main():
    target_dir = "/mnt/data_3t_2/datasets/indextts_train_data/Galgame-VisualNovel-Reupload"
    
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        return

    print("Scanning files...")
    all_files = []
    # 1. 先遍历收集所有文件路径
    for dirpath, dirnames, filenames in os.walk(target_dir):
        for filename in filenames:
            if filename.endswith(".pkl"):
                all_files.append(os.path.join(dirpath, filename))
    
    total_files_count = len(all_files)
    print(f"Found {total_files_count} .pkl files. Starting multiprocessing...")

    success_count = 0
    total_duration_sec = 0.0

    # 2. 多进程处理
    # max_workers默认是CPU核心数，适合IO+计算混合型任务
    with ProcessPoolExecutor(max_workers=16) as executor:
        # 使用tqdm显示进度条
        results = list(tqdm(executor.map(process_single_file, all_files), total=total_files_count, unit="file"))

    # 3. 统计结果
    for success, duration in results:
        success_count += success
        total_duration_sec += duration

    # for file_path in tqdm(all_files, total=total_files_count):
    #     success, duration = process_single_file(file_path)
    #     success_count += success
    #     total_duration_sec += duration

    # 4. 输出格式化时间
    print("\n" + "="*40)
    print("Processing Complete")
    print("="*40)
    print(f"Processed Files: {success_count} / {total_files_count}")
    
    # 时长统计
    seconds = total_duration_sec
    minutes = seconds / 60
    hours = minutes / 60
    
    print(f"Total Duration (Seconds): {seconds:.2f} s")
    print(f"Total Duration (Minutes): {minutes:.2f} min")
    print(f"Total Duration (Hours)  : {hours:.2f} hours")
    print("="*40)

if __name__ == "__main__":
    main()