from collections import OrderedDict
import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt  # 引入绘图库

# 原始路径设置
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from typing import Dict, List, Union
from concurrent.futures import ProcessPoolExecutor
import time
from tqdm import tqdm
from trainers.utils import ProcessedData


def process_single_file(file_path):
    """
    返回: (处理是否成功(int 0/1), 该文件包含的总时长(float), 时长分布字典(Dict))
    """
    file_duration = 0.0
    duration_distribution: Dict[int, int] = {}
    try:
        with open(file_path, 'rb') as f:
            data_sets = pickle.load(f)
        
        data_list: List[ProcessedData] = []
        if isinstance(data_sets, list):
            for item in data_sets:
                data_list.append(item)
        elif isinstance(data_sets, dict):
            for sub_data_list in data_sets.values():
                data_list.extend(sub_data_list)

        for item in data_list:
            if isinstance(item, ProcessedData):
                file_duration += item.duration
                # 四舍五入取整
                d_int = int(item.duration + 0.5)
                if d_int not in duration_distribution:
                    duration_distribution[d_int] = 0
                duration_distribution[d_int] += 1
            
        return 1, file_duration, duration_distribution  # 成功
        
    except Exception as e:
        print(f"\nError processing {file_path}: {e}")
        # [修复] 这里必须返回3个值，否则主进程解包时会报错
        return 0, 0.0, {}

# --- 主逻辑 ---
def main():
    # 设置中文字体（可选，如果系统支持，防止中文乱码，这里暂用英文）
    plt.rcParams['axes.unicode_minus'] = False 

    # target_dir = "/mnt/data_3t_2/datasets/indextts_train_data/Galgame-VisualNovel-Reupload"
    # target_dir = "/mnt/data_3t_2/datasets/indextts_train_data/Gacha_games_jp"
    # target_dir = "/mnt/data_3t_2/datasets/indextts_train_data/Emilia-YODAS/JA"
    # target_dir = "/mnt/data_3t_2/datasets/indextts_train_data/Emilia/JA"
    target_dir = "/mnt/data_3t_2/datasets/indextts_train_data/Japanese-Eroge-Voice"

    save_name = "outputs/duration_distribution.png"
    
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        return

    print("Scanning files...")
    all_files = []
    for dirpath, dirnames, filenames in os.walk(target_dir):
        for filename in filenames:
            if filename.endswith(".pkl"):  #  and "-split-pkl-" in filename
                all_files.append(os.path.join(dirpath, filename))
    
    total_files_count = len(all_files)
    print(f"Found {total_files_count} .pkl files. Starting multiprocessing...")

    # [注意] 调试模式只取前8个，正式跑请去掉这行
    # all_files = all_files[:8] 
    
    success_count = 0
    total_duration_sec = 0.0

    # 2. 多进程处理
    # 为了避免 all_files 切片导致 total 计数对不上，重新计算处理数量
    files_to_process = all_files 
    
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_single_file, files_to_process), total=len(files_to_process), unit="file"))

    # 3. 统计结果
    final_distribution: Dict[int, int] = {}
    
    for success, duration, dist_sub in results:
        success_count += success
        total_duration_sec += duration
        
        # 合并字典
        for k, v in dist_sub.items():
            if k not in final_distribution:
                final_distribution[k] = 0
            final_distribution[k] += v

    # 4. 排序与绘图 (新增部分)
    print("\n" + "="*40)
    print("final_distribution: ", final_distribution)
    print("Generating Plot...")
    
    if final_distribution:
        # 按时长(key)排序
        sorted_durations = sorted(final_distribution.keys())
        counts = [final_distribution[k] for k in sorted_durations]

        # 创建图表
        plt.figure(figsize=(12, 6), dpi=100)
        
        # 绘制柱状图
        plt.bar(sorted_durations, counts, color='skyblue', edgecolor='blue', alpha=0.7)
        
        # 设置标题和标签
        plt.title('Audio Duration Distribution', fontsize=16)
        plt.xlabel('Duration (Seconds)', fontsize=12)
        plt.ylabel('Count (Number of Files)', fontsize=12)
        
        # 添加网格
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        # 如果数据点不过于密集，可以在柱子上显示数值（可选）
        # for a, b in zip(sorted_durations, counts):
        #     plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)

        plt.savefig(save_name)
        print(f"Plot saved to: {os.path.abspath(save_name)}")
        
        # 关闭图表释放内存
        plt.close()
    else:
        print("No data found to plot.")

    # 5. 输出文本统计
    print("\n" + "="*40)
    print("Processing Complete")
    print("="*40)
    print(f"Processed Files: {success_count} / {len(files_to_process)}")
    
    seconds = total_duration_sec
    minutes = seconds / 60
    hours = minutes / 60
    
    print(f"Total Duration (Seconds): {seconds:.2f} s")
    print(f"Total Duration (Minutes): {minutes:.2f} min")
    print(f"Total Duration (Hours)  : {hours:.2f} hours")
    print("="*40)

if __name__ == "__main__":
    main()




# Galgame-VisualNovel-Reupload
# ========================================
# final_distribution:  {6: 687085, 9: 316329, 7: 556328, 4: 937028, 5: 817985, 3: 946003, 2: 856137, 10: 226448, 1: 686886, 13: 73858, 8: 429961, 15: 32520, 11: 158027, 12: 108155, 14: 48928, 17: 14071, 18: 9164, 16: 21370, 0: 37674, 25: 486, 19: 6034, 20: 3772, 22: 1691, 21: 2497, 24: 761, 27: 216, 28: 185, 26: 353, 23: 1141, 29: 100, 30: 72, 34: 24, 31: 53, 36: 5, 32: 32, 35: 18, 33: 20}
# Generating Plot...
# Plot saved to: /mnt/data_sdd/hhy/index-tts/duration_distribution.png

# ========================================
# Processing Complete
# ========================================
# Processed Files: 7008 / 7008
# Total Duration (Seconds): 35912827.47 s
# Total Duration (Minutes): 598547.12 min
# Total Duration (Hours)  : 9975.79 hours
# ========================================



# Gacha_games_jp
# ========================================
# final_distribution:  {3: 18799, 1: 8911, 9: 12426, 4: 21295, 8: 14825, 7: 16893, 13: 5148, 5: 20747, 6: 19265, 11: 8636, 2: 14597, 12: 6661, 10: 10594, 15: 3017, 16: 2268, 14: 4003, 18: 1307, 19: 981, 0: 779, 24: 322, 20: 760, 23: 357, 17: 1590, 31: 87, 27: 164, 30: 91, 22: 449, 26: 197, 34: 61, 29: 124, 36: 26, 25: 294, 35: 55, 33: 77, 32: 88, 21: 583, 28: 170}
# Generating Plot...
# Plot saved to: /mnt/data_sdd/hhy/index-tts/duration_distribution.png

# ========================================
# Processing Complete
# ========================================
# Processed Files: 4457 / 4457
# Total Duration (Seconds): 1395921.67 s
# Total Duration (Minutes): 23265.36 min
# Total Duration (Hours)  : 387.76 hours
# ========================================



# Emilia-YODAS
# ========================================
# final_distribution:  {9: 28728, 4: 56119, 11: 19881, 8: 34816, 12: 16866, 10: 24362, 5: 49975, 7: 40734, 13: 14064, 6: 47238, 3: 32946, 17: 8072, 16: 9102, 18: 7106, 14: 11992, 20: 4601, 24: 1425, 19: 6409, 23: 1759, 21: 2779, 28: 1427, 29: 1796, 27: 1034, 26: 1067, 15: 10220, 22: 2300, 30: 2545, 25: 1175}
# Generating Plot...
# Plot saved to: /mnt/data_sdd/hhy/index-tts/duration_distribution.png

# ========================================
# Processing Complete
# ========================================
# Processed Files: 30 / 30
# Total Duration (Seconds): 3911871.80 s
# Total Duration (Minutes): 65197.86 min
# Total Duration (Hours)  : 1086.63 hours
# ========================================



# Emilia
# ========================================
# final_distribution:  {4: 179884, 6: 90133, 3: 124567, 5: 125604, 10: 30275, 7: 65672, 11: 23688, 12: 19367, 8: 50360, 9: 39159, 18: 7032, 17: 8089, 14: 12878, 24: 1640, 16: 9343, 26: 1204, 28: 2252, 29: 2985, 15: 11016, 23: 1982, 21: 3106, 19: 6352, 27: 1470, 30: 2569, 20: 4729, 25: 1405, 22: 2531, 13: 15560}
# Generating Plot...
# Plot saved to: /mnt/data_sdd/hhy/index-tts/duration_distribution.png

# ========================================
# Processing Complete
# ========================================
# Processed Files: 70 / 70
# Total Duration (Seconds): 5998450.19 s
# Total Duration (Minutes): 99974.17 min
# Total Duration (Hours)  : 1666.24 hours
# ========================================



# Japanese-Eroge-Voice
# ========================================
# final_distribution:  {7: 20362, 2: 14925, 12: 6257, 14: 3083, 1: 11360, 3: 19808, 5: 24498, 8: 17196, 11: 8619, 4: 24318, 9: 14396, 6: 22362, 10: 11713, 29: 126, 0: 317, 13: 4363, 15: 2109, 19: 560, 16: 1503, 18: 742, 30: 151, 24: 157, 17: 978, 26: 79, 20: 509, 23: 187, 22: 253, 27: 77, 21: 368, 25: 108, 28: 74}
# Generating Plot...
# Plot saved to: /mnt/data_sdd/hhy/index-tts/duration_distribution.png

# ========================================
# Processing Complete
# ========================================
# Processed Files: 218 / 218
# Total Duration (Seconds): 1396023.57 s
# Total Duration (Minutes): 23267.06 min
# Total Duration (Hours)  : 387.78 hours
# ========================================