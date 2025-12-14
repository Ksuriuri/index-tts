import os
import glob
import pickle
import sys
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Union, List
from datasets import Dataset
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from trainers.utils import ProcessedData

# ================= 配置区域 =================
# SOURCE_NAME = "Emilia-YODAS-JA"  # 根据实际情况修改名称
# SOURCE_DIR = f"/mnt/data_3t_2/datasets/indextts_train_data/Emilia-YODAS/JA" 

SOURCE_NAME = "Emilia-JA"  # 根据实际情况修改名称
SOURCE_DIR = f"/mnt/data_3t_2/datasets/indextts_train_data/Emilia/JA" 
TARGET_DIR = f"/mnt/data_3t_2/datasets/indextts_train_data/final_train_data/{SOURCE_NAME}_arrow"
SHARD_SIZE = 20000  # 每个分片包含的数据量
MIN_DURATION = 0

def get_all_pkl_files(directory):
    """递归查找所有符合条件的pkl文件"""
    pkl_files = []
    print(f"正在扫描目录: {directory} ...")
    # 使用 os.walk 递归遍历
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))
    print(f"找到 {len(pkl_files)} 个符合条件的 .pkl 文件。")
    return pkl_files

def save_shard(data_buffer, shard_index, output_dir):
    """将缓存的数据保存为一个 arrow dataset 分片"""
    if not data_buffer["text_ids"]:
        return

    # 创建保存目录
    save_path = os.path.join(output_dir, f"{SOURCE_NAME}_part_{shard_index}")
    os.makedirs(save_path, exist_ok=True)
    
    # 从字典创建 Dataset (列式存储创建速度最快)
    ds = Dataset.from_dict(data_buffer)
    
    # 保存到磁盘
    ds.save_to_disk(save_path)
    print(f"已保存分片 part_{shard_index} (包含 {len(data_buffer['text_ids'])} 条数据) 到 {save_path}")

def generate_derangement(n):
    """
    生成一个长度为n的错排索引数组（Derangement）。
    保证 indices[i] != i，即每个位置的索引都不是它原本的位置。
    """
    indices = np.arange(n)

    if n < 2:
        return indices
    
    while True:
        np.random.shuffle(indices)
        # 检查是否有不动点 (Fixed point)
        if not np.any(indices == np.arange(n)):
            return indices

def main():
    # 确保输出目录存在
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    # 获取文件列表
    files = get_all_pkl_files(SOURCE_DIR)
    
    # 初始化缓存字典 (列式存储)
    buffer = {
        "text_ids": [],
        "codes": [],
        "text_len": [],
        "code_len": [],
        "condition": [],
        "emo_vec": [],
        "duration": []
    }
    
    shard_count = 0
    total_processed = 0

    # 进度条
    pbar = tqdm(files, desc="处理文件")
    
    for file_path in pbar:
        try:
            with open(file_path, "rb") as f:
                data_dict: Dict[str, List[ProcessedData]] = pickle.load(f)
            
            if not data_dict:
                continue

            # 遍历字典中的每个说话人，分别处理
            for speaker_id, data_list in data_dict.items():
                if not data_list:
                    continue

                # 1. 第一遍遍历：筛选有效数据并转换为 numpy
                valid_items = []
                for item in data_list:
                    if item.duration >= MIN_DURATION:
                        # 转换为 numpy
                        valid_items.append(item.to_numpy())
                
                count = len(valid_items)
                
                # 如果没有有效数据，跳过该说话人
                if count == 0:
                    continue
                
                # 2. 提取该说话人所有的 condition (保留引用)
                original_conditions = [item.condition for item in valid_items]
                
                # 3. 生成错排索引
                # 如果 count < 2，generate_derangement 会返回 [0]，即不进行交换（因为无法交换）
                shuffled_indices = generate_derangement(count)
                
                # 4. 第二遍遍历：应用打乱后的 condition 并存入 buffer
                for i, item in enumerate(valid_items):
                    # 获取新的 condition 来源索引
                    source_idx = shuffled_indices[i]
                    
                    # 关键步骤：替换 condition
                    # 这样确保了 condition 依然来自同一个 speaker，但来自不同的 utterance
                    item.condition = original_conditions[source_idx]
                    
                    # 添加到全局 buffer
                    buffer["text_ids"].append(item.text_ids)
                    buffer["codes"].append(item.codes)
                    buffer["text_len"].append(item.text_len)
                    buffer["code_len"].append(item.code_len)
                    buffer["condition"].append(item.condition)
                    buffer["emo_vec"].append(item.emo_vec)
                    buffer["duration"].append(item.duration)
                    
                    # 检查是否达到分片大小
                    if len(buffer["text_ids"]) >= SHARD_SIZE:
                        save_shard(buffer, shard_count, TARGET_DIR)
                        # 清空 buffer
                        for key in buffer:
                            buffer[key] = []
                        shard_count += 1
                
                total_processed += count
            
            pbar.set_postfix({"Items": total_processed, "Shards": shard_count})
            
        except Exception as e:
            # 打印详细错误栈以便调试
            import traceback
            traceback.print_exc()
            print(f"\n读取文件 {file_path} 时出错: {e}")

    # 处理剩余的数据
    if len(buffer["text_ids"]) > 0:
        save_shard(buffer, shard_count, TARGET_DIR)
        shard_count += 1

    print(f"\n处理完成！共保存了 {shard_count} 个分片，总计 {total_processed} 条数据。")
    print(f"数据已保存至: {TARGET_DIR}")

if __name__ == "__main__":
    main()