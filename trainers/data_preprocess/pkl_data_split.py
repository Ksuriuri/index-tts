import os
import pickle
from dataclasses import dataclass
import sys
from typing import Dict, Union, List
import torch
import numpy as np
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from trainers.utils import ProcessedData

split_mark = "split-pkl"

def split_pkl_files(root_dir, chunk_size=1000):
    # 1. 扫描文件
    files_to_process = []
    print(f"正在扫描目录: {root_dir} ...")
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".pkl"):
                # 跳过已经是切分过的文件
                if f"-{split_mark}-" in filename:
                    continue
                files_to_process.append(os.path.join(dirpath, filename))

    print(f"共发现 {len(files_to_process)} 个需要处理的 .pkl 文件。")

    # 2. 遍历处理
    for file_path in tqdm(files_to_process, desc="Processing Files"):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            if not isinstance(data, list):
                # print(f"警告: {file_path} 内容不是列表，跳过。")
                continue
            
            # --- 切分逻辑开始 ---
            chunks = []
            total_len = len(data)
            
            # 初步切分
            for i in range(0, total_len, chunk_size):
                chunks.append(data[i : i + chunk_size])
            
            # 处理尾部合并逻辑
            # 如果至少有2个块，且最后一个块的长度小于 chunk_size 的一半
            if len(chunks) >= 2:
                last_chunk_len = len(chunks[-1])
                threshold = chunk_size / 2
                
                if last_chunk_len < threshold:
                    # 将最后一个块的数据追加到倒数第二个块中
                    last_part = chunks.pop()
                    chunks[-1].extend(last_part)
            # --- 切分逻辑结束 ---

            # 获取基础路径信息
            dir_name = os.path.dirname(file_path)
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            # 保存所有块
            for i, chunk_data in enumerate(chunks):
                new_filename = f"{base_name}-{split_mark}-{i}.pkl"
                new_file_path = os.path.join(dir_name, new_filename)
                
                with open(new_file_path, 'wb') as f_out:
                    pickle.dump(chunk_data, f_out)
            
            # 如果需要处理完删除原文件，请取消下面注释
            # os.remove(file_path)

        except Exception as e:
            print(f"\n处理文件 {file_path} 时出错: {e}")

if __name__ == "__main__":
    TARGET_DIR = "/mnt/data_3t_2/datasets/indextts_train_data/Galgame-VisualNovel-Reupload"
    
    if os.path.exists(TARGET_DIR):
        # 默认 chunk_size=1000，即如果尾部少于 500，会合并到前一个
        split_pkl_files(TARGET_DIR, chunk_size=2000)
        print("\n处理完成！")
    else:
        print(f"错误: 目录不存在 - {TARGET_DIR}")