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

def save_shard(data_items: List[Dict], shard_index: int, output_dir: str):
    """
    处理 shuffling 并保存分片
    data_items: 包含 {'data': numpy_obj, 'speaker_id': str} 的列表
    """
    if not data_items:
        return

    # ================= 3. 说话人 Condition Shuffle 逻辑 =================
    # 按说话人分组
    speaker_groups = defaultdict(list)
    for idx, item in enumerate(data_items):
        spk = item['speaker_id']
        speaker_groups[spk].append(idx)

    # 遍历每个说话人组
    for spk, indices in speaker_groups.items():
        n_samples = len(indices)
        if n_samples < 2:
            continue

        # 注意：这里假设 condition 是 numpy array。如果维度一致，np.stack 速度很快。
        current_conditions = np.stack([data_items[i]['data'].condition for i in indices])
        
        deranged_idxs = generate_derangement(n_samples)
        shuffled_conditions = current_conditions[deranged_idxs]

        # 将打乱后的 condition 赋值回去
        for k, original_idx in enumerate(indices):
            data_items[original_idx]['data'].condition = shuffled_conditions[k]

    # 初始化缓存字典
    buffer = {
        "text_ids": [], "codes": [], "text_len": [], "code_len": [],
        "condition": [], "emo_vec": [], "duration": []
    }

    for item in data_items:
        np_obj = item['data']
        buffer["text_ids"].append(np_obj.text_ids)
        buffer["codes"].append(np_obj.codes)
        buffer["text_len"].append(np_obj.text_len)
        buffer["code_len"].append(np_obj.code_len)
        buffer["condition"].append(np_obj.condition)
        buffer["emo_vec"].append(np_obj.emo_vec)
        buffer["duration"].append(np_obj.duration)

    # 保存
    save_path = os.path.join(output_dir, f"part_{shard_index}")
    os.makedirs(save_path, exist_ok=True)
    ds = Dataset.from_dict(buffer)
    ds.save_to_disk(save_path)
    print(f"已保存分片 part_{shard_index} (包含 {len(data_items)} 条数据)")


def main():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    shard_count = 0
    total_processed = 0
    
    # 暂存待处理的数据项，攒够 SHARD_SIZE 后统一进行 shuffle 和保存
    # 结构: [{'data': numpy_obj, 'speaker_id': str}, ...]
    pending_items = []

    # 1. 遍历多个 Source
    for source_name in SOURCE_NAMES:
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

                    # ================= 3. 说话人 ID 逻辑 =================
                    raw_speaker = row.get("speaker", None)
                    if raw_speaker is not None:
                        # 相同的 speaker 字段即为同一说话人
                        spk_id = str(raw_speaker)
                    else:
                        # 若为 None，则 data_org_idx 相同的数据来源于同一个说话人
                        spk_id = f"{source_name}_idx_{parquet_idx}"

                    # 基础长度/时长过滤
                    if processed_data.duration < MIN_DURATION or processed_data.duration > MAX_DURATION or \
                       processed_data.text_len < MIN_TEXT_TOKENS or processed_data.text_len > MAX_TEXT_TOKENS:
                        continue
                    
                    processed_data_np = processed_data.to_numpy()
                    
                    # 加入待处理队列
                    pending_items.append({
                        'data': processed_data_np,
                        'speaker_id': spk_id
                    })

                    # 检查是否达到分片大小，进行 Shuffle 和保存
                    if len(pending_items) >= SHARD_SIZE:
                        save_shard(pending_items, shard_count, TARGET_DIR)
                        shard_count += 1
                        # 清空列表
                        pending_items = []
                    
                pbar.set_postfix({"Buff": len(pending_items), "Shards": shard_count})

            except Exception as e:
                print(f"\n处理文件 {file_path} 时出错: {traceback.format_exc()}")
            
            print(f"{file_path} 处理完成，跳过 cer: {cer_skip_num}, diarization: {diarization_skip_num}")

    # 处理剩余的数据
    if len(pending_items) > 0:
        save_shard(pending_items, shard_count, TARGET_DIR)
        shard_count += 1

    print(f"\n处理完成！共保存了 {shard_count} 个分片。")
    print(f"数据已保存至: {TARGET_DIR}")

if __name__ == "__main__":
    main()