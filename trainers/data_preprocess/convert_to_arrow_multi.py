import os
import pickle
import sys
import random
import traceback
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, List, Any, Optional
from datasets import Dataset
from tqdm import tqdm
from collections import defaultdict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# ================= 配置区域 (保持不变) =================

PREPROCESS_ROOT = "/mnt/data_3t_1/datasets/preprocess"
DATA_ROOT = "/mnt/data_3t_2/datasets/indextts_train_data_v2"
TARGET_DIR = f"/mnt/data_3t_2/datasets/final_train_data/jp_es_260228"

SOURCE_NAMES = {
    # jp
    "Emilia_JA": 0.0,
    "Emilia-YODAS_JA": 0.0,
    "Gacha_games_jp": 0.10,
    # synthesis
    "Galgame-VisualNovel-Reupload": 0.0,
    "Japanese-Eroge-Voice": 0.01,

    # es
    "google-chilean-spanish": 0.20,
    "MLS_Spanish": 0.20,
    "voxpopuli": 0.20,
}

# 每个 source 单独配置尾部静音过滤范围 (min_sec, max_sec)，只有在此 dict 中的 source 才启用过滤
END_SILENCE_FILTER: Dict[str, tuple[float, float]] = {
    # jp
    "Emilia_JA": (0.1, 0.7),
    "Emilia-YODAS_JA": (0.1, 0.7),
    "Gacha_games_jp": (0.1, 0.7),
    # synthesis
    "Galgame-VisualNovel-Reupload": (0.1, 0.7),
    "Japanese-Eroge-Voice": (0.1, 0.7),
    # es
    "google-chilean-spanish": (0.0, 0.7),
    "MLS_Spanish": (0.0, 0.7),
    "voxpopuli": (0.0, 0.7),
}

SHARD_SIZE = 40000 
MIN_DURATION = 0
MAX_DURATION = 36
MIN_TEXT_TOKENS = 1
MAX_TEXT_TOKENS = 600
CER_TYPE = "cer"  # 默认 CER 类型，可被 SOURCE_CER_TYPES 覆盖
# CER_TYPE = "pron_CER"
# 每个 source 可单独指定 CER 类型，未在此处的 source 使用上面的 CER_TYPE
SOURCE_CER_TYPES: Dict[str, str] = {
    "Emilia_JA": "pron_CER",
    "Emilia-YODAS_JA": "pron_CER",
    "Gacha_games_jp": "pron_CER",
    "Galgame-VisualNovel-Reupload": "pron_CER",
    "Japanese-Eroge-Voice": "pron_CER",
}

# 并行配置
NUM_WORKERS = 8  # max(1, multiprocessing.cpu_count() - 4) # 预留核心给系统和Saver
MAX_PENDING_SAVES = 2 # 限制后台同时保存的分片数，防止内存爆炸

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from trainers.utils import ProcessedData

# ================= 工具函数 =================

def get_parquet_path(pkl_path: str, source_name: str) -> str:
    try:
        path_parts = pkl_path.split(os.sep)
        idx = path_parts.index(source_name)
        rel_path = os.sep.join(path_parts[idx+1:])
        rel_path = rel_path.replace('.pkl', '.parquet')
        parquet_path = os.path.join(PREPROCESS_ROOT, source_name, rel_path)
        return parquet_path
    except ValueError:
        return None

def generate_derangement(n):
    indices = np.arange(n)
    if n < 2: return indices
    while True:
        np.random.shuffle(indices)
        if not np.any(indices == np.arange(n)):
            return indices

def process_single_file(args):
    """
    这是 Worker 进程执行的函数。
    读取一个文件，返回过滤后的 valid_items 列表。
    """
    file_path, source_name, cer_threshold, cer_type = args
    valid_items = []
    stats = {"cer_skip": 0, "diar_skip": 0, "not_silence_skip": 0, "processed": 1}

    try:
        parquet_path = get_parquet_path(file_path, source_name)
        if not parquet_path or not os.path.exists(parquet_path):
            return [], stats

        try:
            pf = pq.ParquetFile(parquet_path)
            meta_len = pf.metadata.num_rows
            batch_iterator = pf.iter_batches(batch_size=256, columns=["whisper_large_v3", "speaker", "speaker_diarization"])
        except Exception:
            return [], stats

        with open(file_path, "rb") as f:
            data_list = pickle.load(f)

        if not data_list:
            return [], stats

        # 确保按 index 升序，以便与 iter_batches 顺序遍历匹配
        data_list.sort(key=lambda x: x["index"])

        current_batch = None
        batch_start = 0
        batch_end = 0

        data_idx = 0
        num_data = len(data_list)

        while data_idx < num_data:
            item = data_list[data_idx]
            parquet_idx = item["index"]
            processed_data: ProcessedData = item["data"]
            
            # 安全检查索引越界
            if parquet_idx >= meta_len:
                data_idx += 1
                continue
                
            while parquet_idx >= batch_end:
                try:
                    batch = next(batch_iterator)
                    current_batch = batch.to_pydict()
                    batch_start = batch_end
                    batch_end += batch.num_rows
                except StopIteration:
                    break
                    
            if parquet_idx >= batch_end:
                break
                
            local_idx = parquet_idx - batch_start
            
            # 从 batch 字典获取对应行，处理可能缺少的列
            whisper_info = current_batch.get("whisper_large_v3", [{}])[local_idx] if "whisper_large_v3" in current_batch else {}
            speaker_diar = current_batch.get("speaker_diarization", [[]])[local_idx] if "speaker_diarization" in current_batch else []
            raw_speaker = current_batch.get("speaker", [None])[local_idx] if "speaker" in current_batch else None
            
            data_idx += 1

            # CER 过滤
            cer = whisper_info.get(cer_type, 1.0)
            if cer > cer_threshold:
                stats["cer_skip"] += 1
                continue

            # Diarization 过滤
            unique_speakers = set(seg.get('speaker', '') if isinstance(seg, dict) else seg['speaker'] for seg in speaker_diar)
            if len(unique_speakers) != 1:
                stats["diar_skip"] += 1
                continue
            
            # 尾部静音过滤
            if source_name in END_SILENCE_FILTER:
                end_silence_min, end_silence_max = END_SILENCE_FILTER[source_name]
                total_duration = processed_data.duration
                segments = list(whisper_info.get("segments", []))
                
                skip_flag = True
                if segments:
                    last_seg_end = segments[-1]["end"]
                    tail_gap = total_duration - last_seg_end
                    if end_silence_min <= tail_gap <= end_silence_max:
                        skip_flag = False
                if skip_flag:
                    stats["not_silence_skip"] += 1
                    continue

            # Speaker ID 生成
            if raw_speaker is not None:
                spk_id = str(raw_speaker)
            else:
                spk_id = f"{source_name}_idx_{parquet_idx}"

            # 时长/长度过滤
            if (processed_data.duration < MIN_DURATION or processed_data.duration > MAX_DURATION or
                processed_data.text_len < MIN_TEXT_TOKENS or processed_data.text_len > MAX_TEXT_TOKENS):
                continue

            processed_data_np = processed_data.to_numpy()

            valid_items.append({
                'data': processed_data_np,
                'speaker_id': spk_id
            })

    except Exception:
        # 捕获异常防止进程崩溃
        return [], stats

    return valid_items, stats

def save_shard_task(data_items: List[Dict], shard_index: int, output_dir: str):
    """
    这是 Saver 进程执行的函数。
    包含复杂的 Shuffle 逻辑和 IO 写盘。
    """
    if not data_items:
        return f"Shard {shard_index} is empty"

    try:
        # --- Speaker Condition Shuffle 逻辑 ---
        speaker_groups = defaultdict(list)
        for idx, item in enumerate(data_items):
            speaker_groups[item['speaker_id']].append(idx)

        for spk, indices in speaker_groups.items():
            n_samples = len(indices)
            if n_samples < 2: continue

            # 提取 condition
            current_conditions = np.stack([data_items[i]['data'].condition for i in indices])
            
            # 错排
            deranged_idxs = generate_derangement(n_samples)
            shuffled_conditions = current_conditions[deranged_idxs]

            # 赋值回原处
            for k, original_idx in enumerate(indices):
                data_items[original_idx]['data'].condition = shuffled_conditions[k]

        # --- 构建 Dataset ---
        buffer = defaultdict(list)
        # 显式列出字段，比 append 更快一点
        keys = ["text_ids", "codes", "text_len", "code_len", "condition", "emo_vec", "duration"]
        
        for item in data_items:
            np_obj = item['data']
            # 假设 np_obj 有这些属性
            buffer["text_ids"].append(np_obj.text_ids)
            buffer["codes"].append(np_obj.codes)
            buffer["text_len"].append(np_obj.text_len)
            buffer["code_len"].append(np_obj.code_len)
            buffer["condition"].append(np_obj.condition)
            buffer["emo_vec"].append(np_obj.emo_vec)
            buffer["duration"].append(np_obj.duration)

        save_path = os.path.join(output_dir, f"part_{shard_index}")
        os.makedirs(save_path, exist_ok=True)
        
        ds = Dataset.from_dict(buffer)
        ds.save_to_disk(save_path)
        
        return f"Done Shard {shard_index}: {len(data_items)} items"
    
    except Exception as e:
        traceback.print_exc()
        return f"Error in Shard {shard_index}: {str(e)}"

# ================= 主程序 =================

def get_all_pkl_files(directory):
    pkl_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))
    return pkl_files

def main():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    # 1. 收集所有任务
    all_tasks = []
    for source_name, cer_threshold in SOURCE_NAMES.items():
        source_dir = os.path.join(DATA_ROOT, source_name)
        if not os.path.exists(source_dir):
            continue
        cer_type = SOURCE_CER_TYPES.get(source_name, CER_TYPE)
        files = get_all_pkl_files(source_dir)
        print(f"Source: {source_name}, Files: {len(files)}, CER_TYPE: {cer_type}")
        # 将参数打包，方便 map 调用
        for f in files:
            all_tasks.append((f, source_name, cer_threshold, cer_type))

    # 打乱文件处理顺序，防止某种特定类型数据扎堆（可选）
    random.shuffle(all_tasks) 

    pending_items = []
    shard_count = 0
    save_futures = []
    
    # 2. 启动 Saver 线程池 (ProcessPoolExecutor 避免 GIL 锁住数据序列化)
    # max_workers=2 意味着最多同时有2个分片在进行shuffle/write
    # 如果内存紧张，改为 1
    saver_executor = ProcessPoolExecutor(max_workers=2)

    # 3. 启动 Reader 进程池
    print(f"Starting processing with {NUM_WORKERS} reader workers...")
    
    # 使用 imap_unordered 提高响应速度，谁处理完就先返回谁
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        pbar = tqdm(
            pool.imap_unordered(process_single_file, all_tasks, chunksize=10),
            total=len(all_tasks),
            desc="文件处理中",
            unit="文件",
        )
        for result_items, stats in pbar:
            if result_items:
                pending_items.extend(result_items)

            # 4. 检查是否需要保存分片
            while len(pending_items) >= SHARD_SIZE:
                # 切片取出 SHARD_SIZE 个数据
                shard_data = pending_items[:SHARD_SIZE]
                pending_items = pending_items[SHARD_SIZE:]
                
                # 提交给后台保存，不阻塞当前循环
                future = saver_executor.submit(save_shard_task, shard_data, shard_count, TARGET_DIR)
                save_futures.append(future)
                shard_count += 1
                
                # 内存保护：如果后台积压了太多保存任务，稍微等一下
                # 清理已完成的任务
                save_futures = [f for f in save_futures if not f.done()]
                if len(save_futures) >= MAX_PENDING_SAVES:
                    pbar.set_description(f"Waiting for save (Backlog: {len(save_futures)})...")
                    # 等待最早的一个任务完成
                    save_futures[0].result() 
                    pbar.set_description("Processing")

            # 更新进度条信息
            pbar.set_postfix({
                "Buff": len(pending_items), 
                "Shards": shard_count,
                "Backlog": len(save_futures)
            })

            print(f"Processed done, stats: {stats}")

    # 5. 处理剩余数据
    if len(pending_items) > 0:
        print(f"Saving final shard {shard_count} with {len(pending_items)} items...")
        future = saver_executor.submit(save_shard_task, pending_items, shard_count, TARGET_DIR)
        save_futures.append(future)
        shard_count += 1

    # 6. 等待所有保存任务完成
    print("Waiting for background saves to finish...")
    for f in save_futures:
        print(f.result())

    saver_executor.shutdown()
    print(f"\n全部完成！共保存 {shard_count} 个分片。")

if __name__ == "__main__":
    # 设置 start method，防止 cuda 初始化冲突或 fork 问题
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()