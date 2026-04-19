import os
import pickle
import sys
import traceback
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, Union, List, Any
from tqdm import tqdm
from collections import defaultdict
from tqdm.contrib.concurrent import process_map  # 核心并行库
from functools import partial

# 假设原始项目结构保留
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from trainers.utils import ProcessedData

# ================= 配置区域 =================
PREPROCESS_ROOT = "/mnt/data_3t_1/datasets/preprocess"
DATA_ROOT = "/mnt/data_3t_2/datasets/indextts_train_data_v2"

SOURCE_NAMES = {
    # jp
    "Emilia_JA": 0.15,
    "Emilia-YODAS_JA": 0.15,
    "Gacha_games_jp": 0.20,
    # synthesis
    "Galgame-VisualNovel-Reupload": 0.10,
    "Japanese-Eroge-Voice": 0.10,

    # # es
    # "google-chilean-spanish": 0.20,
    # "MLS_Spanish": 0.20,
    # "voxpopuli": 0.20,
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
    
    # # es
    # "google-chilean-spanish": (0.0, 0.7),
    # "MLS_Spanish": (0.0, 0.7),
    # "voxpopuli": (0.0, 0.7),
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

# 这些 source 在统计时若某个 speaker_id 仅出现 1 条样本，则丢弃该样本
# （与 convert_to_arrow_multi.py 中 REQUIRE_MULTI_SAMPLE_SOURCES 对齐；
#   注意此处采用 per-source 全局粒度，convert 脚本是 per-shard 粒度，统计值
#   作为 convert 结果的上界参考。）
REQUIRE_MULTI_SAMPLE_SOURCES: List[str] = [
    "Emilia_JA",
    "Emilia-YODAS_JA",
    "Gacha_games_jp",
]

# 并行相关配置
MAX_WORKERS = 8  # os.cpu_count()  # 使用所有 CPU 核心，也可以手动指定如 16

def get_parquet_path(pkl_path: str, source_name: str) -> str:
    try:
        path_parts = pkl_path.split(os.sep)
        if source_name not in path_parts:
             return None
        idx = path_parts.index(source_name)
        rel_path = os.sep.join(path_parts[idx+1:])
        rel_path = rel_path.replace('.pkl', '.parquet')
        parquet_path = os.path.join(PREPROCESS_ROOT, source_name, rel_path)
        return parquet_path
    except ValueError:
        return None

def get_all_pkl_files(directory):
    """递归查找所有符合条件的pkl文件"""
    pkl_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))
    return pkl_files

def process_single_file(args):
    """
    处理单个文件的逻辑，用于并行调用
    Args:
        args: tuple (file_path, source_name, cer_threshold, cer_type)
    Returns:
        dict: 包含统计信息的字典
    """
    file_path, source_name, cer_threshold, cer_type = args
    
    # 统计变量初始化
    stats = {
        "source_name": source_name,
        "valid_count": 0,
        "valid_duration": 0.0,
        "cer_skip": 0,
        "silence_skip": 0,
        "diarization_skip": 0,
        # 仅 REQUIRE_MULTI_SAMPLE_SOURCES 中的 source 才填充：
        # speaker_id -> [count, duration_sum]
        "speaker_stats": defaultdict(lambda: [0, 0.0]),
        "error": None
    }
    track_speaker = source_name in REQUIRE_MULTI_SAMPLE_SOURCES

    try:
        # 1. 获取 Parquet 路径
        parquet_path = get_parquet_path(file_path, source_name)
        if not parquet_path or not os.path.exists(parquet_path):
            # 这里的 print 在多进程中可能会乱序，尽量减少
            return stats 

        # 2. 读取 Parquet
        try:
            pf = pq.ParquetFile(parquet_path)
            meta_len = pf.metadata.num_rows
            batch_columns = ["whisper_large_v3", "speaker", "speaker_diarization"]
            batch_iterator = pf.iter_batches(batch_size=256, columns=batch_columns)
        except Exception as e:
            stats["error"] = f"Parquet Error: {e}"
            return stats

        # 3. 读取 Pickle
        with open(file_path, "rb") as f:
            data_list: list[dict] = pickle.load(f)

        if not data_list:
            return stats

        # 确保按 index 升序，以便与 iter_batches 顺序遍历匹配
        data_list.sort(key=lambda x: x["index"])

        current_batch = None
        batch_start = 0
        batch_end = 0

        data_idx = 0
        num_data = len(data_list)

        # 4. 遍历数据项
        while data_idx < num_data:
            item = data_list[data_idx]
            processed_data: ProcessedData = item["data"]
            parquet_idx = item["index"]
            
            # 确保索引不越界
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
            
            # --- 过滤逻辑开始 ---
            
            # A. CER 过滤
            cer = whisper_info.get(cer_type, 1.0)
            if cer > cer_threshold:
                stats["cer_skip"] += 1
                continue
            
            # B. 尾部静音过滤
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
                    stats["silence_skip"] += 1
                    continue
            
            # C. 说话人 Diarization 过滤
            unique_speakers = set(seg.get('speaker', '') if isinstance(seg, dict) else seg['speaker'] for seg in speaker_diar)
            if len(unique_speakers) != 1:
                stats["diarization_skip"] += 1
                continue

            # D. 基础长度/时长过滤
            if (processed_data.duration < MIN_DURATION or 
                processed_data.duration > MAX_DURATION or 
                processed_data.text_len < MIN_TEXT_TOKENS or 
                processed_data.text_len > MAX_TEXT_TOKENS):
                continue

            # --- 统计有效数据 ---
            if track_speaker:
                if raw_speaker is not None:
                    spk_id = str(raw_speaker)
                else:
                    spk_id = f"{source_name}_idx_{parquet_idx}"
                entry = stats["speaker_stats"][spk_id]
                entry[0] += 1
                entry[1] += processed_data.duration
            else:
                stats["valid_count"] += 1
                stats["valid_duration"] += processed_data.duration

    except Exception as e:
        stats["error"] = f"Process Error: {traceback.format_exc()}"

    # defaultdict 在跨进程序列化时无 lambda factory，转回普通 dict
    stats["speaker_stats"] = dict(stats["speaker_stats"])
    return stats

def main():
    # 1. 收集所有任务
    all_tasks = []
    print("正在扫描文件列表...")
    
    for source_name, cer_threshold in SOURCE_NAMES.items():
        source_dir = os.path.join(DATA_ROOT, source_name)
        if not os.path.exists(source_dir):
            print(f"跳过不存在的目录: {source_dir}")
            continue

        cer_type = SOURCE_CER_TYPES.get(source_name, CER_TYPE)
        files = get_all_pkl_files(source_dir)
        print(f"Source: {source_name}, Files: {len(files)}, CER_TYPE: {cer_type}")
        
        # 将任务打包成 tuple: (file_path, source_name, cer_threshold, cer_type)
        for f in files:
            all_tasks.append((f, source_name, cer_threshold, cer_type))

    print(f"总任务数: {len(all_tasks)}")
    
    if not all_tasks:
        return

    # 2. 并行处理
    # chunksize=1 表示每个进程每次领1个任务。
    # 如果文件非常多且处理极快，可以适当调大 chunksize (例如 10) 以减少通信开销
    results = process_map(
        process_single_file, 
        all_tasks, 
        max_workers=MAX_WORKERS, 
        chunksize=1, 
        desc="Parallel Processing"
    )

    # 3. 聚合结果
    final_num_dict = defaultdict(int)
    final_duration_dict = defaultdict(float)
    # 仅 REQUIRE_MULTI_SAMPLE_SOURCES 用：source -> speaker_id -> [count, duration]
    speaker_aggr: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    # 记录每个 source 由于 singleton 被丢弃的样本数与时长
    singleton_drop_count: Dict[str, int] = defaultdict(int)
    singleton_drop_duration: Dict[str, float] = defaultdict(float)

    # 错误日志聚合
    error_logs = []

    print("\n正在聚合统计结果...")
    for res in results:
        src = res["source_name"]

        # 记录错误
        if res["error"]:
            error_logs.append(res["error"])

        final_num_dict[src] += res["valid_count"]
        final_duration_dict[src] += res["valid_duration"]

        # 汇总 per-speaker 统计
        for spk_id, (cnt, dur) in res.get("speaker_stats", {}).items():
            entry = speaker_aggr[src][spk_id]
            entry[0] += cnt
            entry[1] += dur

    # 对 REQUIRE_MULTI_SAMPLE_SOURCES：丢弃 speaker 总样本数 < 2 的样本
    for src in REQUIRE_MULTI_SAMPLE_SOURCES:
        for spk_id, (cnt, dur) in speaker_aggr.get(src, {}).items():
            if cnt < 2:
                singleton_drop_count[src] += cnt
                singleton_drop_duration[src] += dur
                continue
            final_num_dict[src] += cnt
            final_duration_dict[src] += dur

    # 4. 输出报告
    print("\n" + "="*30)
    print("处理完成报告")
    print("="*30)
    
    if error_logs:
        print(f"警告: 出现了 {len(error_logs)} 个文件处理错误 (仅显示前5个):")
        for err in error_logs[:5]:
            print(err)
        print("-" * 20)

    print(f"{'Source Name':<30} | {'Count':<10} | {'Hours':<10} | {'DropSingleton':<14}")
    print("-" * 76)

    total_duration_all = 0
    total_count_all = 0

    for source_name in SOURCE_NAMES:
        count = final_num_dict[source_name]
        dur_hours = final_duration_dict[source_name] / 3600
        total_count_all += count
        total_duration_all += final_duration_dict[source_name]

        drop_str = ""
        if source_name in REQUIRE_MULTI_SAMPLE_SOURCES:
            drop_str = f"{singleton_drop_count[source_name]} ({singleton_drop_duration[source_name]/3600:.2f}h)"

        print(f"{source_name:<30} | {count:<10} | {dur_hours:<10.2f} | {drop_str:<14}")

    print("-" * 76)
    print(f"{'TOTAL':<30} | {total_count_all:<10} | {(total_duration_all / 3600):.2f} h")

if __name__ == "__main__":
    # Windows/macOS 下 multiprocessing 需要这个保护，Linux 下建议也保留
    main()