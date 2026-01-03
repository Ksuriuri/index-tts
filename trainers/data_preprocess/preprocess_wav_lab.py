# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tanhe/miniconda3/envs/index-tts/lib/python3.10/site-packages/nvidia/cudnn/lib

import os
import io
import glob
import queue
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Set
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from jiwer import cer
from tqdm import tqdm
from loguru import logger
from torch.multiprocessing import Process, Queue, Value
import torch.multiprocessing as mp

# 使用 faster-whisper
from faster_whisper import WhisperModel

# 强制设置启动方式
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# --- 配置 ---
# 数据集根目录结构应为: DATASET_ROOT / 游戏名 / 角色名 / {音频.wav, 文本.lab}
LANGS = "jp"
DATASET_DIRS = [
    f"/mnt/data_3t_1/datasets/raw_data/Genshin_Dataset/{LANGS}",
    f"/mnt/data_3t_1/datasets/raw_data/StarRail_Dataset/{LANGS}",
    f"/mnt/data_3t_1/datasets/raw_data/WutheringWaves_Dataset/{LANGS}",
]
OUTPUT_DIR = f"/mnt/data_3t_1/datasets/preprocess/Gacha_games_{LANGS}"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "resume_checkpoint.json") 

WHISPER_MODEL_SIZE = "large-v3"
COMPUTE_TYPE = "float16" 
DEVICE_NUM = 8
PROCESSORS_PER_DEVICE = 1
CPU_WORKERS_NUM = 1
BATCH_SIZE = 16
SAVE_INTERVAL = 20000
MAX_AUDIO_DURATION = -1  # 36

@dataclass
class ASRTask:
    audio_bytes: bytes
    audio_raw: np.ndarray  # float16
    sample_rate: int       
    text_gt: str
    speaker: str           # 新增 speaker 字段
    source_key: str        # 用于断点续传的 key (例如: Game/Speaker)

class AudioLoaderWorker(Process):
    def __init__(self, dir_queue: Queue, gpu_task_queue: Queue, worker_id: int, checkpoint: Dict[str, int], sample_pbar_counter):
        super().__init__(daemon=True)
        self.dir_queue = dir_queue
        self.gpu_task_queue = gpu_task_queue
        self.worker_id = worker_id
        self.checkpoint = checkpoint
        self.sample_pbar_counter = sample_pbar_counter

    def run(self):
        logger.info(f"[CPU-Loader-{self.worker_id}] Started.")
        current_batch = []

        while True:
            try:
                # 获取任务: (游戏名, 角色名, 绝对路径)
                task_data = self.dir_queue.get(timeout=10)
            except queue.Empty:
                break
            
            if task_data is None: break
            
            game_name, speaker_name, dir_path = task_data
            # Checkpoint key 使用相对路径: Game/Speaker
            source_key = f"{game_name}/{speaker_name}"

            try:
                # 获取该目录下所有 wav 文件并排序 (保证顺序一致性)
                wav_files = sorted(glob.glob(os.path.join(dir_path, "*.wav")))
                total_in_dir = len(wav_files)
                
                # 检查断点
                skip_count = self.checkpoint.get(source_key, 0)
                if skip_count > 0:
                    if skip_count >= total_in_dir:
                        logger.info(f"[Loader-{self.worker_id}] Skipping {source_key} (completed).")
                        continue
                    logger.info(f"[Loader-{self.worker_id}] {source_key} resuming from index {skip_count}")

                processed_count = 0
                
                for idx, wav_path in enumerate(wav_files):
                    # 跳过已处理的
                    if idx < skip_count:
                        continue

                    processed_count += 1 # 记录本次处理的数量（不管成功失败都要计数，防止死循环）
                    
                    lab_path = os.path.splitext(wav_path)[0] + ".lab"
                    
                    # 检查 lab 文件是否存在
                    if not os.path.exists(lab_path):
                        continue

                    try:
                        # 1. 读取文本
                        with open(lab_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        
                        if not text: continue

                        # 2. 读取音频
                        # 使用 soundfile 读取，速度较快
                        with open(wav_path, 'rb') as f:
                            audio_bytes = f.read()
                            
                        # 重新用 BytesIO 读取一遍获取 numpy 数组 (或者直接用 sf.read(wav_path))
                        # 为了保持 audio_bytes 用于存储，这里分开操作
                        with io.BytesIO(audio_bytes) as buf:
                            array, sr = sf.read(buf)

                        # 时长过滤
                        if MAX_AUDIO_DURATION > 0 and len(array) / sr > MAX_AUDIO_DURATION:
                            continue
                        
                        # 转单声道
                        if array.ndim > 1: array = np.mean(array, axis=1)
                        
                        task = ASRTask(
                            audio_bytes=audio_bytes, 
                            audio_raw=array.astype(np.float16), # 传输使用 float16 节省带宽
                            sample_rate=sr, 
                            text_gt=text,
                            speaker=source_key,
                            source_key=source_key
                        )
                        current_batch.append(task)

                        # 攒够一个 Batch 发送给 GPU
                        if len(current_batch) >= BATCH_SIZE:
                            self.gpu_task_queue.put(current_batch)
                            current_batch = []
                        
                        # 更新全局进度条计数器 (仅用于显示)
                        with self.sample_pbar_counter.get_lock():
                            self.sample_pbar_counter.value += 1

                    except Exception as e:
                        logger.warning(f"Error processing {wav_path}: {e}")
                        continue
                
                # 如果最后还有剩余的 batch
                if current_batch:
                    self.gpu_task_queue.put(current_batch)
                    current_batch = []
                
                logger.info(f"[Loader-{self.worker_id}] Finished dir {source_key}")

            except Exception as e:
                logger.error(f"Error accessing dir {dir_path}: {e}")
        
        if current_batch:
            self.gpu_task_queue.put(current_batch)
        logger.info(f"[CPU-Loader-{self.worker_id}] Finished.")

class GPUASRWorker(Process):
    def __init__(self, input_queue: Queue, output_queue: Queue, gpu_id: int, worker_id: int):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.gpu_id = gpu_id
        self.worker_id = worker_id

    def run(self):
        device_str = f"cuda:{self.gpu_id}"
        logger.info(f"[GPU-Worker-{self.worker_id}] Loading model on {device_str}...")
        try:
            model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", device_index=self.gpu_id, compute_type=COMPUTE_TYPE)
        except Exception as e:
            logger.error(f"Failed to load Whisper model on {device_str}: {e}")
            return

        resamplers = {}

        while True:
            try:
                tasks = self.input_queue.get(timeout=30)
                if tasks is None: break
            except queue.Empty: continue

            for task in tasks:
                try:
                    audio_fp32 = task.audio_raw.astype(np.float32)
                    
                    # Resample logic
                    if task.sample_rate != 16000:
                        audio_tensor = torch.from_numpy(audio_fp32).to(device_str)
                        if task.sample_rate not in resamplers:
                            resamplers[task.sample_rate] = torchaudio.transforms.Resample(task.sample_rate, 16000).to(device_str)
                        audio_16k = resamplers[task.sample_rate](audio_tensor).cpu().numpy()
                    else:
                        audio_16k = audio_fp32 
                    
                    # 推理
                    segments_gen, info = model.transcribe(audio_16k, beam_size=1, vad_filter=True)
                    segments_list = [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()} for s in segments_gen]
                    text_pred = "".join([s["text"] for s in segments_list]).strip()
                    
                    gt_clean = task.text_gt.strip()
                    error_rate = cer(gt_clean, text_pred) if len(gt_clean) > 0 else 1.0
                    
                    self.output_queue.put({
                        "audio": task.audio_bytes,
                        "text": task.text_gt,
                        "speaker": task.speaker,  # 这里填入真实的 speaker
                        "whisper_large_v3": {
                            "text": text_pred,
                            "cer": float(error_rate),
                            "language": info.language,
                            "segments": segments_list
                        },
                        "_source_key": task.source_key # 用于 Checkpoint
                    })
                except Exception as e:
                    logger.error(f"Inference error on {device_str}: {e}")

class ParquetWriterWorker(Process):
    def __init__(self, result_queue: Queue, output_dir: str, save_interval: int, checkpoint_path: str, initial_checkpoint: dict):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.checkpoint_path = checkpoint_path
        self.checkpoint = initial_checkpoint

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        buffer = []
        current_cycle_counts = {} 
        
        # 查找下一个分片编号
        existing_parts = glob.glob(os.path.join(self.output_dir, "part_*.parquet"))
        if existing_parts:
            # 安全地提取编号
            indices = []
            for f in existing_parts:
                try:
                    name = os.path.basename(f)
                    idx = int(name.split('_')[1].split('.')[0])
                    indices.append(idx)
                except:
                    pass
            file_idx = max(indices) + 1 if indices else 0
        else:
            file_idx = 0
        
        pbar = tqdm(desc="Samples Written", unit="samples", dynamic_ncols=True)

        while True:
            try:
                data = self.result_queue.get(timeout=20)
                if data == "DONE": break
                
                # 提取用于 checkpoint 的 key
                source_key = data.pop("_source_key")
                current_cycle_counts[source_key] = current_cycle_counts.get(source_key, 0) + 1
                
                buffer.append(data)
                pbar.update(1)

                if len(buffer) >= self.save_interval:
                    self._save(buffer, file_idx, current_cycle_counts)
                    buffer, current_cycle_counts = [], {}
                    file_idx += 1
            except queue.Empty:
                if buffer:
                    self._save(buffer, file_idx, current_cycle_counts)
                    buffer, current_cycle_counts = [], {}
                    file_idx += 1
                continue

        if buffer:
            self._save(buffer, file_idx, current_cycle_counts)
        pbar.close()
        logger.info("Writer finished.")

    def _save(self, data_list, idx, cycle_counts):
        save_path = os.path.join(self.output_dir, f"part_{idx:04d}.parquet")
        try:
            pd.DataFrame(data_list).to_parquet(save_path, engine='pyarrow', index=False)
            
            for src, count in cycle_counts.items():
                self.checkpoint[src] = self.checkpoint.get(src, 0) + count
            
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {save_path} ({len(data_list)} samples).")
        except Exception as e:
            logger.error(f"Failed to save parquet {save_path}: {e}")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint = load_checkpoint()
    
    # 收集所有的 (Game, Speaker, FullPath) 任务
    speaker_tasks = []
    
    for game_dir in DATASET_DIRS:
        game_name = f"{os.path.basename(os.path.dirname(game_dir))}/{(os.path.basename(game_dir))}"
        # 扫描该游戏下的角色目录
        role_dirs = [d for d in glob.glob(os.path.join(game_dir, "*")) if os.path.isdir(d)]
        for role_dir in role_dirs:
            role_name = os.path.basename(role_dir)
            speaker_tasks.append((game_name, role_name, role_dir))

    logger.info(f"Found {len(speaker_tasks)} speaker directories.")
    
    # --- 2. 队列初始化 ---
    dir_queue = Queue()
    for task in speaker_tasks:
        dir_queue.put(task)
    
    gpu_task_queue = Queue(maxsize=DEVICE_NUM * 32) 
    result_queue = Queue()
    
    # 共享变量用于监控样本级进度 (Loader 读取数量)
    sample_pbar_counter = Value('i', 0)

    # --- 3. 启动进程 ---
    
    # 启动 Writer
    writer = ParquetWriterWorker(result_queue, OUTPUT_DIR, SAVE_INTERVAL, CHECKPOINT_PATH, checkpoint)
    writer.start()

    # 启动 GPU Workers
    gpu_workers = []
    for gpu_id in range(DEVICE_NUM):
        for w_id in range(PROCESSORS_PER_DEVICE):
            worker_global_id = gpu_id * PROCESSORS_PER_DEVICE + w_id
            p = GPUASRWorker(gpu_task_queue, result_queue, gpu_id, worker_global_id)
            p.start()
            gpu_workers.append(p)

    # 启动 CPU Loaders
    cpu_workers = []
    for i in range(CPU_WORKERS_NUM):
        p = AudioLoaderWorker(dir_queue, gpu_task_queue, i, checkpoint, sample_pbar_counter)
        p.start()
        cpu_workers.append(p)

    # --- 4. 监控进度 ---
    # 由于不知道具体的 wav 文件总数（扫描所有文件太慢），这里我们监控“已处理样本数”
    # 如果你想有一个总进度条，可以在扫描目录时顺便 glob 一下文件数，但这会增加启动时间。
    
    logger.info("Processing started...")
    try:
        with tqdm(desc="Total Samples Loaded", unit="sample", dynamic_ncols=True) as pbar:
            last_val = 0
            while any(p.is_alive() for p in cpu_workers):
                curr_val = sample_pbar_counter.value
                if curr_val > last_val:
                    pbar.update(curr_val - last_val)
                    last_val = curr_val
                time.sleep(1)
            # 补齐最后一点
            curr_val = sample_pbar_counter.value
            if curr_val > last_val:
                pbar.update(curr_val - last_val)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Stopping...")
        # 简单的退出逻辑，可能需要强制 kill
        for p in cpu_workers: p.terminate()
        for p in gpu_workers: p.terminate()
        writer.terminate()
        return

    logger.info("All loaders finished. Waiting for GPU workers...")

    # 等待完成并清理
    for p in cpu_workers: p.join()
    
    # 发送结束信号给 GPU Worker
    for _ in gpu_workers: gpu_task_queue.put(None)
    for p in gpu_workers: p.join()
    
    result_queue.put("DONE")
    writer.join()
    logger.info("All done.")

if __name__ == "__main__":
    main()