# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tanhe/miniconda3/envs/index-tts/lib/python3.10/site-packages/nvidia/cudnn/lib

import os
import io
import glob
import queue
import time
import json
import tarfile
import traceback
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Set, Optional

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
DATASET_NAME = "Japanese-Eroge-Voice"
DATASET_ROOT = f"/mnt/data_3t_1/datasets/raw_data/{DATASET_NAME}"
OUTPUT_DIR = f"/mnt/data_3t_1/datasets/preprocess/{DATASET_NAME}"
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
    speaker: Optional[str]
    source_file: str  

class AudioLoaderWorker(Process):
    def __init__(self, file_queue: Queue, gpu_task_queue: Queue, worker_id: int, checkpoint: Dict[str, int], file_pbar_counter):
        super().__init__(daemon=True)
        self.file_queue = file_queue
        self.gpu_task_queue = gpu_task_queue
        self.worker_id = worker_id
        self.checkpoint = checkpoint
        self.file_pbar_counter = file_pbar_counter

    def run(self):
        logger.info(f"[CPU-Loader-{self.worker_id}] Started.")
        current_batch = []
        
        AUDIO_EXTS = ['.mp3', '.wav', '.flac', '.ogg']

        while True:
            try:
                tar_path = self.file_queue.get(timeout=10)
            except queue.Empty:
                break
            
            if tar_path is None: break

            try:
                # 获取该文件已处理的样本数 (断点续传)
                skip_count = self.checkpoint.get(tar_path, 0)
                if skip_count > 0:
                    logger.info(f"[Loader-{self.worker_id}] {os.path.basename(tar_path)} resuming, skipping first {skip_count} samples")

                # 缓冲区，用于配对 key.txt 和 key.mp3/wav/flac
                sample_buffer = {}
                processed_count = 0 # 当前文件已提取的有效样本计数
                
                with tarfile.open(tar_path, "r") as tar:
                    for member in tar:
                        if not member.isfile():
                            continue
                        
                        file_name = os.path.basename(member.name)
                        if file_name.startswith("._"): continue # 跳过 Mac 系统缓存文件

                        key, ext = os.path.splitext(file_name)
                        ext = ext.lower() # 统一小写
                        
                        # 修改：过滤逻辑，寻找 .txt 和音频
                        if ext not in ['.txt'] and ext not in AUDIO_EXTS:
                            continue

                        if key not in sample_buffer:
                            sample_buffer[key] = {}
                        
                        try:
                            f = tar.extractfile(member)
                            if f is None: continue
                            
                            # 修改：处理 .txt 文本文件
                            if ext == '.txt':
                                try:
                                    content_bytes = f.read()
                                    text_content = content_bytes.decode('utf-8').strip()
                                    sample_buffer[key]['text'] = text_content
                                except Exception:
                                    pass # 文本解析失败忽略
                            elif ext in AUDIO_EXTS:
                                sample_buffer[key]['audio'] = f.read()
                        except Exception as e:
                            # 读取错误则清理 buffer
                            if key in sample_buffer: del sample_buffer[key]
                            continue
                        
                        # 修改：检查是否凑齐了一对 (Audio + Text)
                        if 'text' in sample_buffer[key] and 'audio' in sample_buffer[key]:
                            data_item = sample_buffer.pop(key)
                            
                            # 断点续传逻辑
                            if processed_count < skip_count:
                                processed_count += 1
                                continue
                            
                            text = data_item['text']
                            audio_bytes = data_item['audio']
                            
                            if not text or not audio_bytes: 
                                continue

                            try:
                                # 转换音频
                                with io.BytesIO(audio_bytes) as audio_io:
                                    # 注意：sf.read 会返回 float64 或 float32，这里转 float32
                                    array, sr = sf.read(audio_io, dtype='float32')
                                
                                # 检查时长
                                if MAX_AUDIO_DURATION > 0 and len(array) / sr > MAX_AUDIO_DURATION: 
                                    continue
                                
                                # 转单声道
                                if array.ndim > 1: array = np.mean(array, axis=1)
                                
                                task = ASRTask(
                                    audio_bytes=audio_bytes, 
                                    audio_raw=array.astype(np.float16), 
                                    sample_rate=sr, 
                                    text_gt=str(text),
                                    speaker=None,
                                    source_file=tar_path
                                )
                                current_batch.append(task)
                                processed_count += 1

                                if len(current_batch) >= BATCH_SIZE:
                                    self.gpu_task_queue.put(current_batch)
                                    current_batch = []
                                    
                            except Exception as e:
                                # 音频处理出错，跳过
                                continue
                
                # 文件处理完后，更新文件进度条
                with self.file_pbar_counter.get_lock():
                    self.file_pbar_counter.value += 1
                
                logger.info(f"[Loader-{self.worker_id}] Finished {os.path.basename(tar_path)}. Total extracted: {processed_count}")

            except Exception as e:
                logger.error(f"Error loading {tar_path}: {traceback.format_exc()}")
        
        # 处理剩余的 batch
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
            logger.error(f"Failed to load Whisper model: {e}")
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
                    audio_tensor = torch.from_numpy(audio_fp32).to(device_str)
                    
                    if task.sample_rate != 16000:
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
                        "speaker": task.speaker,
                        "whisper_large_v3": {
                            "text": text_pred,
                            "cer": float(error_rate),
                            "language": info.language,
                            "segments": segments_list
                        },
                        "_source_file": task.source_file
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
        # 记录当前 batch 中每个源文件贡献了多少条数据
        current_cycle_counts = {} 
        
        existing_parts = glob.glob(os.path.join(self.output_dir, "part_*.parquet"))
        if existing_parts:
            # 简单的文件序号递增
            indices = []
            for f in existing_parts:
                try:
                    idx = int(os.path.basename(f).split('_')[1].split('.')[0])
                    indices.append(idx)
                except: pass
            file_idx = max(indices) + 1 if indices else 0
        else:
            file_idx = 0
        
        # 样本进度条
        pbar = tqdm(desc="Samples Processed", unit="audios", dynamic_ncols=True)

        while True:
            try:
                data = self.result_queue.get(timeout=20)
                if data == "DONE": break
                
                src_file = data.pop("_source_file")
                current_cycle_counts[src_file] = current_cycle_counts.get(src_file, 0) + 1
                
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
        # PyArrow/Pandas 自动处理 None 为 Nullable String
        pd.DataFrame(data_list).to_parquet(save_path, engine='pyarrow', index=False)
        
        for src, count in cycle_counts.items():
            self.checkpoint[src] = self.checkpoint.get(src, 0) + count
        
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(self.checkpoint, f, indent=2, ensure_ascii=False)
        logger.info(f"Checkpoint saved. Part {idx} contains {len(data_list)} samples.")

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
    
    # 扫描 .tar 文件
    logger.info(f"Scanning .tar files in {DATASET_ROOT}...")
    all_tar_files = sorted(glob.glob(os.path.join(DATASET_ROOT, "**/*.tar"), recursive=True))
    total_files = len(all_tar_files)
    logger.info(f"Found {total_files} tar files.")
    
    if total_files == 0:
        logger.error(f"No .tar files found in {DATASET_ROOT}")
        return

    file_queue = Queue()
    for f in all_tar_files:
        file_queue.put(f)
    
    file_pbar_counter = Value('i', 0)
    
    # 适当增大队列
    gpu_task_queue = Queue(maxsize=DEVICE_NUM * 24) 
    result_queue = Queue()

    # 启动 Writer
    writer = ParquetWriterWorker(result_queue, OUTPUT_DIR, SAVE_INTERVAL, CHECKPOINT_PATH, checkpoint)
    writer.start()

    # 启动 GPU Workers
    gpu_workers = []
    for gpu_id in range(DEVICE_NUM):
        for w_id in range(PROCESSORS_PER_DEVICE):
            p = GPUASRWorker(gpu_task_queue, result_queue, gpu_id, gpu_id * PROCESSORS_PER_DEVICE + w_id)
            p.start()
            gpu_workers.append(p)

    # 启动 CPU Loaders
    cpu_workers = []
    for i in range(CPU_WORKERS_NUM):
        p = AudioLoaderWorker(file_queue, gpu_task_queue, i, checkpoint, file_pbar_counter)
        p.start()
        cpu_workers.append(p)

    # 主进程监控总文件进度
    with tqdm(total=total_files, desc="Files Progress", unit="file", dynamic_ncols=True) as fbar:
        last_val = 0
        while any(p.is_alive() for p in cpu_workers):
            curr_val = file_pbar_counter.value
            if curr_val > last_val:
                fbar.update(curr_val - last_val)
                last_val = curr_val
            time.sleep(1)
        fbar.update(file_pbar_counter.value - last_val)

    # 等待完成并清理
    for p in cpu_workers: p.join()
    logger.info("CPU Loaders finished.")
    
    for _ in gpu_workers: gpu_task_queue.put(None)
    for p in gpu_workers: p.join()
    logger.info("GPU Workers finished.")
    
    result_queue.put("DONE")
    writer.join()

if __name__ == "__main__":
    main()