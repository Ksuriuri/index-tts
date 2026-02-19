# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tanhe/miniconda3/envs/index-tts/lib/python3.10/site-packages/nvidia/cudnn/lib

import os
import io
import glob
import queue
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Set

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
DATASET_ROOT = "/mnt/data_3t_1/datasets/raw_data/Galgame-VisualNovel-Reupload"
OUTPUT_DIR = "/mnt/data_3t_1/datasets/preprocess/Galgame-VisualNovel-Reupload"
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
    audio_raw: np.ndarray  # 传输时将使用 float16
    sample_rate: int       
    text_gt: str
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

        while True:
            try:
                parquet_path = self.file_queue.get(timeout=10)
            except queue.Empty:
                break
            
            if parquet_path is None: break

            try:
                skip_count = self.checkpoint.get(parquet_path, 0)
                if skip_count > 0:
                    logger.info(f"[Loader-{self.worker_id}] {parquet_path} resuming from index {skip_count}")

                parquet_file = pq.ParquetFile(parquet_path)
                rows_read = 0
                
                for batch_data in parquet_file.iter_batches(batch_size=256, columns=['audio', 'text']):
                    batch_len = len(batch_data)
                    if rows_read + batch_len <= skip_count:
                        rows_read += batch_len
                        continue
                    
                    audio_col = batch_data['audio']
                    text_col = batch_data['text']
                    
                    for i in range(len(batch_data)):
                        current_row_idx = rows_read + i
                        if current_row_idx < skip_count:
                            continue

                        try:
                            text = str(text_col[i])
                            audio_struct = audio_col[i].as_py()
                            audio_bytes = audio_struct['bytes']
                            if not audio_bytes or not text: continue

                            with io.BytesIO(audio_bytes) as f:
                                array, sr = sf.read(f)
                            
                            if MAX_AUDIO_DURATION > 0 and len(array) / sr > MAX_AUDIO_DURATION: continue
                            if array.ndim > 1: array = np.mean(array, axis=1)
                            
                            task = ASRTask(
                                audio_bytes=audio_bytes, 
                                audio_raw=array.astype(np.float16), 
                                sample_rate=sr, 
                                text_gt=text,
                                source_file=parquet_path
                            )
                            current_batch.append(task)

                            if len(current_batch) >= BATCH_SIZE:
                                self.gpu_task_queue.put(current_batch)
                                current_batch = []
                        except Exception:
                            continue
                    
                    rows_read += batch_len
                
                # 文件处理完后，更新进度条计数器
                with self.file_pbar_counter.get_lock():
                    self.file_pbar_counter.value += 1
                
                logger.info(f"[Loader-{self.worker_id}] Finished {os.path.basename(parquet_path)}")

            except Exception as e:
                logger.error(f"Error loading {parquet_path}: {e}")
        
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
        model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", device_index=self.gpu_id, compute_type=COMPUTE_TYPE)
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
                        "speaker": None,
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
        current_cycle_counts = {} 
        
        existing_parts = glob.glob(os.path.join(self.output_dir, "part_*.parquet"))
        file_idx = max([int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing_parts]) + 1 if existing_parts else 0
        
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
    
    all_parquet_files = sorted(glob.glob(os.path.join(DATASET_ROOT, "**/*.parquet"), recursive=True))
    total_files = len(all_parquet_files)
    
    file_queue = Queue()
    for f in all_parquet_files:
        file_queue.put(f)
    
    # 共享变量用于更新文件进度条
    file_pbar_counter = Value('i', 0)
    
    gpu_task_queue = Queue(maxsize=DEVICE_NUM * 16) 
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
        # 最后补齐
        fbar.update(total_files - last_val)

    # 等待完成并清理
    for p in cpu_workers: p.join()
    for _ in gpu_workers: gpu_task_queue.put(None)
    for p in gpu_workers: p.join()
    result_queue.put("DONE")
    writer.join()

if __name__ == "__main__":
    main()