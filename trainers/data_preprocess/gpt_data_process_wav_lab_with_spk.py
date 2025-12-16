from dataclasses import dataclass
import os
from pathlib import Path
import random
import sys
import pickle
import glob
import threading
import time
import traceback
from typing import Dict, List
import librosa
import numpy as np
from tqdm import tqdm
from loguru import logger
import queue
import pyarrow.parquet as pq
import io
import torch
import torchaudio
import soundfile as sf
from torch.multiprocessing import Process, Queue
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
mp.set_sharing_strategy('file_system')

# 原始路径设置
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)


from trainers.data_preprocess.gpt_data_process_worker import DataPreprocessor, DataPreprocessorReqData


random.seed(42)
LANGS = "jp"
DATASET_DIRS = [
    f"/mnt/data_3t_1/datasets/raw_data/Genshin_Dataset/{LANGS}",
    f"/mnt/data_3t_1/datasets/raw_data/StarRail_Dataset/{LANGS}",
    f"/mnt/data_3t_1/datasets/raw_data/WutheringWaves_Dataset/{LANGS}",
]
OUTPUT_DIR = f"/mnt/data_3t_2/datasets/indextts_train_data/Gacha_games_{LANGS}"
MODEL_DIR = "./checkpoints/IndexTTS-2-vLLM"
TARGET_SR = 16000
CPU_WORKERS_NUM = 1  # 负责读取和解码的CPU进程数
DEVICE_NUM = 8
PROCESSORS_PER_DEVICE = 1
MAX_GPU_TASK_QUEUE_SIZE = 64  # 限制队列大小防止内存爆炸
MAX_AUDIO_DURATION = 36
BATCH_SIZE = 12



@dataclass
class FileManifest:
    """告诉Writer某个文件预计有多少条数据"""
    output_path: str
    total_samples: int
    

class AudioLoaderWorker(Process):
    def __init__(self, file_queue: Queue, gpu_task_queue: Queue, writer_control_queue: Queue, worker_id: int):
        super().__init__(daemon=True)
        self.file_queue = file_queue
        self.gpu_task_queue = gpu_task_queue
        self.writer_control_queue = writer_control_queue
        self.worker_id = worker_id

    def run(self):
        logger.info(f"[CPU-Loader-{self.worker_id}] Started.")
        batch_req = []
        while True:
            try:
                input_dirs, output_path = self.file_queue.get(timeout=3)
            except queue.Empty:
                break # 队列空了，退出

            try:
                if os.path.exists(output_path):
                    logger.error(f"[CPU-Loader-{self.worker_id}] Skipping {output_path} as it already exists.")
                    continue

                search_pattern = os.path.join(input_dirs, "*.wav")
                wav_files = glob.glob(search_pattern, recursive=True)

                valid_count = 0
                duration_skip_num = 0
                non_audio_or_text_skip_num = 0
                pf_start_time = time.time()
                for wav_path in tqdm(wav_files, desc=f"CPU-Loader-{self.worker_id}: {output_path}"):
                    lab_path = os.path.splitext(wav_path)[0] + ".lab"
        
                    if not os.path.exists(lab_path):
                        non_audio_or_text_skip_num += 1
                        continue

                    # 1. 读取文本
                    with open(lab_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    if not text:
                        non_audio_or_text_skip_num += 1
                        continue

                    try:
                        # 音频解码
                        array, sampling_rate = librosa.load(wav_path, sr=TARGET_SR)
                        duration = array.shape[0] / sampling_rate
                        if duration > MAX_AUDIO_DURATION:
                            duration_skip_num += 1
                            continue
                        
                        req = DataPreprocessorReqData(
                            text=text,
                            audio=array, # float32 numpy array
                            orig_sr=sampling_rate,
                            file_rel_path=output_path,
                        )
                        batch_req.append(req)
                        valid_count += 1

                        if len(batch_req) >= BATCH_SIZE:
                            # if self.gpu_task_queue.empty():
                            #     logger.error(f"[CPU-Loader-{self.worker_id}] gpu_task_queue is empty.")
                            self.gpu_task_queue.put(batch_req)
                            batch_req = []
                        # self.gpu_task_queue.put(req)

                    except Exception as e:
                        logger.error(f"Error processing audio in {output_path}: {traceback.format_exc()}")
                        continue
                logger.error(f"[CPU-Loader-{self.worker_id}: {output_path}] Samples: {valid_count}. \
                             Duration skip: {duration_skip_num}, non_audio_or_text skip: {non_audio_or_text_skip_num}, pf time: {time.time()-pf_start_time:.4f}s")
                
                # 发送清单给 Writer
                if valid_count > 0:
                    manifest = FileManifest(
                        output_path=output_path,
                        total_samples=valid_count,
                    )
                    self.writer_control_queue.put(manifest)
                            
                # 发送剩余的 batch
                if len(batch_req) > 0:
                    self.gpu_task_queue.put(batch_req)
                    batch_req = []
                
                # logger.debug(f"[CPU-Loader-{self.worker_id}] Processed {rel_path}: {valid_count} samples.")

            except Exception as e:
                logger.error(f"[CPU-Loader-{self.worker_id}] File Error {output_path}: {traceback.format_exc()}")


class ResultWriterWorker(Process):
    def __init__(self, result_queue: Queue, writer_control_queue: Queue, total_files: int):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.writer_control_queue = writer_control_queue
        self.total_files = total_files

    def run(self):
        logger.info("[Writer] Started.")
        
        self.file_buffers = {}
        finished_files_count = 0
        
        # 进度条
        pbar = tqdm(total=self.total_files, unit="file", desc="Processing Parquet Files")
        
        while finished_files_count < self.total_files:
            # 优先处理控制信息（新文件注册）
            while not self.writer_control_queue.empty():
                try:
                    manifest: FileManifest = self.writer_control_queue.get_nowait()
                    if manifest.output_path not in self.file_buffers:
                        self.file_buffers[manifest.output_path] = {
                            'expected': manifest.total_samples, # 未知
                            'data': [],
                            'received': 0,
                        }
                    else:
                        self.file_buffers[manifest.output_path]['expected'] = manifest.total_samples
                except queue.Empty:
                    break

            # 处理处理结果
            try:
                res = self.result_queue.get(timeout=0.1)
                for res_ in res:
                    output_path, processed_data = res_
                    
                    if output_path not in self.file_buffers:
                        self.file_buffers[output_path] = {
                            'expected': -1, # 未知
                            'data': [],
                            'received': 0,
                        }
                    
                    self.file_buffers[output_path]['data'].append(processed_data)
                    self.file_buffers[output_path]['received'] += 1
                    
                    # 检查是否完成
                    buf = self.file_buffers[output_path]
                    if buf['expected'] != -1 and buf['received'] >= buf['expected']:  # * 0.9
                        try:
                            self.save_file(output_path, buf['data'])
                        finally:
                            del self.file_buffers[output_path]
                            finished_files_count += 1
                            pbar.update(1)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[Writer] Error: {traceback.format_exc()}")
                traceback.print_exc()

        pbar.close()
        logger.info("[Writer] All files processed.")

    def save_file(self, output_path, data_list):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, "wb") as f:
                pickle.dump(data_list, f)
            # logger.info(f"Saved {output_path}")
        except Exception as e:
            logger.error(f"Failed to save {output_path}: {traceback.format_exc()}")


def main():
    if os.path.exists(OUTPUT_DIR):
        logger.warning(f"Output directory {OUTPUT_DIR} already exists.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 扫描文件
    logger.info("Scanning files...")
    all_role_dirs = []
    for role_datas in DATASET_DIRS:
        game_name = Path(role_datas).parts[-2]
        role_dirs = [str(role_dir) for role_dir in Path(role_datas).iterdir() if role_dir.is_dir()]
        save_paths = [os.path.join(OUTPUT_DIR, game_name, Path(role_dir).name + ".pkl") for role_dir in role_dirs]
        all_role_dirs.extend([(input_dirs, output_path) for input_dirs, output_path in zip(role_dirs, save_paths)])

    total_files = len(all_role_dirs)
    if total_files == 0:
        return

    # 2. 创建队列
    # Manager Queue 稍微慢一点，但在这种大量数据场景下比较稳定，
    # 或者使用 multiprocessing.Queue (Pipe based)
    file_queue = Queue()  # CPU loaders input
    gpu_task_queue = Queue(maxsize=MAX_GPU_TASK_QUEUE_SIZE) # GPU workers input (bounded to prevent OOM)
    writer_control_queue = Queue() # CPU -> Writer (Manifest)
    result_queue = Queue() # GPU -> Writer (Results)

    # 3. 填充任务队列
    for dir_tuple in all_role_dirs:
        file_queue.put(dir_tuple)

    # 启动 Writer
    writer = ResultWriterWorker(result_queue, writer_control_queue, total_files)
    writer.start()

    # 启动 GPU Workers
    gpu_workers = []
    for gpu_id in range(DEVICE_NUM):
        for w_id in range(PROCESSORS_PER_DEVICE):
            worker_idx = gpu_id * PROCESSORS_PER_DEVICE + w_id
            p = DataPreprocessor(
                model_dir=MODEL_DIR,
                input_queue=gpu_task_queue,
                output_queue=result_queue,
                worker_id=worker_idx,
                gpu_id=gpu_id
            )
            p.start()
            gpu_workers.append(p)
    logger.info(f"Started {len(gpu_workers)} GPU workers.")

    # 启动 CPU Loaders
    cpu_workers = []
    for i in range(CPU_WORKERS_NUM):
        p = AudioLoaderWorker(file_queue, gpu_task_queue, writer_control_queue, i)
        p.start()
        cpu_workers.append(p)
    logger.info(f"Started {len(cpu_workers)} CPU loaders.")

    # 5. 等待 CPU Loaders 完成
    for p in cpu_workers:
        p.join()
    logger.info("All CPU loaders finished.")

    # 6. 通知 GPU Workers 退出
    # 发送足够的 None 信号
    for _ in gpu_workers:
        gpu_task_queue.put(None)
    
    for p in gpu_workers:
        p.join()
    logger.info("All GPU workers finished.")

    # 7. 等待 Writer 完成
    writer.join()
    logger.info("Writer finished. Done.")

if __name__ == "__main__":
    main()
