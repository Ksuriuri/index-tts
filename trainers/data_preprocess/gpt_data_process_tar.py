from dataclasses import dataclass
import os
import random
import sys
import pickle
import glob
import threading
import time
import traceback
import json
import tarfile
from typing import Dict, List
import uuid
import numpy as np
from tqdm import tqdm
from loguru import logger
import queue
import io
import torch
import soundfile as sf
from torch.multiprocessing import Process, Queue
import torch.multiprocessing as mp

# 建议设置 start method
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
mp.set_sharing_strategy('file_system')

# 原始路径设置
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

# 请确保 gpt_data_process_worker.py 已经应用了上面的修改
from trainers.data_preprocess.gpt_data_process_worker import DataPreprocessor, DataPreprocessorReqData

random.seed(42)
# 修改为实际的输入输出路径
DATASET_ROOT = "/mnt/data_3t_1/datasets/raw_data/Emilia-YODAS/JA" 
OUTPUT_DIR = "/mnt/data_3t_2/datasets/indextts_train_data/Emilia-YODAS/JA"
MODEL_DIR = "./checkpoints/IndexTTS-2-vLLM"
TARGET_SR = 16000
CPU_WORKERS_NUM = 1   # Tar解压和解码比较耗CPU，可以适当增加
DEVICE_NUM = 8
PROCESSORS_PER_DEVICE = 1
MAX_GPU_TASK_QUEUE_SIZE = 32
MAX_AUDIO_DURATION = 36
BATCH_SIZE = 12


@dataclass
class FileManifest:
    """告诉Writer某个文件预计有多少条数据"""
    file_rel_path: str
    total_samples: int
    original_path: str
    

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
                tar_path = self.file_queue.get(timeout=3)
            except queue.Empty:
                break 

            if tar_path is None:
                break

            try:
                # 计算相对路径，输出后缀改为 .pkl
                rel_path = os.path.relpath(tar_path, DATASET_ROOT)
                # 假设输入是 .tar，输出对应改为 .pkl
                output_rel_path = rel_path.replace(".tar", ".pkl")
                output_path = os.path.join(OUTPUT_DIR, output_rel_path)
                
                if os.path.exists(output_path):
                    logger.info(f"[CPU-Loader-{self.worker_id}] Skipping {rel_path} as it already exists.")
                    continue

                valid_count = 0
                duration_skip_num = 0
                non_audio_or_text_skip_num = 0
                pf_start_time = time.time()
                
                # 缓冲区，用于配对 key.json 和 key.mp3
                # 结构: { "file_key": {"json": {...}, "audio": bytes} }
                sample_buffer = {}
                
                pbar = tqdm(desc=f"CPU-Loader-{self.worker_id}: {os.path.basename(tar_path)}")
                
                try:
                    # 使用 tarfile 读取流
                    with tarfile.open(tar_path, "r") as tar:
                        for member in tar:
                            if not member.isfile():
                                continue
                            
                            file_name = os.path.basename(member.name)
                            # Emilia 数据集通常结构: key.json, key.mp3
                            key, ext = os.path.splitext(file_name)
                            
                            # 过滤掉非目标文件
                            if ext not in ['.json', '.mp3']:
                                continue
                                
                            if key not in sample_buffer:
                                sample_buffer[key] = {}
                            
                            try:
                                f = tar.extractfile(member)
                                if f is None: continue
                                
                                if ext == '.json':
                                    content = json.load(f)
                                    sample_buffer[key]['json'] = content
                                elif ext == '.mp3':
                                    sample_buffer[key]['audio'] = f.read()
                            except Exception as e:
                                logger.warning(f"Error reading member {file_name}: {e}")
                                del sample_buffer[key]
                                continue

                            # 检查是否凑齐了一对
                            if 'json' in sample_buffer[key] and 'audio' in sample_buffer[key]:
                                data_item = sample_buffer.pop(key) # 取出并从 buffer 删除
                                
                                json_data = data_item['json']
                                audio_bytes = data_item['audio']
                                
                                text = json_data.get('text', "")
                                speaker = json_data.get('speaker', uuid.uuid4().hex)
                                
                                if not text or not audio_bytes:
                                    non_audio_or_text_skip_num += 1
                                    continue
                                
                                try:
                                    # 音频解码
                                    # sf.read 可以直接读取 bytes (需用 io.BytesIO 包裹)
                                    array, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')
                                    
                                    # 检查时长
                                    duration = array.shape[0] / sampling_rate
                                    if duration > MAX_AUDIO_DURATION:
                                        duration_skip_num += 1
                                        continue

                                    # 转单声道
                                    if array.ndim > 1:
                                        array = np.mean(array, axis=1)
                                    
                                    req = DataPreprocessorReqData(
                                        text=text,
                                        audio=array, 
                                        orig_sr=sampling_rate,
                                        file_rel_path=output_rel_path, # 使用计算好的输出路径标识
                                        speaker_id=speaker             # 传递 speaker
                                    )
                                    batch_req.append(req)
                                    valid_count += 1

                                    if len(batch_req) >= BATCH_SIZE:
                                        self.gpu_task_queue.put(batch_req)
                                        batch_req = []
                                        
                                except Exception as e:
                                    logger.error(f"Error processing audio in {key}: {e}")
                                    continue
                                finally:
                                    pbar.update(1)

                except Exception as e:
                    logger.error(f"Tar read error {tar_path}: {e}")

                pbar.close()
                logger.info(f"[CPU-Loader-{self.worker_id}: {rel_path}] Samples: {valid_count}. "
                            f"Duration skip: {duration_skip_num}, incomplete/error skip: {non_audio_or_text_skip_num}, time: {time.time()-pf_start_time:.2f}s")
                            
                # 发送剩余的 batch
                if len(batch_req) > 0:
                    self.gpu_task_queue.put(batch_req)
                    batch_req = []
                
                # 发送清单给 Writer
                # 注意：如果没有任何有效数据，也建议发送一个 count=0 的 manifest 以便 Writer 能够标记该文件结束
                manifest = FileManifest(
                    file_rel_path=output_rel_path,
                    total_samples=valid_count,
                    original_path=tar_path
                )
                self.writer_control_queue.put(manifest)

            except Exception as e:
                logger.error(f"[CPU-Loader-{self.worker_id}] File Error {tar_path}: {traceback.format_exc()}")


class ResultWriterWorker(Process):
    def __init__(self, result_queue: Queue, writer_control_queue: Queue, total_files: int):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.writer_control_queue = writer_control_queue
        self.total_files = total_files

    def run(self):
        logger.info("[Writer] Started.")
        
        # Buffer结构修改：
        # { file_rel_path: {'expected': int, 'data': Dict[speaker_id, List[ProcessedData]], 'received': int} }
        self.file_buffers = {}
        finished_files_count = 0
        
        pbar = tqdm(total=self.total_files, unit="file", desc="Processing Tar Files")
        
        while finished_files_count < self.total_files:
            # 1. 注册新文件 (Manifest)
            while not self.writer_control_queue.empty():
                try:
                    manifest: FileManifest = self.writer_control_queue.get_nowait()
                    file_key = manifest.file_rel_path
                    
                    if file_key not in self.file_buffers:
                        # 如果还没收到任何数据
                        self.file_buffers[file_key] = {
                            'expected': manifest.total_samples,
                            'data': {},  # 改为字典
                            'received': 0,
                        }
                    else:
                        # 已经有数据先到了
                        self.file_buffers[file_key]['expected'] = manifest.total_samples
                    
                    # 特殊情况：如果文件本身没有有效数据(total_samples=0)，直接完成
                    if manifest.total_samples == 0:
                        del self.file_buffers[file_key]
                        finished_files_count += 1
                        pbar.update(1)

                except queue.Empty:
                    break

            # 2. 接收处理结果
            try:
                # 获取结果: List[(file_rel_path, speaker_id, processed_data)]
                res = self.result_queue.get(timeout=0.1)
                for res_item in res:
                    file_rel_path, speaker_id, processed_data = res_item
                    
                    if file_rel_path not in self.file_buffers:
                        self.file_buffers[file_rel_path] = {
                            'expected': -1, 
                            'data': {}, # 改为字典
                            'received': 0,
                        }
                    
                    buffer_entry = self.file_buffers[file_rel_path]
                    
                    # 按 Speaker 归类
                    if speaker_id not in buffer_entry['data']:
                        buffer_entry['data'][speaker_id] = []
                    buffer_entry['data'][speaker_id].append(processed_data)
                    
                    buffer_entry['received'] += 1
                    
                    # 检查是否完成
                    # 稍微放宽条件 (0.95) 防止偶尔丢包导致死锁，或者依靠超时机制
                    if buffer_entry['expected'] != -1 and buffer_entry['received'] >= buffer_entry['expected'] * 0.95:
                        self._finish_file(file_rel_path, pbar)
                        finished_files_count += 1
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[Writer] Error: {traceback.format_exc()}")

        pbar.close()
        logger.info("[Writer] All files processed.")

    def _finish_file(self, file_rel_path, pbar):
        """保存并清理"""
        if file_rel_path in self.file_buffers:
            buf = self.file_buffers[file_rel_path]
            
            try:
                self.save_file(file_rel_path, buf['data'])
            finally:
                del self.file_buffers[file_rel_path]
                # 这里不增加 finished_files_count，在调用处增加
                pbar.update(1)

    def save_file(self, rel_path, data_dict):
        """
        data_dict: Dict[speaker_id, List[ProcessedData]]
        """            
        output_path = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            # 直接保存字典
            with open(output_path, "wb") as f:
                pickle.dump(data_dict, f)
        except Exception as e:
            logger.error(f"Failed to save {output_path}: {traceback.format_exc()}")


def main():
    if os.path.exists(OUTPUT_DIR):
        logger.warning(f"Output directory {OUTPUT_DIR} already exists.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 扫描文件 (扫描 .tar)
    logger.info("Scanning .tar files...")
    all_tar_files = []
    # 如果数据在 root 下有子文件夹结构，用 scandir 递归或者 glob
    # 假设结构是 DATASET_ROOT/*.tar 或者 DATASET_ROOT/*/*.tar
    
    # 递归搜索所有 .tar 文件
    for root, dirs, files in os.walk(DATASET_ROOT):
        for file in files:
            if file.endswith(".tar"):
                all_tar_files.append(os.path.join(root, file))

    # 排序以保证确定性
    all_tar_files.sort()
    
    total_files = len(all_tar_files)
    logger.info(f"Found {total_files} tar files.")

    if total_files == 0:
        logger.error(f"No .tar files found in {DATASET_ROOT}")
        return

    # 2. 创建队列
    file_queue = Queue() 
    gpu_task_queue = Queue(maxsize=MAX_GPU_TASK_QUEUE_SIZE)
    writer_control_queue = Queue()
    result_queue = Queue()

    # 3. 填充任务队列
    for f in all_tar_files:
        file_queue.put(f)

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