import os
import io
import glob
import tempfile
import time
import queue
import traceback
import torch
import torchaudio
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from loguru import logger
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Value
import soundfile as sf

# 引入 Pyannote
from pyannote.audio import Pipeline

# --- 配置 ---
# DATASET_NAME = "Galgame-VisualNovel-Reupload"
# DATASET_NAME = "Gacha_games_jp"
# DATASET_NAME = "Emilia_JA"
DATASET_NAME = "Emilia-YODAS_JA"
# DATASET_NAME = "Japanese-Eroge-Voice"
DATASET_DIR = f"/mnt/data_3t_1/datasets/preprocess/{DATASET_NAME}"
HF_TOKEN = os.environ.get('HF_TOKEN', None)

DEVICE_NUM = 2
PROCESSORS_PER_DEVICE = 1  

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

class DiarizationWorker(Process):
    def __init__(self, file_queue: Queue, gpu_id: int, worker_id: int, progress_counter):
        super().__init__(daemon=True)
        self.file_queue = file_queue
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.progress_counter = progress_counter

    def run(self):
        # 延迟导入 tqdm，避免多进程冲突
        from tqdm import tqdm

        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        device_str = f"cuda:{self.gpu_id}"
        # logger.info 会干扰 tqdm 界面，但为了调试保留，建议在正式跑时减少 log 输出
        logger.info(f"[Worker-{self.worker_id}] Initializing Pipeline on {device_str}...")

        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1", 
                token=HF_TOKEN
            )
            pipeline.to(torch.device(device_str))
        except Exception as e:
            logger.error(f"[Worker-{self.worker_id}] Failed to load pipeline: {e}")
            return

        while True:
            try:
                parquet_path = self.file_queue.get(timeout=5)
            except queue.Empty:
                break
            
            if parquet_path is None: break

            try:
                # 读取 Parquet
                df = pd.read_parquet(parquet_path)
                
                if "speaker_diarization" in df.columns and df["speaker_diarization"].iloc[0] is not None:
                    # 即使跳过也更新全局计数
                    with self.progress_counter.get_lock():
                        self.progress_counter.value += 1
                    continue

                diarization_results = []
                temp_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
                # temp_dir = "/mnt/data_sdd/hhy/index-tts/outputs/temp"
                
                # --- 为当前文件创建一个进度条 ---
                # position=self.worker_id + 1 确保每个 worker 占一行，不与全局进度条(position 0)重叠
                # leave=False 运行完后自动清除该 worker 的小进度条，保持界面整洁
                file_pbar = tqdm(
                    total=len(df), 
                    desc=f"W-{self.worker_id} | {os.path.basename(parquet_path)[:20]}",
                    position=self.worker_id + 1, 
                    leave=False, 
                    dynamic_ncols=True
                )

                for idx, row in df.iterrows():
                    audio_bytes = row['audio']
                    if not audio_bytes:
                        diarization_results.append([])
                        file_pbar.update(1)
                        continue

                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_dir, delete=True) as tmp_wav:
                            tmp_wav.write(audio_bytes)
                            tmp_wav.flush()
                            
                            diarization = pipeline(tmp_wav.name)
                            
                            segments = []
                            for turn, speaker in diarization.speaker_diarization:
                                segments.append({
                                    "start": round(turn.start, 3),
                                    "end": round(turn.end, 3),
                                    "speaker": speaker
                                })
                            diarization_results.append(segments)
                    except Exception as e:
                        diarization_results.append([])
                    
                    file_pbar.update(1) # 更新 Worker 内部进度

                file_pbar.close() # 处理完一个文件关闭进度条

                # 保存结果
                df['speaker_diarization'] = diarization_results
                df.to_parquet(parquet_path, engine='pyarrow', index=False)
                
                # 更新全局进度
                with self.progress_counter.get_lock():
                    self.progress_counter.value += 1

            except Exception as e:
                logger.error(f"[Worker-{self.worker_id}] Error: {traceback.format_exc()}")

        logger.info(f"[Worker-{self.worker_id}] Finished.")

def main():
    if not os.path.exists(DATASET_DIR):
        logger.error(f"Input directory {DATASET_DIR} does not exist.")
        return

    all_parquet_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.parquet")))
    total_files = len(all_parquet_files)
    logger.info(f"Found {total_files} parquet files. Starting {DEVICE_NUM * PROCESSORS_PER_DEVICE} workers.")

    file_queue = Queue()
    for f in all_parquet_files:
        file_queue.put(f)

    processed_counter = Value('i', 0)
    workers = []
    
    # 启动 Workers
    for gpu_id in range(DEVICE_NUM):
        for p_id in range(PROCESSORS_PER_DEVICE):
            worker_id = gpu_id * PROCESSORS_PER_DEVICE + p_id
            p = DiarizationWorker(file_queue, gpu_id, worker_id, processed_counter)
            p.start()
            workers.append(p)

    # 全局进度条 (Position 0)
    with tqdm(total=total_files, desc="TOTAL PROGRESS", position=0, dynamic_ncols=True) as pbar:
        last_val = 0
        while any(p.is_alive() for p in workers):
            curr_val = processed_counter.value
            if curr_val > last_val:
                pbar.update(curr_val - last_val)
                last_val = curr_val
            time.sleep(1)
        pbar.update(processed_counter.value - last_val)

    for p in workers:
        p.join()
        
    # 清理屏幕，防止 tqdm 残留行
    print("\n" * (DEVICE_NUM * PROCESSORS_PER_DEVICE + 1))
    logger.info("All processing done.")

if __name__ == "__main__":
    main()