# 处理 MLS-Spanish 类数据结构: audio/*/*/*.flac + 根目录 transcripts.txt
# 参考 preprocess_wav_lab.py，尽量保持原有逻辑与代码结构

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

# --- 配置 (MLS-Spanish 新数据结构) ---
# 根目录: train/ 下有 audio/ 与 transcripts.txt
# audio/ 下有子文件夹如 10013, 10256；10013 下又有 9951 等；9951 下为同说话人的 .flac 文件
DATASET_ROOT = "/mnt/data_3t_1/datasets/raw_data/Spanish/MLS-Spanish/mls_spanish/train"
AUDIO_DIR = os.path.join(DATASET_ROOT, "audio")
TRANSCRIPTS_PATH = os.path.join(DATASET_ROOT, "transcripts.txt")
OUTPUT_DIR = "/mnt/data_3t_1/datasets/preprocess/Spanish/MLS_Spanish"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "resume_checkpoint.json")

WHISPER_MODEL_SIZE = "large-v3"
COMPUTE_TYPE = "float16"
DEVICE_NUM = 8
PROCESSORS_PER_DEVICE = 1
CPU_WORKERS_NUM = 1
BATCH_SIZE = 16
SAVE_INTERVAL = 20000
MAX_AUDIO_DURATION = -1  # 秒，-1 表示不限制


@dataclass
class ASRTask:
    audio_bytes: bytes
    audio_raw: np.ndarray  # float16
    sample_rate: int
    text_gt: str
    speaker: str
    source_key: str


def load_transcripts(path: str) -> Dict[str, str]:
    """读取 transcripts.txt，格式为: id\ttext 或 id 空格 text。同 id 出现多次时保留最后一次。"""
    out = {}
    if not os.path.isfile(path):
        logger.warning(f"Transcripts file not found: {path}")
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 支持 tab 或空白分隔，只分一次，保证 text 内空格保留
            if "\t" in line:
                id_, text = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                id_, text = parts[0], parts[1]
            id_ = id_.strip()
            text = text.strip()
            if id_ and text:
                out[id_] = text
    logger.info(f"Loaded {len(out)} transcript entries from {path}")
    return out


def collect_speaker_tasks(audio_dir: str) -> List[tuple]:
    """
    扫描 audio_dir，结构为 audio_dir / X / Y / *.flac。
    返回 [(speaker_key, dir_path), ...]，其中 speaker_key 为 "X/Y"，dir_path 为含 .flac 的目录绝对路径。
    """
    tasks = []
    top_dirs = [d for d in glob.glob(os.path.join(audio_dir, "*")) if os.path.isdir(d)]
    for top in sorted(top_dirs):
        top_name = os.path.basename(top)
        sub_dirs = [d for d in glob.glob(os.path.join(top, "*")) if os.path.isdir(d)]
        for sub in sorted(sub_dirs):
            sub_name = os.path.basename(sub)
            flac_files = glob.glob(os.path.join(sub, "*.flac"))
            if not flac_files:
                continue
            speaker_key = f"{top_name}/{sub_name}"
            tasks.append((speaker_key, sub))
    return tasks


class AudioLoaderWorker(Process):
    def __init__(self, dir_queue: Queue, gpu_task_queue: Queue, worker_id: int, checkpoint: Dict[str, int], sample_pbar_counter, transcript_dict: Dict[str, str]):
        super().__init__(daemon=True)
        self.dir_queue = dir_queue
        self.gpu_task_queue = gpu_task_queue
        self.worker_id = worker_id
        self.checkpoint = checkpoint
        self.sample_pbar_counter = sample_pbar_counter
        self.transcript_dict = transcript_dict

    def run(self):
        logger.info(f"[CPU-Loader-{self.worker_id}] Started.")
        current_batch = []

        while True:
            try:
                task_data = self.dir_queue.get(timeout=10)
            except queue.Empty:
                break

            if task_data is None:
                break

            speaker_key, dir_path = task_data
            source_key = speaker_key

            try:
                flac_files = sorted(glob.glob(os.path.join(dir_path, "*.flac")))
                total_in_dir = len(flac_files)

                skip_count = self.checkpoint.get(source_key, 0)
                if skip_count > 0:
                    if skip_count >= total_in_dir:
                        logger.info(f"[Loader-{self.worker_id}] Skipping {source_key} (completed).")
                        continue
                    logger.info(f"[Loader-{self.worker_id}] {source_key} resuming from index {skip_count}")

                processed_count = 0

                for idx, flac_path in enumerate(flac_files):
                    if idx < skip_count:
                        continue

                    processed_count += 1
                    # 音频 id 为文件名无扩展名，与 transcripts.txt 中的 id 对应
                    audio_id = os.path.splitext(os.path.basename(flac_path))[0]
                    text_gt = self.transcript_dict.get(audio_id)
                    if not text_gt or not text_gt.strip():
                        continue

                    try:
                        with open(flac_path, "rb") as f:
                            audio_bytes = f.read()
                        with io.BytesIO(audio_bytes) as buf:
                            array, sr = sf.read(buf)

                        if MAX_AUDIO_DURATION > 0 and len(array) / sr > MAX_AUDIO_DURATION:
                            continue

                        if array.ndim > 1:
                            array = np.mean(array, axis=1)

                        task = ASRTask(
                            audio_bytes=audio_bytes,
                            audio_raw=array.astype(np.float16),
                            sample_rate=sr,
                            text_gt=text_gt.strip(),
                            speaker=source_key,
                            source_key=source_key,
                        )
                        current_batch.append(task)

                        if len(current_batch) >= BATCH_SIZE:
                            self.gpu_task_queue.put(current_batch)
                            current_batch = []

                        with self.sample_pbar_counter.get_lock():
                            self.sample_pbar_counter.value += 1

                    except Exception as e:
                        logger.warning(f"Error processing {flac_path}: {e}")
                        continue

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
                if tasks is None:
                    break
            except queue.Empty:
                continue

            for task in tasks:
                try:
                    audio_fp32 = task.audio_raw.astype(np.float32)

                    if task.sample_rate != 16000:
                        audio_tensor = torch.from_numpy(audio_fp32).to(device_str)
                        if task.sample_rate not in resamplers:
                            resamplers[task.sample_rate] = torchaudio.transforms.Resample(task.sample_rate, 16000).to(device_str)
                        audio_16k = resamplers[task.sample_rate](audio_tensor).cpu().numpy()
                    else:
                        audio_16k = audio_fp32

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
                            "segments": segments_list,
                        },
                        "_source_key": task.source_key,
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
        if existing_parts:
            indices = []
            for f in existing_parts:
                try:
                    name = os.path.basename(f)
                    idx = int(name.split("_")[1].split(".")[0])
                    indices.append(idx)
                except Exception:
                    pass
            file_idx = max(indices) + 1 if indices else 0
        else:
            file_idx = 0

        pbar = tqdm(desc="Samples Written", unit="samples", dynamic_ncols=True)

        while True:
            try:
                data = self.result_queue.get(timeout=20)
                if data == "DONE":
                    break

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
            pd.DataFrame(data_list).to_parquet(save_path, engine="pyarrow", index=False)

            for src, count in cycle_counts.items():
                self.checkpoint[src] = self.checkpoint.get(src, 0) + count

            with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(self.checkpoint, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {save_path} ({len(data_list)} samples).")
        except Exception as e:
            logger.error(f"Failed to save parquet {save_path}: {e}")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint = load_checkpoint()

    transcript_dict = load_transcripts(TRANSCRIPTS_PATH)
    if not transcript_dict:
        logger.error("No transcripts loaded. Exit.")
        return

    speaker_tasks = collect_speaker_tasks(AUDIO_DIR)
    logger.info(f"Found {len(speaker_tasks)} speaker directories (audio/*/* with .flac).")

    dir_queue = Queue()
    for task in speaker_tasks:
        dir_queue.put(task)

    gpu_task_queue = Queue(maxsize=DEVICE_NUM * 32)
    result_queue = Queue()
    sample_pbar_counter = Value("i", 0)

    writer = ParquetWriterWorker(result_queue, OUTPUT_DIR, SAVE_INTERVAL, CHECKPOINT_PATH, checkpoint)
    writer.start()

    gpu_workers = []
    for gpu_id in range(DEVICE_NUM):
        for w_id in range(PROCESSORS_PER_DEVICE):
            worker_global_id = gpu_id * PROCESSORS_PER_DEVICE + w_id
            p = GPUASRWorker(gpu_task_queue, result_queue, gpu_id, worker_global_id)
            p.start()
            gpu_workers.append(p)

    cpu_workers = []
    for i in range(CPU_WORKERS_NUM):
        p = AudioLoaderWorker(dir_queue, gpu_task_queue, i, checkpoint, sample_pbar_counter, transcript_dict)
        p.start()
        cpu_workers.append(p)

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
            curr_val = sample_pbar_counter.value
            if curr_val > last_val:
                pbar.update(curr_val - last_val)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Stopping...")
        for p in cpu_workers:
            p.terminate()
        for p in gpu_workers:
            p.terminate()
        writer.terminate()
        return

    logger.info("All loaders finished. Waiting for GPU workers...")

    for p in cpu_workers:
        p.join()

    for _ in gpu_workers:
        gpu_task_queue.put(None)
    for p in gpu_workers:
        p.join()

    result_queue.put("DONE")
    writer.join()
    logger.info("All done.")


if __name__ == "__main__":
    main()
