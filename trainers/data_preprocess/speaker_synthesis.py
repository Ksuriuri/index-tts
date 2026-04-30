import json
import os
import io
import glob
import random
random.seed(42)
import time
import queue
import traceback
import filetype
import pandas as pd
import aiohttp 
import asyncio 
from tqdm import tqdm
from loguru import logger
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Value
import soundfile as sf


# DATASET_NAME = "Galgame-VisualNovel-Reupload"
# DATASET_NAME = "Japanese-Eroge-Voice"
DATASET_NAME = "maa"

DATASET_DIR = f"/mnt/data_3t_1/datasets/preprocess/{DATASET_NAME}"
OUTPUT_DIR = f"/mnt/data_3t_1/datasets/preprocess/synthesis_data/{DATASET_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 加载 TTS 文本（与 emo_synthesis 一致，使用 txt 中的行） ---
TTS_TEXTS = []
with open("/mnt/data_sdd/hhy/index-tts/assets/tts_dataset.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            TTS_TEXTS.append(line)

MIN_AUDIO_DURATION = 3  # 6
MAX_AUDIO_DURATION = 30  # 36

# for jp
# CER_THRESHOLD = 0.20
# CER_TYPE = "pron_CER"

# for es
CER_THRESHOLD = 0.30
CER_TYPE = "cer"

# --- 新的配置参数 ---
NUM_TTS_PORTS = 8           # 端口数量 (0-7)
MAX_REQ_PER_PORT = 4       # 每个端口最大并发
TOTAL_CONCURRENCY = NUM_TTS_PORTS * MAX_REQ_PER_PORT # 总并发 256

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


# 异步 TTS 请求函数
async def tts_gen_async(session, text, audio_bytes, port_id):
    kind = filetype.guess(audio_bytes)
    
    if kind is None:
        raise ValueError("无法识别音频格式")
    else:
        audio_ext = kind.extension
        mime_type = kind.mime

    file_name = f"ref_audio.{audio_ext}"

    base_port = 11990
    selected_port = base_port + port_id # 使用传入的 port_id
    target_url = f'http://0.0.0.0:{selected_port}/tts'

    emo_vec = {
        "joy": 0.0, "anger": 0.0, "sadness": 0.0, 
        "fear": 0.0, "disgust": 0.0, "depression": 0.0, 
        "surprise": 0.0, "calm": 0.0
    }
    
    # 使用 aiohttp 的 FormData
    data = aiohttp.FormData()
    data.add_field('synthesis_text', text)
    data.add_field('emo', json.dumps(emo_vec, ensure_ascii=False))
    # add_field 处理文件上传
    data.add_field('wav', io.BytesIO(audio_bytes), filename=file_name, content_type=mime_type)

    try:
        # 异步发送 POST 请求
        async with session.post(target_url, data=data, timeout=60) as response:
            if response.status == 200:
                return await response.read()
            else:
                text_response = await response.text()
                raise Exception(f"服务异常 ({selected_port}): {response.status} - {text_response}")
            
    except Exception as e:
        raise Exception(f"连接端口 {selected_port} 失败: {e}")


class TTSWorker(Process):
    def __init__(self, file_queue: Queue, worker_id: int, progress_counter):
        super().__init__(daemon=True)
        self.file_queue = file_queue
        self.worker_id = worker_id
        self.progress_counter = progress_counter

    def run(self):
        # 启动异步事件循环
        asyncio.run(self._run_async())

    async def _process_single_row(self, session, port_queue, current_port_id, idx, row_data, df, pbar):
        """
        处理单行数据的异步任务
        """
        try:
            audio_bytes = row_data['audio']
            
            # 1. 基础检查
            if not audio_bytes:
                df.at[idx, 'audio'] = b""
                df.at[idx, 'text'] = ""
                return

            # 2. 音频时长检查
            try:
                # 注意：在大并发下，这里的 CPU 同步操作可能会轻微阻塞 EventLoop
                # 如果发现性能瓶颈，可以考虑 loop.run_in_executor
                array, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')
                duration = array.shape[0] / sampling_rate
                
                if duration > MAX_AUDIO_DURATION or duration < MIN_AUDIO_DURATION:
                    df.at[idx, 'audio'] = b""
                    df.at[idx, 'text'] = ""
                    return
            except Exception:
                df.at[idx, 'audio'] = b""
                df.at[idx, 'text'] = ""
                return

            # 3. 准备文本和请求（与 emo_synthesis 一致，从 txt 词表随机选取）
            syn_text = random.choice(TTS_TEXTS)
            
            # 4. 发送请求，使用分配到的 current_port_id
            new_audio = await tts_gen_async(session, syn_text, audio_bytes, current_port_id)
            
            # 5. 更新结果
            df.at[idx, 'audio'] = new_audio
            df.at[idx, 'text'] = syn_text
        
        except Exception as e:
            # logger.error(f"Error processing row {idx} on port {current_port_id}: {e}")
            df.at[idx, 'audio'] = b""
            df.at[idx, 'text'] = ""
        finally:
            # 关键修改：任务结束，将端口 ID 放回队列，供下一个任务使用
            port_queue.put_nowait(current_port_id)
            pbar.update(1)

    async def _run_async(self):
        from tqdm import tqdm

        # --- 初始化端口队列 ---
        port_queue = asyncio.Queue()
        for p in range(NUM_TTS_PORTS):
            for _ in range(MAX_REQ_PER_PORT):
                port_queue.put_nowait(p)

        logger.info(f"Worker started. Total concurrency slots: {port_queue.qsize()}")

        # 创建异步 session，连接池大小要大于总并发数
        connector = aiohttp.TCPConnector(limit=TOTAL_CONCURRENCY + 20) 
        async with aiohttp.ClientSession(connector=connector) as session:
            
            while True:
                try:
                    try:
                        parquet_path = self.file_queue.get(timeout=5)
                    except queue.Empty:
                        break
                    
                    if parquet_path is None: break

                    output_path = os.path.join(OUTPUT_DIR, os.path.relpath(parquet_path, DATASET_DIR))
                    
                    if os.path.exists(output_path):
                        with self.progress_counter.get_lock():
                            self.progress_counter.value += 1
                        continue

                    # 读取数据
                    df = pd.read_parquet(parquet_path)
                    df['origin_idx'] = df.index
                    df['text'] = df['text'].astype(str)

                    file_pbar = tqdm(
                        total=len(df), 
                        desc=f"Processing | {os.path.basename(parquet_path)[:20]}",
                        position=1, 
                        leave=False, 
                        dynamic_ncols=True
                    )

                    # --- 核心异步并发逻辑 ---
                    tasks = []

                    # 遍历 DataFrame 提交任务
                    for idx in df.index:
                        whisper_info = df.at[idx, 'whisper_large_v3']
                        if whisper_info.get(CER_TYPE) > CER_THRESHOLD:
                            df.at[idx, 'audio'] = b""
                            df.at[idx, 'text'] = ""
                            continue

                        unique_speakers = set(seg['speaker'] for seg in df.at[idx, 'speaker_diarization'])
                        if len(unique_speakers) != 1:
                            df.at[idx, 'audio'] = b""
                            df.at[idx, 'text'] = ""
                            continue

                        # 1. 从队列获取一个可用的端口 ID
                        # 如果所有 256 个 slot 都在忙，这里会阻塞，实现了并发控制
                        current_port_id = await port_queue.get()
                        
                        # 2. 创建任务
                        row_data = {'audio': df.at[idx, 'audio']}
                        task = asyncio.create_task(
                            self._process_single_row(
                                session, port_queue, current_port_id,
                                idx, row_data, df, file_pbar
                            )
                        )
                        tasks.append(task)

                    # 等待所有任务完成
                    if tasks:
                        await asyncio.gather(*tasks)
                    
                    file_pbar.close()

                    # 保存结果
                    df_to_save = df[['audio', 'text', 'origin_idx']]
                    df_to_save.to_parquet(output_path, engine='pyarrow', index=False)
                    
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
    logger.info(f"Found {total_files} parquet files.")

    file_queue = Queue()
    for f in all_parquet_files:
        file_queue.put(f)

    processed_counter = Value('i', 0)
    workers = []
    
    # --- 修改：只启动 1 个 Worker ---
    logger.info(f"Starting single TTSWorker with {TOTAL_CONCURRENCY} concurrent requests across {NUM_TTS_PORTS} ports.")
    
    p = TTSWorker(file_queue, 0, processed_counter)
    p.start()
    workers.append(p)

    # 全局进度条 (Position 0)
    with tqdm(total=total_files, desc="TOTAL FILES", position=0, dynamic_ncols=True) as pbar:
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
        
    print("\n")
    logger.info("All processing done.")

if __name__ == "__main__":
    main()