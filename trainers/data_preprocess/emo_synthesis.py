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


DATASET_NAMES = [
    # "Emilia_JA",
    # "Emilia-YODAS_JA",
    # "Gacha_games_jp",
    # "Galgame-VisualNovel-Reupload",
    # "Japanese-Eroge-Voice",

    # "google-chilean-spanish",
    "voxpopuli",
    "MLS_Spanish",
]

DATA_ROOT = "/mnt/data_3t_1/datasets/preprocess"
OUTPUT_ROOT = "/mnt/data_3t_1/datasets/preprocess/emo_synthesis_data"

# --- 加载 TTS 文本 ---
TTS_TEXTS = []
with open("/mnt/data_sdd/hhy/index-tts/assets/tts_dataset.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            TTS_TEXTS.append(line)

MIN_AUDIO_DURATION = 3  # 6
MAX_AUDIO_DURATION = 30  # 36

CER_THRESHOLD = 0.30  # 0.10
CER_TYPE = "cer"
# CER_TYPE = "pron_CER"

# --- 并发配置 ---
NUM_TTS_PORTS = 8           # TTS 端口数量 (0-7)
MAX_REQ_PER_PORT = 2        # 每个 TTS 端口最大并发
TOTAL_CONCURRENCY = NUM_TTS_PORTS * MAX_REQ_PER_PORT
MAX_VC_CONCURRENCY = 16     # VC 服务全局最大并发
TTS_TIMEOUT_SECONDS = 30
VC_TIMEOUT_SECONDS = 20
MAX_TIMEOUT_RETRIES = 3

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
    
    def build_data():
        data = aiohttp.FormData()
        data.add_field('synthesis_text', text)
        data.add_field('emo', json.dumps(emo_vec, ensure_ascii=False))
        data.add_field('wav', io.BytesIO(audio_bytes), filename=file_name, content_type=mime_type)
        return data

    try:
        return await post_with_timeout_retry(
            session=session,
            url=target_url,
            build_data=build_data,
            timeout_seconds=TTS_TIMEOUT_SECONDS,
            request_name=f"TTS({selected_port})",
        )
            
    except Exception as e:
        raise Exception(f"连接端口 {selected_port} 失败: {e}")

async def post_with_timeout_retry(session, url, build_data, timeout_seconds, request_name):
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    for attempt in range(1, MAX_TIMEOUT_RETRIES + 1):
        try:
            async with session.post(url, data=build_data(), timeout=timeout) as response:
                if response.status == 200:
                    return await response.read()

                text_response = await response.text()
                raise Exception(f"{request_name} 服务异常: {response.status} - {text_response}")
        except asyncio.TimeoutError as e:
            if attempt >= MAX_TIMEOUT_RETRIES:
                raise Exception(
                    f"{request_name} 超时，已重试 {MAX_TIMEOUT_RETRIES} 次，每次超时 {timeout_seconds} 秒"
                ) from e
            logger.warning(
                f"{request_name} 第 {attempt}/{MAX_TIMEOUT_RETRIES} 次请求超时，准备重试"
            )

# 异步 VC 请求函数
async def vc_convert_async(session, source_audio_bytes, target_audio_bytes, host="http://127.0.0.1:11452"):
    url = f"{host}/convert"
    
    def build_data():
        data = aiohttp.FormData()
        data.add_field('source', io.BytesIO(source_audio_bytes), filename='source.wav', content_type='audio/wav')
        data.add_field('target', io.BytesIO(target_audio_bytes), filename='target.wav', content_type='audio/wav')
        
        data.add_field('diffusion_steps', '30')
        data.add_field('length_adjust', '1.0')
        data.add_field('intelligibility_cfg_rate', '0.7')
        data.add_field('similarity_cfg_rate', '0.7')
        data.add_field('top_p', '0.9')
        data.add_field('temperature', '1.0')
        data.add_field('repetition_penalty', '1.0')
        data.add_field('convert_style', 'False')
        data.add_field('anonymization_only', 'False')
        return data

    try:
        return await post_with_timeout_retry(
            session=session,
            url=url,
            build_data=build_data,
            timeout_seconds=VC_TIMEOUT_SECONDS,
            request_name="VC",
        )
    except Exception as e:
        raise Exception(f"连接VC服务失败: {e}")

class TTSWorker(Process):
    def __init__(self, file_queue: Queue, worker_id: int, progress_counter, dataset_dir: str, output_dir: str):
        super().__init__(daemon=True)
        self.file_queue = file_queue
        self.worker_id = worker_id
        self.progress_counter = progress_counter
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir

    def run(self):
        # 启动异步事件循环
        asyncio.run(self._run_async())

    async def _process_single_row(
        self,
        session,
        port_queue,
        vc_semaphore,
        current_port_id,
        idx,
        row_data,
        target_candidate_indices,
        target_candidate_positions,
        target_audio_map,
        df,
        pbar,
    ):
        """
        处理单行数据的异步任务
        """
        port_released = False
        try:
            audio_bytes = row_data['audio']
            
            # 1. 基础检查
            if not audio_bytes:
                df.at[idx, 'audio'] = b""
                df.at[idx, 'text'] = ""
                return

            # 2. 音频时长检查
            try:
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

            # 3. 准备文本和请求
            syn_text = random.choice(TTS_TEXTS)
            
            # 4. 发送 TTS 请求。TTS 返回后立即释放端口槽位，避免被后续 VC 阶段占住。
            try:
                tts_audio = await tts_gen_async(session, syn_text, audio_bytes, current_port_id)
            finally:
                port_queue.put_nowait(current_port_id)
                port_released = True
            
            # 5. 随机选择目标音频
            candidate_count = len(target_candidate_indices)
            idx_position = target_candidate_positions.get(idx)
            if candidate_count == 0 or (candidate_count == 1 and idx_position is not None):
                df.at[idx, 'audio'] = b""
                df.at[idx, 'text'] = ""
                return

            if idx_position is None:
                target_idx = random.choice(target_candidate_indices)
            else:
                random_pos = random.randrange(candidate_count - 1)
                if random_pos >= idx_position:
                    random_pos += 1
                target_idx = target_candidate_indices[random_pos]

            target_audio_bytes = target_audio_map[target_idx]
            
            # 6. 调用 VC 接口，使用独立的全局限流
            async with vc_semaphore:
                new_audio = await vc_convert_async(session, tts_audio, target_audio_bytes)
            
            # 7. 更新结果
            df.at[idx, 'audio'] = new_audio
            df.at[idx, 'text'] = syn_text
        
        except Exception as e:
            # logger.error(f"Error processing row {idx} on port {current_port_id}: {e}")
            df.at[idx, 'audio'] = b""
            df.at[idx, 'text'] = ""
        finally:
            if not port_released:
                port_queue.put_nowait(current_port_id)
            pbar.update(1)

    async def _run_async(self):
        from tqdm import tqdm

        # --- 初始化端口队列 ---
        port_queue = asyncio.Queue()
        for p in range(NUM_TTS_PORTS):
            for _ in range(MAX_REQ_PER_PORT):
                port_queue.put_nowait(p)

        vc_semaphore = asyncio.Semaphore(MAX_VC_CONCURRENCY)
        logger.info(
            f"Worker started. TTS slots: {port_queue.qsize()}, VC concurrency: {MAX_VC_CONCURRENCY}"
        )

        # 连接池大小覆盖 TTS 和 VC 的并发请求
        connector = aiohttp.TCPConnector(limit=TOTAL_CONCURRENCY + MAX_VC_CONCURRENCY + 20)
        async with aiohttp.ClientSession(connector=connector) as session:
            
            while True:
                try:
                    try:
                        parquet_path = self.file_queue.get(timeout=5)
                    except queue.Empty:
                        break
                    
                    if parquet_path is None: break

                    output_path = os.path.join(self.output_dir, os.path.relpath(parquet_path, self.dataset_dir))
                    
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
                    processable_indices = []
                    target_candidate_indices = []
                    target_audio_map = {}

                    # 先完成轻量筛选，并预构建目标音频池，避免每个任务都扫描整张 df
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

                        processable_indices.append(idx)
                        if df.at[idx, 'audio']:
                            target_candidate_indices.append(idx)
                            target_audio_map[idx] = df.at[idx, 'audio']

                    target_candidate_positions = {
                        candidate_idx: pos
                        for pos, candidate_idx in enumerate(target_candidate_indices)
                    }

                    # 遍历可处理行提交任务
                    for idx in processable_indices:
                        # 1. 从队列获取一个可用的 TTS 端口槽位
                        current_port_id = await port_queue.get()
                        
                        # 2. 创建任务
                        row_data = {'audio': df.at[idx, 'audio']}
                        task = asyncio.create_task(
                            self._process_single_row(
                                session, port_queue, vc_semaphore, current_port_id,
                                idx, row_data, target_candidate_indices,
                                target_candidate_positions, target_audio_map, df, file_pbar
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
    for dataset_name in DATASET_NAMES:
        dataset_dir = os.path.join(DATA_ROOT, dataset_name)
        output_dir = os.path.join(OUTPUT_ROOT, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(dataset_dir):
            logger.warning(f"Input directory {dataset_dir} does not exist, skipping.")
            continue

        all_parquet_files = sorted(glob.glob(os.path.join(dataset_dir, "*.parquet")))
        total_files = len(all_parquet_files)
        logger.info(f"===== Dataset: {dataset_name} | {total_files} parquet files =====")

        if total_files == 0:
            logger.warning(f"No parquet files found in {dataset_dir}, skipping.")
            continue

        file_queue = Queue()
        for f in all_parquet_files:
            file_queue.put(f)

        processed_counter = Value('i', 0)
        workers = []

        logger.info(f"Starting single TTSWorker with {TOTAL_CONCURRENCY} concurrent requests across {NUM_TTS_PORTS} ports.")

        p = TTSWorker(file_queue, 0, processed_counter, dataset_dir, output_dir)
        p.start()
        workers.append(p)

        with tqdm(total=total_files, desc=f"[{dataset_name}] FILES", position=0, dynamic_ncols=True) as pbar:
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
        logger.info(f"Dataset {dataset_name} done.")

    logger.info("All datasets processing done.")

if __name__ == "__main__":
    main()
