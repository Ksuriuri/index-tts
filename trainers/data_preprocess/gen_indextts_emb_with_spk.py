from dataclasses import dataclass
import os
import random
import sys
import pickle
import glob
import threading
import time
import traceback
from typing import Dict, List
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

import safetensors
from omegaconf import OmegaConf
from transformers import SeamlessM4TFeatureExtractor
from typing import List as _List

# 原始路径设置
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer
from trainers.utils import ProcessedData

random.seed(42)

# DATASET_ROOT = "/mnt/data_3t_1/datasets/preprocess"
# DATASET_ROOTS = [
#     f"{DATASET_ROOT}/Emilia_JA",
#     f"{DATASET_ROOT}/Emilia-YODAS_JA",
#     f"{DATASET_ROOT}/Gacha_games_jp",
# ]

DATASET_ROOT = "/mnt/data_3t_1/datasets/preprocess"
DATASET_ROOTS = [
    f"{DATASET_ROOT}/Emilia_JA",
    f"{DATASET_ROOT}/Emilia-YODAS_JA",
    f"{DATASET_ROOT}/Gacha_games_jp",
    f"{DATASET_ROOT}/google-chilean-spanish",
    f"{DATASET_ROOT}/MLS_Spanish",
    f"{DATASET_ROOT}/voxpopuli",
]

EMO_SYNTHESIS_ROOT = "/mnt/data_3t_1/datasets/preprocess/emo_synthesis_data"

OUTPUT_DIR = f"/mnt/data_3t_2/datasets/indextts_train_data_v2/jp_es"

MODEL_DIR = "./checkpoints/IndexTTS-2-vLLM"
BPE_MODEL_PATH = os.path.join(MODEL_DIR, "jp_es_bpe.model")
TARGET_SR = 16000
CPU_WORKERS_NUM = 1  # 负责读取和解码的CPU进程数
DEVICE_NUM = 8
PROCESSORS_PER_DEVICE = 1
MAX_GPU_TASK_QUEUE_SIZE = 16  # 限制队列大小防止内存爆炸
MAX_AUDIO_DURATION = 36
BATCH_SIZE = 12


TARGET_SR = 16000


class GPUFeatureExtractor(torch.nn.Module):
    """
    GPU-based replacement for SeamlessM4TFeatureExtractor.
    Replicates the exact same computation (Kaldi-style fbank) using batched
    PyTorch ops on GPU, eliminating the numpy per-frame Python for-loop.
    """

    def __init__(self, mel_filters_np, window_np, num_mel_bins=80, stride=2):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.stride = stride
        self.frame_length = 400
        self.hop_length = 160
        self.fft_length = 512
        self.preemphasis_coeff = 0.97
        self.mel_floor = 1.192092955078125e-07
        self.scale = float(2 ** 15)

        self.register_buffer('window', torch.from_numpy(window_np).float())
        self.register_buffer('mel_filters', torch.from_numpy(mel_filters_np).float())

    @torch.no_grad()
    def forward(self, audios: _List[torch.Tensor]) -> dict:
        """
        Args:
            audios: list of 1D GPU tensors, float32, values in [-1, 1], sr=16000
        Returns:
            dict with input_features [B, T//stride, mel*stride] and
            attention_mask [B, T//stride], both on GPU.
        """
        device = audios[0].device
        batch_size = len(audios)
        lengths = [a.shape[0] for a in audios]
        max_len = max(lengths)

        padded = torch.zeros(batch_size, max_len, device=device)
        for i, a in enumerate(audios):
            padded[i, :lengths[i]] = a

        padded = padded * self.scale

        # [B, num_frames, frame_length]
        frames = padded.unfold(1, self.frame_length, self.hop_length)
        frame_counts = [(l - self.frame_length) // self.hop_length + 1 for l in lengths]
        max_valid = max(frame_counts)
        frames = frames[:, :max_valid, :]

        # DC offset removal
        frames = frames - frames.mean(dim=2, keepdim=True)

        # Pre-emphasis
        preemph = torch.empty_like(frames)
        preemph[:, :, 0] = frames[:, :, 0] * (1.0 - self.preemphasis_coeff)
        preemph[:, :, 1:] = frames[:, :, 1:] - self.preemphasis_coeff * frames[:, :, :-1]

        # Window
        preemph = preemph * self.window

        # Zero-pad to fft_length → rfft → power spectrum
        padded_frames = torch.nn.functional.pad(
            preemph, (0, self.fft_length - self.frame_length)
        )
        power = torch.fft.rfft(padded_frames).abs().square()  # [B, T, 257]

        # Mel filterbank + log
        mel = torch.matmul(power, self.mel_filters)             # [B, T, 80]
        mel = torch.clamp(mel, min=self.mel_floor)
        mel = torch.log(mel)

        # Attention mask for valid frames
        arange = torch.arange(max_valid, device=device).unsqueeze(0)  # [1, T]
        fc_tensor = torch.tensor(frame_counts, device=device).unsqueeze(1)  # [B, 1]
        attention_mask = (arange < fc_tensor).long()  # [B, T]

        # Per-mel-bin normalization  (zero-mean, unit-var with ddof=1)
        for i in range(batch_size):
            fc = frame_counts[i]
            seg = mel[i, :fc]                            # [fc, 80]
            mean = seg.mean(dim=0)                       # [80]
            var = seg.var(dim=0, unbiased=True)           # [80]
            mel[i, :fc] = (seg - mean) / torch.sqrt(var + 1e-7)

        # Pad to multiple of stride
        remainder = max_valid % self.stride
        if remainder != 0:
            pad_n = self.stride - remainder
            mel = torch.nn.functional.pad(mel, (0, 0, 0, pad_n))
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_n))
            max_valid += pad_n

        # Stride reshape: [B, T, 80] → [B, T//s, 80*s]
        mel = mel.reshape(batch_size, max_valid // self.stride,
                          self.num_mel_bins * self.stride)

        # Attention mask stride (keep index % stride == 1)
        indices = torch.arange(max_valid, device=device)
        attention_mask = attention_mask[:, indices % self.stride == 1]

        return {"input_features": mel, "attention_mask": attention_mask}


@dataclass
class DataPreprocessorReqData:
    text: str
    audio: torch.Tensor
    emo_audio: torch.Tensor
    audio_sr: int
    emo_sr: int
    file_rel_path: str
    original_index: int
    speaker_id: str = None


class DataPreprocessor(Process):
    def __init__(
        self,
        model_dir: str,
        input_queue: Queue,
        output_queue: Queue,
        worker_id: int = 0,
        gpu_id: int = 0,
        daemon = True
    ):
        super().__init__(daemon=daemon)
        self.model_dir = model_dir
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.input_queue = input_queue
        self.output_queue = output_queue

        # 缓存 GPU 上的重采样器
        self.resamplers = {} 

    def init_models(self):
        cfg_path = os.path.join(self.model_dir, "config.yaml")
        cfg = OmegaConf.load(cfg_path)
        cfg.gpt.number_text_tokens = 12000  # 使用原始权重

        self.device = torch.device(f"cuda")  # :{self.gpu_id}
        self.dtype = torch.float32

        bpe_path = BPE_MODEL_PATH
        normalizer = TextNormalizer()
        normalizer.load()
        self.tokenizer = TextTokenizer(bpe_path, normalizer)

        self.gpt = UnifiedVoice(**cfg.gpt)
        gpt_path = os.path.join(self.model_dir, "gpt.pth")
        load_checkpoint(self.gpt, gpt_path)
        self.gpt = self.gpt.to(self.device)
        self.gpt.eval() # 确保进入eval模式
        logger.info(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] gpt initializing...')

        hf_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            os.path.join(self.model_dir, "w2v-bert-2.0"),
        )
        self.gpu_feature_extractor = GPUFeatureExtractor(
            mel_filters_np=hf_extractor.mel_filters,
            window_np=hf_extractor.window,
        ).to(self.device)
        self.gpu_feature_extractor.eval()
        del hf_extractor
        logger.info(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] gpu_feature_extractor initializing...')

        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(self.model_dir, cfg.w2v_stat),
            os.path.join(self.model_dir, "w2v-bert-2.0")
        )
        self.semantic_model = self.semantic_model.to(device=self.device, dtype=self.dtype)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(device=self.device, dtype=self.dtype)
        self.semantic_std = self.semantic_std.to(device=self.device, dtype=self.dtype)
        logger.info(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] semantic_model initializing...')

        self.semantic_codec = build_semantic_codec(cfg.semantic_codec)
        semantic_code_ckpt = os.path.join(self.model_dir, "semantic_codec/model.safetensors")
        safetensors.torch.load_model(self.semantic_codec, semantic_code_ckpt)
        self.semantic_codec = self.semantic_codec.to(device=self.device, dtype=self.dtype)
        self.semantic_codec.eval()
        logger.info(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] semantic_codec initializing...')

        # 关闭所有梯度
        for param in self.gpt.parameters():
            param.requires_grad = False
        for param in self.semantic_model.parameters():
            param.requires_grad = False
        for param in self.semantic_codec.parameters():
            param.requires_grad = False

    def get_resampler(self, orig_freq):
        """在 GPU 上获取或创建重采样器"""
        if orig_freq not in self.resamplers:
            self.resamplers[orig_freq] = torchaudio.transforms.Resample(
                orig_freq=orig_freq,
                new_freq=TARGET_SR
            ).to(self.device)
        return self.resamplers[orig_freq]

    def healthy_check(self):
        try:
            fake_audio = torch.zeros(16000, dtype=torch.float32)
            fake_text = "123"
            self.preprocess(fake_text, fake_audio, fake_audio)
        except Exception as e:
            logger.error(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] healthy check error: {e}')
            return False
        return True

    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        init_flag = False
        for _ in range(3):
            try:
                self.init_models()
                if self.healthy_check():
                    logger.info(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] init models success')
                    init_flag = True
                    break
            except Exception as e:
                logger.error(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] init models error: {e}')

        while init_flag:
            try:
                input_data: List[DataPreprocessorReqData] = self.input_queue.get(timeout=10)
            except queue.Empty:
                continue

            if input_data is None:
                break

            try:
                processed_results = self.preprocess_batch_logic(input_data)
                self.output_queue.put(processed_results)
            except Exception as e:
                logger.error(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] preprocess error: {traceback.format_exc()}')
                if not self.healthy_check():
                    break

    @torch.no_grad()
    def preprocess_batch_logic(self, input_data_list: List[DataPreprocessorReqData]):
        texts = [d.text for d in input_data_list]
        
        audio_tensors = []
        emo_audio_tensors = []
        for d in input_data_list:
            wav = d.audio.to(self.device, non_blocking=True)
            if d.audio_sr != TARGET_SR:
                resampler = self.get_resampler(d.audio_sr)
                wav = resampler(wav)
            audio_tensors.append(wav)

            emo_wav = d.emo_audio.to(self.device, non_blocking=True)
            if d.emo_sr != TARGET_SR:
                resampler = self.get_resampler(d.emo_sr)
                emo_wav = resampler(emo_wav)
            emo_audio_tensors.append(emo_wav)
        
        processed_datas = self.preprocess_batch(texts, audio_tensors, emo_audio_tensors)
        
        results = []
        for i, p_data in enumerate(processed_datas):
            orig_idx = input_data_list[i].original_index
            results.append((input_data_list[i].file_rel_path, orig_idx, p_data))
            
        return results

    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    @torch.no_grad()
    def preprocess(
        self,
        text: str,
        audio: torch.Tensor,
        emo_audio: torch.Tensor,
    ):
        """
        audio: torch.Tensor, [audio_len], sampling_rate=16000, torch.float32, [-1, 1]
        emo_audio: torch.Tensor, [audio_len], sampling_rate=16000, torch.float32, [-1, 1]
        """
        duration = audio.shape[0] / 16000
        # Tokenize Text
        text_tokens_list = self.tokenizer.tokenize(text)
        text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        if len(text_ids) == 0:
            raise ValueError("text_ids is empty")
        text_ids = torch.tensor(text_ids, dtype=torch.int16)  # [text_len]
        text_len = len(text_ids)

        # Extract Features (GPU)
        audio_gpu = audio.to(self.device) if not audio.is_cuda else audio
        inputs = self.gpu_feature_extractor([audio_gpu])
        input_features = inputs["input_features"]
        attention_mask = inputs["attention_mask"]
        
        # Get Speaker Condition Embedding
        spk_cond_emb = self.get_emb(input_features, attention_mask)

        # Quantize / Codec
        cond_lengths = attention_mask.sum(dim=1).long()
        semantic_code, _ = self.semantic_codec.quantize(spk_cond_emb)
        semantic_code = semantic_code.squeeze(0)
        code_len = semantic_code.shape[0]

        # Get Conditioning
        feat_t = spk_cond_emb.transpose(1, 2)
        cond_lengths_device = cond_lengths.to(spk_cond_emb.device)
        
        conditioning = self.gpt.get_conditioning(feat_t, cond_lengths_device).squeeze(0)  # [32, 1280]
        
        # emo_vec: 全部使用情感音频 (GPU)
        emo_audio_gpu = emo_audio.to(self.device) if not emo_audio.is_cuda else emo_audio
        inputs_emo = self.gpu_feature_extractor([emo_audio_gpu])
        input_features_emo = inputs_emo["input_features"]
        attention_mask_emo = inputs_emo["attention_mask"]
        spk_cond_emb_emo = self.get_emb(input_features_emo, attention_mask_emo)
        cond_lengths_emo = attention_mask_emo.sum(dim=1).long()
        cond_lengths_device_emo = cond_lengths_emo.to(spk_cond_emb_emo.device)
        emo_vec = self.gpt.get_emovec(spk_cond_emb_emo, cond_lengths_device_emo).squeeze(0)  # [1280]

        processed_data = ProcessedData(
            text_ids=text_ids.to(device="cpu", dtype=torch.int16),
            codes=semantic_code.to(device="cpu", dtype=torch.int16),
            text_len=text_len,
            code_len=code_len,
            condition=conditioning.to(device="cpu", dtype=torch.float16),
            emo_vec=emo_vec.to(device="cpu", dtype=torch.float16),
            duration=duration,
        )
        return processed_data

    @torch.no_grad()
    def preprocess_batch(
        self,
        texts: List[str],
        audios: List[torch.Tensor],
        emo_audios: List[torch.Tensor],
    ):
        """
        audio: torch.Tensor, [audio_len], sampling_rate=16000, torch.float32, [-1, 1]
        emo_audios: torch.Tensor, [audio_len], sampling_rate=16000, torch.float32, [-1, 1]
        """
        batch_size = len(texts)
        durations = [audio.shape[0] / 16000 for audio in audios]

        # Tokenize Text
        text_ids_list = []
        text_lens = []
        for text in texts:
            text_tokens_list = self.tokenizer.tokenize(text)
            text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
            text_ids = torch.tensor(text_ids, dtype=torch.int16)
            text_len = len(text_ids)

            text_ids_list.append(text_ids)
            text_lens.append(text_len)

        # Extract Features (GPU)
        inputs = self.gpu_feature_extractor(audios)
        input_features = inputs["input_features"]
        attention_mask = inputs["attention_mask"]
        
        # Get Speaker Condition Embedding
        spk_cond_emb = self.get_emb(input_features, attention_mask)

        # Quantize / Codec
        cond_lengths = attention_mask.sum(dim=1).long()
        semantic_code, _ = self.semantic_codec.quantize(spk_cond_emb)  # [b, code_len]

        semantic_codes = []
        code_lens = []
        for b in range(batch_size):
            semantic_code_ = semantic_code[b, :cond_lengths[b]]
            semantic_codes.append(semantic_code_)
            code_lens.append(semantic_code_.shape[0])

        # Get Conditioning
        feat_t = spk_cond_emb.transpose(1, 2)
        cond_lengths_device = cond_lengths.to(spk_cond_emb.device)
        
        conditioning = self.gpt.get_conditioning(feat_t, cond_lengths_device)  # [b, 32, 1280]
        
        # Extract Features for emo audios (GPU)
        inputs_emo = self.gpu_feature_extractor(emo_audios)
        input_features_emo = inputs_emo["input_features"]
        attention_mask_emo = inputs_emo["attention_mask"]
        
        spk_cond_emb_emo = self.get_emb(input_features_emo, attention_mask_emo)
        cond_lengths_emo = attention_mask_emo.sum(dim=1).long()
        cond_lengths_device_emo = cond_lengths_emo.to(spk_cond_emb_emo.device)
        
        emo_vec = self.gpt.get_emovec(spk_cond_emb_emo, cond_lengths_device_emo)  # [b, 1280]

        conditioning = conditioning.to(device="cpu", dtype=torch.float16)
        emo_vec = emo_vec.to(device="cpu", dtype=torch.float16)

        processed_datas = []
        for b in range(batch_size):
            processed_data = ProcessedData(
                text_ids=text_ids_list[b],
                codes=semantic_codes[b].to(device="cpu", dtype=torch.int16),
                text_len=text_lens[b],
                code_len=code_lens[b],
                condition=conditioning[b].clone(),
                emo_vec=emo_vec[b].clone(),
                duration=durations[b],
            )
            processed_datas.append(processed_data.to_numpy())
        return processed_datas


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

    def get_emo_synthesis_path(self, original_parquet_path):
        """
        根据原始parquet路径获取对应的情感音频parquet路径
        """
        rel_path = os.path.relpath(original_parquet_path, DATASET_ROOT)
        dataset_name = rel_path.split(os.sep)[0]
        
        emo_synthesis_dataset_name = f"{dataset_name}"
        emo_synthesis_rel_path = os.path.join(emo_synthesis_dataset_name, os.sep.join(rel_path.split(os.sep)[1:]))
        emo_synthesis_path = os.path.join(EMO_SYNTHESIS_ROOT, emo_synthesis_rel_path)
        
        return emo_synthesis_path

    def run(self):
        logger.info(f"[CPU-Loader-{self.worker_id}] Started.")
        batch_req = []
        while True:
            try:
                parquet_path = self.file_queue.get(timeout=3)
            except queue.Empty:
                break

            if parquet_path is None:
                break

            try:
                # 计算相对路径，用于保持输出结构
                rel_path = os.path.relpath(parquet_path, DATASET_ROOT)
                output_path = os.path.join(OUTPUT_DIR, rel_path.replace(".parquet", ".pkl"))
                if os.path.exists(output_path):
                    logger.error(f"[CPU-Loader-{self.worker_id}] Skipping {rel_path} as it already exists.")
                    continue

                emo_synthesis_parquet_path = self.get_emo_synthesis_path(parquet_path)
                if not os.path.exists(emo_synthesis_parquet_path):
                    logger.warning(f"[CPU-Loader-{self.worker_id}] Emo synthesis file not found: {emo_synthesis_parquet_path}, skipping {rel_path}")
                    continue

                parquet_file = pq.ParquetFile(parquet_path)
                emo_synthesis_parquet_file = pq.ParquetFile(emo_synthesis_parquet_path)
                
                orig_num_rows = parquet_file.metadata.num_rows
                emo_num_rows = emo_synthesis_parquet_file.metadata.num_rows
                if orig_num_rows != emo_num_rows:
                    logger.warning(f"[CPU-Loader-{self.worker_id}] Row count mismatch: original={orig_num_rows}, emo_synthesis={emo_num_rows}, skipping {rel_path}")
                    continue
                
                valid_count = 0
                duration_skip_num = 0
                non_audio_or_text_skip_num = 0
                pf_start_time = time.time()
                pbar = tqdm(desc=f"CPU-Loader-{self.worker_id}: {rel_path}")

                global_row_offset = 0 

                orig_batches = parquet_file.iter_batches(batch_size=256, columns=['audio', 'text'])
                emo_batches = emo_synthesis_parquet_file.iter_batches(batch_size=256, columns=['audio', 'text'])
                for batch, emo_batch in zip(orig_batches, emo_batches):
                    audio_col = batch['audio']
                    text_col = batch['text']
                    emo_audio_col = emo_batch['audio']
                    batch_len = len(batch)

                    for i in range(batch_len):
                        # 计算当前行在整个Parquet文件中的绝对下标
                        current_file_index = global_row_offset + i
                        
                        text = str(text_col[i])

                        try:
                            array, sampling_rate = sf.read(io.BytesIO(audio_col[i]), dtype='float32')
                            duration = array.shape[0] / sampling_rate
                            if duration > MAX_AUDIO_DURATION:
                                duration_skip_num += 1
                                continue

                            if array.ndim > 1:
                                array = np.mean(array, axis=1)
                            
                            audio_tensor = torch.from_numpy(array).float()
                            audio_tensor.share_memory_()
                            
                            if not emo_audio_col[i] or len(emo_audio_col[i].as_py()) == 0:
                                continue
                            
                            emo_array, emo_sampling_rate = sf.read(io.BytesIO(emo_audio_col[i]), dtype='float32')
                            if emo_array.ndim > 1:
                                emo_array = np.mean(emo_array, axis=1)
                            if len(emo_array) == 0:
                                continue
                            
                            emo_audio_tensor = torch.from_numpy(emo_array).float()
                            emo_audio_tensor.share_memory_()

                            req = DataPreprocessorReqData(
                                text=text,
                                audio=audio_tensor,  # float32
                                emo_audio=emo_audio_tensor,  # float32
                                audio_sr=sampling_rate,
                                emo_sr=emo_sampling_rate,
                                file_rel_path=rel_path,
                                original_index=current_file_index
                            )
                            batch_req.append(req)
                            valid_count += 1

                            if len(batch_req) >= BATCH_SIZE:
                                self.gpu_task_queue.put(batch_req)
                                batch_req = []

                        except Exception as e:
                            logger.error(f"Error processing audio in {rel_path}: {traceback.format_exc()}")
                            continue
                        pbar.update(1)
                    
                    # 更新全局偏移量
                    global_row_offset += batch_len

                logger.error(f"[CPU-Loader-{self.worker_id}: {rel_path}] Samples: {valid_count}. Duration skip: {duration_skip_num}, time: {time.time()-pf_start_time:.4f}s")
                            
                if len(batch_req) > 0:
                    self.gpu_task_queue.put(batch_req)
                    batch_req = []
                
                if valid_count > 0:
                    manifest = FileManifest(
                        file_rel_path=rel_path,
                        total_samples=valid_count,
                        original_path=parquet_path
                    )
                    self.writer_control_queue.put(manifest)

            except Exception as e:
                logger.error(f"[CPU-Loader-{self.worker_id}] File Error {parquet_path}: {traceback.format_exc()}")


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
        pbar = tqdm(total=self.total_files, unit="file", desc="Processing Parquet Files")
        
        while finished_files_count < self.total_files:
            while not self.writer_control_queue.empty():
                try:
                    manifest: FileManifest = self.writer_control_queue.get_nowait()
                    if manifest.file_rel_path not in self.file_buffers:
                        logger.error(f"[Writer] manifest.file_rel_path not in file_buffers: {manifest.file_rel_path}")
                    else:
                        self.file_buffers[manifest.file_rel_path]['expected'] = manifest.total_samples
                except queue.Empty:
                    break

            try:
                res = self.result_queue.get(timeout=0.1)
                for res_ in res:
                    file_rel_path, original_index, processed_data = res_
                    
                    if file_rel_path not in self.file_buffers:
                        self.file_buffers[file_rel_path] = {
                            'expected': -1,
                            'data': [],
                            'received': 0,
                        }
                    
                    data_item = {
                        "index": original_index,
                        "data": processed_data
                    }
                    self.file_buffers[file_rel_path]['data'].append(data_item)
                    
                    self.file_buffers[file_rel_path]['received'] += 1
                    
                    # 检查是否完成
                    buf = self.file_buffers[file_rel_path]
                    if buf['expected'] != -1 and buf['received'] >= buf['expected'] * 0.9:
                        try:
                            self.save_file(file_rel_path, buf['data'])
                        finally:
                            del self.file_buffers[file_rel_path]
                            finished_files_count += 1
                            pbar.update(1)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[Writer] Error: {traceback.format_exc()}")

        pbar.close()
        logger.info("[Writer] All files processed.")

    def save_file(self, rel_path, data_list):
        # 为了方便后续使用，建议按 index 排序
        data_list.sort(key=lambda x: x['index'])
        
        output_path = os.path.join(OUTPUT_DIR, rel_path.replace(".parquet", ".pkl"))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, "wb") as f:
                pickle.dump(data_list, f)
        except Exception as e:
            logger.error(f"Failed to save {output_path}: {traceback.format_exc()}")


def main():
    if os.path.exists(OUTPUT_DIR):
        logger.warning(f"Output directory {OUTPUT_DIR} already exists.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Scanning files...")
    all_parquet_files = []
    for folder in DATASET_ROOTS:
        parquet_files = glob.glob(os.path.join(folder, "*.parquet"))
        all_parquet_files.extend(parquet_files)

    all_parquet_files = all_parquet_files  # [:CPU_WORKERS_NUM]

    total_files = len(all_parquet_files)
    logger.info(f"Found {total_files} parquet files.")

    if total_files == 0:
        return

    file_queue = Queue() 
    gpu_task_queue = Queue(maxsize=MAX_GPU_TASK_QUEUE_SIZE)
    writer_control_queue = Queue() 
    result_queue = Queue()

    for f in all_parquet_files:
        file_queue.put(f)

    writer = ResultWriterWorker(result_queue, writer_control_queue, total_files)
    writer.start()

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

    cpu_workers = []
    for i in range(CPU_WORKERS_NUM):
        p = AudioLoaderWorker(file_queue, gpu_task_queue, writer_control_queue, i)
        p.start()
        cpu_workers.append(p)
    logger.info(f"Started {len(cpu_workers)} CPU loaders.")

    for p in cpu_workers:
        p.join()
    logger.info("All CPU loaders finished.")

    for _ in gpu_workers:
        gpu_task_queue.put(None)
    
    for p in gpu_workers:
        p.join()
    logger.info("All GPU workers finished.")

    writer.join()
    logger.info("Writer finished. Done.")

if __name__ == "__main__":
    main()