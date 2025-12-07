import os
import random
import sys
import pickle
import glob
from typing import Dict
import numpy as np
from tqdm import tqdm  # 用于显示进度条
from loguru import logger

import io
from datasets import load_dataset, Features, Value
import torchaudio
import soundfile as sf
import pyarrow.parquet as pq

# 原始路径设置
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

import librosa
import torch

def _load_audio(audio_path, max_audio_length_seconds, verbose=False, sr=None):
    try:
        if not sr:
            audio, sr = librosa.load(audio_path)
        else:
            audio, _ = librosa.load(audio_path, sr=sr)
        
        audio = torch.tensor(audio).unsqueeze(0)
        max_audio_samples = int(max_audio_length_seconds * sr)
        
        if audio.shape[1] > max_audio_samples:
            if verbose: logger.info(f"Skipping {audio_path}: Audio too long.")
            return None, None
        return audio, sr
    except Exception as e:
        logger.info(f"Error loading audio {audio_path}: {e}")
        return None, None
    
def from_wav_with_lab(wav_path, max_audio_duration=36):
    # 构建对应的 lab 文件路径
    lab_path = os.path.splitext(wav_path)[0] + ".lab"
    
    if not os.path.exists(lab_path):
        raise FileNotFoundError(f"Missing label file for {wav_path}")
    
    with open(lab_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    if not text:
        logger.info(f"Warning: Empty text in {lab_path}")
        raise ValueError("Empty text in {lab_path}")

    audio, sr = _load_audio(wav_path, max_audio_duration, sr=16000)
    if audio is None:
        raise ValueError(f"Error loading audio {wav_path}")
    
    return text, audio


def from_hf_parquet(data_dir, target_sr):
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    total_files_num = 0

    resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    for folder in subfolders[:1]:  # 
        # game_name = os.path.basename(folder)
        
        parquet_files = glob.glob(os.path.join(folder, "*.parquet"))
        total_files_num += len(parquet_files)
        
        if not parquet_files:
            continue
        
        for parquet_path in parquet_files[:1]:
            pf = pq.ParquetFile(parquet_path)

            # ds = load_dataset("parquet", data_files=parquet_path, split="train", streaming=True)
            # ds = ds.cast_column("audio", Features({
            #     "bytes": Value("large_binary"), 
            #     "path": Value("string")
            # }))
            
            # 遍历当前数据集
            # for sample in tqdm(ds, desc=f"Loading {parquet_path}"):
            pbar = tqdm(desc=f"Loading {parquet_path}")
            for batch in pf.iter_batches(batch_size=128, columns=['audio', 'text']):
                pylist = batch.to_pylist()
                for sample in pylist:
                    # print(sample.keys())
                    audio_bytes = sample['audio']['bytes']
                    if audio_bytes:
                        try:
                            array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
                            duration = array.shape[0] / sampling_rate
                            waveform = torch.from_numpy(array).float()
                            if waveform.dim() > 1:
                                waveform = torch.mean(waveform, dim=1)
                            if sampling_rate != target_sr:
                                if sampling_rate not in resamplers:
                                    resamplers[sampling_rate] = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sr)
                                waveform = resamplers[sampling_rate](waveform)
                                # waveform = torchaudio.functional.resample(waveform, orig_freq=sampling_rate, new_freq=target_sr)
                            # waveform.share_memory_()
                            array = waveform.numpy()
                            print(array.max(), array.min(), array.mean())
                            sf.write("outputs/temp.wav", array, target_sr)
                            # pbar.update(1)
                        except Exception as e:
                            print(f"手动解码失败: {e}")
                    break
                break
    print(f"总共处理了 {total_files_num} 个文件")


if __name__ == "__main__":
    from_hf_parquet("/mnt/data_3t_1/datasets/raw_data/Galgame-VisualNovel-Reupload", target_sr=16000)
    


