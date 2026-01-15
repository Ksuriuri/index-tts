import io
import os
import sys
import pickle
import traceback
import numpy as np
import pandas as pd
import soundfile as sf  # 需要安装: pip install soundfile
from tqdm import tqdm

# 原始路径设置 (保持与你的一致)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

# 尝试导入 ProcessedData，如果环境不对可能需要你手动调整 sys.path
try:
    from trainers.utils import ProcessedData
except ImportError:
    print("[Warn] Could not import ProcessedData. Pickle loading might fail if classes are not in path.")

# ================= 配置区域 =================
SOURCE_NAMES = [
    # "Emilia_JA",
    # "Emilia-YODAS_JA",
    "Gacha_games_jp",
    # "Galgame-VisualNovel-Reupload",
    # "Japanese-Eroge-Voice"
]

PREPROCESS_ROOT = "/mnt/data_3t_1/datasets/preprocess"
DATA_ROOT = "/mnt/data_3t_2/datasets/indextts_train_data_v2"
SAMPLE_OUTPUT_DIR = "./outputs/bad_case_samples"  # 样本保存路径
THRESHOLD_GAP = 0.5  # 超过0.5秒则保存
MAX_SAMPLES_PER_SOURCE = 10  # 每个数据源最多采多少个样

# ================= 辅助函数 =================

def get_parquet_path(pkl_path: str, source_name: str) -> str:
    try:
        path_parts = pkl_path.split(os.sep)
        if source_name in path_parts:
            idx = path_parts.index(source_name)
            rel_path = os.sep.join(path_parts[idx+1:])
            rel_path = rel_path.replace('.pkl', '.parquet')
            parquet_path = os.path.join(PREPROCESS_ROOT, source_name, rel_path)
            return parquet_path
        return None
    except ValueError:
        return None

def get_all_pkl_files(directory):
    pkl_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))
    return pkl_files

def save_debug_audio(output_dir, source_name, pkl_name, idx, audio_bytes, whisper_text, duration, last_end):
    """
    保存音频和对应的文本信息
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"{source_name}_{pkl_name.replace('.pkl','')}_{idx}"
    
    # 1. 保存音频
    wav_path = os.path.join(output_dir, f"{base_name}_GAP_{duration-last_end:.2f}s.wav")
    
    try:
        with io.BytesIO(audio_bytes) as audio_io:
            # 注意：sf.read 会返回 float64 或 float32，这里转 float32
            array, sr = sf.read(audio_io, dtype='float32')
            sf.write(wav_path, array, sr)
            
    except Exception as e:
        print(f"[Error] Failed to save wav for {base_name}: {e}")
        return

    # 2. 保存文本信息
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Source: {source_name}\n")
        f.write(f"Total Duration: {duration:.4f}s\n")
        f.write(f"Whisper Last End: {last_end:.4f}s\n")
        f.write(f"Tail Gap: {duration - last_end:.4f}s\n")
        f.write("-" * 20 + "\n")
        f.write(f"Whisper Text:\n{whisper_text}\n")

# ================= 核心逻辑 =================

def process_source_and_extract(source_name):
    source_dir = os.path.join(DATA_ROOT, source_name)
    if not os.path.exists(source_dir):
        return

    pkl_files = get_all_pkl_files(source_dir)
    print(f"--> Extracting from {source_name}...")
    
    saved_count = 0

    for pkl_path in tqdm(pkl_files, desc=f"Scanning {source_name}"):
        if saved_count >= MAX_SAMPLES_PER_SOURCE:
            break
            
        try:
            parquet_path = get_parquet_path(pkl_path, source_name)
            if not parquet_path or not os.path.exists(parquet_path):
                continue
            
            # 读取 Meta
            df_meta = pd.read_parquet(parquet_path, columns=["audio", "whisper_large_v3"])
            
            # 读取 Pickle
            with open(pkl_path, "rb") as f:
                data_list = pickle.load(f)
            if not data_list:
                continue

            for item in data_list:
                if saved_count >= MAX_SAMPLES_PER_SOURCE:
                    break

                idx = item["index"]
                processed_obj = item["data"]
                
                if idx >= len(df_meta):
                    continue
                
                whisper_info = df_meta.iloc[idx]["whisper_large_v3"]
                total_duration = processed_obj.duration
                segments = list(whisper_info.get("segments", []))
                
                if segments:
                    last_seg_end = segments[-1]["end"]
                    tail_gap = total_duration - last_seg_end
                    
                    # === 判断条件 ===
                    if tail_gap > THRESHOLD_GAP:
                        # 拼接所有文本以便查看
                        full_text = "".join([s["text"] for s in segments])
                        
                        save_debug_audio(
                            output_dir=os.path.join(SAMPLE_OUTPUT_DIR, source_name),
                            source_name=source_name,
                            pkl_name=os.path.basename(pkl_path),
                            idx=idx,
                            audio_bytes=df_meta.iloc[idx]["audio"],
                            whisper_text=full_text,
                            duration=total_duration,
                            last_end=last_seg_end
                        )
                        saved_count += 1
                        
        except Exception as e:
            # 忽略单个文件的错误，继续下一个
            continue

def main():
    if os.path.exists(SAMPLE_OUTPUT_DIR):
        print(f"Warning: Output dir {SAMPLE_OUTPUT_DIR} already exists.")
    
    for source in SOURCE_NAMES:
        process_source_and_extract(source)
    
    print(f"\nDone! Check samples in: {os.path.abspath(SAMPLE_OUTPUT_DIR)}")

if __name__ == "__main__":
    main()