import os
import glob
import json
import pandas as pd
import MeCab
import unidic_lite
from jiwer import cer
from tqdm import tqdm
import re

# --- 配置 (Configuration) ---
# 现在支持列表格式 (Supports a list of paths)
# DATASET_NAME = "Galgame-VisualNovel-Reupload"
# DATASET_NAME = "Gacha_games_jp"
# DATASET_NAME = "Emilia_JA"
# DATASET_NAME = "Emilia-YODAS_JA"
INPUT_DIRS = [
    "/mnt/data_3t_1/datasets/preprocess/Galgame-VisualNovel-Reupload",
    "/mnt/data_3t_1/datasets/preprocess/Gacha_games_jp",
    "/mnt/data_3t_1/datasets/preprocess/Emilia_JA",
    "/mnt/data_3t_1/datasets/preprocess/Emilia-YODAS_JA",
    "/mnt/data_3t_1/datasets/preprocess/Japanese-Eroge-Voice",
]
# 如果想覆盖原文件，保持为 None；如果想保存到新目录，请填写新路径
OUTPUT_DIR = None

class JapaneseNormalizer:
    def __init__(self):
        # 使用 unidic_lite 字典
        self.tagger = MeCab.Tagger(f"-d '{unidic_lite.DICDIR}'")
        
    def to_kana(self, text: str) -> str:
        if not text or pd.isna(text):
            return ""
        
        text = str(text).replace(" ", "").strip()
        node = self.tagger.parseToNode(text)
        kana_list = []
        
        while node:
            if not node.surface:
                node = node.next
                continue

            features = node.feature.split(',')
            
            # unidic-lite feature index: 9 is pron, 6 is lemma/lForm
            token_kana = ""
            if len(features) > 9 and features[9] != "*":
                token_kana = features[9]
            elif len(features) > 6 and features[6] != "*":
                token_kana = features[6]
            else:
                token_kana = node.surface
            
            kana_list.append(self._hira_to_kata(token_kana))
            node = node.next
            
        return "".join(kana_list)

    def _hira_to_kata(self, text):
        return "".join([chr(ord(c) + 96) if 12353 <= ord(c) <= 12438 else c for c in text])

def process_parquet_file(file_path, normalizer, output_root=None):
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
        if df.empty:
            return 0

        def calculate_row_pron_cer(row):
            whisper_data = row.get('whisper_large_v3')
            if whisper_data is None or not isinstance(whisper_data, dict):
                return whisper_data

            text_gt = row.get('text', "")
            text_pred = whisper_data.get('text', "")
            
            kana_gt = normalizer.to_kana(text_gt)
            kana_pred = normalizer.to_kana(text_pred)
            
            if not kana_gt:
                pron_cer = 0.0 if not kana_pred else 1.0
            else:
                pron_cer = cer(kana_gt, kana_pred)
            
            new_whisper_data = whisper_data.copy()
            new_whisper_data['pron_CER'] = float(pron_cer)
            return new_whisper_data

        tqdm.pandas(desc=f"Processing {os.path.basename(file_path)}", leave=False)
        df['whisper_large_v3'] = df.progress_apply(calculate_row_pron_cer, axis=1)

        # 确定保存路径
        if output_root:
            # 保持子目录结构 (Keep sub-directory structure if possible)
            # 这里简单处理：如果多个输入目录有同名文件，建议 output_root 不为空时手动区分
            os.makedirs(output_root, exist_ok=True)
            save_path = os.path.join(output_root, os.path.basename(file_path))
        else:
            save_path = file_path
        
        df.to_parquet(save_path, engine='pyarrow', index=False)
        return len(df)

    except Exception as e:
        print(f"\nError processing {file_path}: {e}")
        return 0

def main():
    # 确保 INPUT_DIRS 是列表
    input_paths = INPUT_DIRS if isinstance(INPUT_DIRS, list) else [INPUT_DIRS]
    
    all_files = []
    for directory in input_paths:
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            continue
        
        # 扫描当前目录下的所有 parquet
        found = glob.glob(os.path.join(directory, "*.parquet"))
        print(f"Found {len(found)} files in {directory}")
        all_files.extend(found)

    if not all_files:
        print("No parquet files found to process.")
        return

    print(f"Total files to process: {len(all_files)}")
    normalizer = JapaneseNormalizer()
    
    total_samples = 0
    # 使用主进度条遍历所有文件
    for f in tqdm(all_files, desc="Overall Progress"):
        count = process_parquet_file(f, normalizer, OUTPUT_DIR)
        total_samples += count
        
    print(f"\nAll done! Processed {total_samples} samples across {len(all_files)} files.")

if __name__ == "__main__":
    main()