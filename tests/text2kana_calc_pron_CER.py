import pandas as pd
import MeCab
import unidic_lite
from jiwer import cer
import os

# --- 配置 ---
INPUT_FILE = "/mnt/data_3t_1/datasets/preprocess/Japanese-Eroge-Voice/part_0000.parquet" # 请确保文件路径正确
SAMPLE_SIZE = 5  # 打印前 5 条结果

class JapaneseNormalizer:
    def __init__(self):
        # 初始化 MeCab，使用 unidic-lite 字典
        self.tagger = MeCab.Tagger(f"-d '{unidic_lite.DICDIR}'")
        
    def to_kana(self, text: str) -> str:
        if not text or pd.isna(text): return ""
        text = str(text).replace(" ", "").strip()
        
        node = self.tagger.parseToNode(text)
        kana_list = []
        while node:
            if node.surface:
                features = node.feature.split(',')
                # UniDic 格式: [9]是发音, [6]是词形, 都不行则用原文
                token_kana = features[9] if len(features) > 9 and features[9] != "*" else \
                             features[6] if len(features) > 6 and features[6] != "*" else \
                             node.surface
                kana_list.append(self._hira_to_kata(token_kana))
            node = node.next
        return "".join(kana_list)

    def _hira_to_kata(self, text):
        return "".join([chr(ord(c) + 96) if 12353 <= ord(c) <= 12438 else c for c in text])

def quick_test():
    if not os.path.exists(INPUT_FILE):
        print(f"找不到文件: {INPUT_FILE}")
        return

    # 1. 加载一小部分数据
    df = pd.read_parquet(INPUT_FILE, engine='pyarrow').head(SAMPLE_SIZE)
    normalizer = JapaneseNormalizer()

    print(f"{'='*20} 转换测试结果 {'='*20}\n")

    for i, row in df.iterrows():
        text_gt = row['text']
        # 假设 whisper_large_v3 存储的是字典，提取其中的 text
        whisper_dict = row['whisper_large_v3']
        text_pred = whisper_dict.get('text', "") if isinstance(whisper_dict, dict) else ""

        # 执行转换
        kana_gt = normalizer.to_kana(text_gt)
        kana_pred = normalizer.to_kana(text_pred)
        
        # 计算 CER (基于假名)
        error_rate = cer(kana_gt, kana_pred) if kana_gt else (0.0 if not kana_pred else 1.0)

        # 打印对比
        print(f"ID: {i}")
        print(f"原文 (GT):   {text_gt}")
        print(f"预测 (Pred): {text_pred}")
        print(f"假名 (GT):   {kana_gt}")
        print(f"假名 (Pred): {kana_pred}")
        print(f"Pron_CER:    {error_rate:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    quick_test()