import os
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm

# 假设你的原始代码保存在 text_utils.py 中
# 如果没有，请将 TextNormalizer 类的定义复制到这里
try:
    from text_utils import TextNormalizer
except ImportError:
    # 为了演示方便，这里内联一个简化版的 TextNormalizer，逻辑与你提供的一致
    import re
    import unicodedata
    class TextNormalizer:
        def __init__(self):
            self.jp_char_rep_map = {
                "：": ",", "；": ",", ";": ",", "，": ",", "。": ".",
                "！": "!", "？": "?", "\n": " ", "·": "-", "、": ",",
                "...": "…", ",,,": "…", "……": "…",
                "“": "'", "”": "'", '"': "'", "「": "'", "」": "'",
                "（": "'", "）": "'", "(": "'", ")": "'",
                "—": "-", "～": "-", "~": "-", ":": ",",
            }
            self._jp_cleanup_pattern = re.compile("|".join(re.escape(p) for p in self.jp_char_rep_map.keys()))

        def normalize_japanese(self, text: str) -> str:
            text = text.strip()
            if not text: return ""
            # 移除 speaker 标记 (和你代码一致)
            text = re.sub(r"^\s*(?:speaker|spk)\s*\d+\s*[:：]\s*", "", text, flags=re.IGNORECASE)
            text = unicodedata.normalize("NFKC", text)
            text = re.sub(r"\s+", " ", text)
            # 关键：应用字符映射
            text = self._jp_cleanup_pattern.sub(lambda x: self.jp_char_rep_map[x.group()], text)
            return text.strip()

def train_bpe_model():
    # --- 配置参数 ---
    VOCAB_SIZE = 32000          # 词表大小，推荐 16000 或 32000
    CHARACTER_COVERAGE = 0.9995 # 日语包含大量汉字，0.9995 可以覆盖绝大多数，剩下的转为 UNK
    MODEL_PREFIX = "japanese_bpe"
    TRAIN_FILE = "corpus_jp_normalized.txt"
    MAX_SENTENCES = 5000000     # 使用500万个句子进行训练，通常足够收敛

    # --- 1. 准备 Normalizer ---
    print("Initializing Normalizer...")
    normalizer = TextNormalizer()

    # --- 2. 准备数据 (使用 HuggingFace Wikipedia) ---
    print("Loading Japanese Wikipedia dataset...")
    # 使用 streaming=True 避免一次性下载整个维基百科，节省内存和磁盘
    dataset = load_dataset("wikipedia", "20220301.ja", split="train", streaming=True, trust_remote_code=True)

    print(f"Processing dataset and saving to {TRAIN_FILE}...")
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        count = 0
        for data in tqdm(dataset):
            text = data.get("text", "")
            if not text:
                continue
            
            # 按行分割，避免把整篇文章当做一个长句子，有助于 SentencePiece 学习
            lines = text.split('\n')
            for line in lines:
                # *** 核心步骤：使用与推理时完全相同的 Normalizer ***
                # 这样模型就会学习到 "." 而不是 "。" 是句号
                norm_text = normalizer.normalize_japanese(line)
                
                if len(norm_text) > 10: # 过滤太短的垃圾数据
                    f.write(norm_text + "\n")
                    count += 1
            
            if count >= MAX_SENTENCES:
                break
    
    print(f"Total sentences collected: {count}")

    # --- 3. 训练 SentencePiece 模型 ---
    # 参数说明：
    # --input: 输入文件
    # --model_prefix: 输出模型前缀
    # --vocab_size: 词表大小
    # --character_coverage: 字符覆盖率
    # --model_type: bpe
    # --bos_id=0, --eos_id=1, --unk_id=2: 对齐你现有 tokenizer 的 ID 逻辑
    # --pad_id=3: 为 pad 预留位置 (虽然你代码里是-1，但SP模型内部需要物理ID)
    
    train_command = (
        f"--input={TRAIN_FILE} "
        f"--model_prefix={MODEL_PREFIX} "
        f"--vocab_size={VOCAB_SIZE} "
        f"--character_coverage={CHARACTER_COVERAGE} "
        f"--model_type=bpe "
        f"--bos_id=0 "
        f"--eos_id=1 "
        f"--unk_id=2 "
        f"--pad_id=3 "
        f"--input_sentence_size=5000000 " 
        f"--shuffle_input_sentence=true "
        f"--normalization_rule_name=identity" # 既然我们已经外部 Normalize 过了，这里设为 identity 避免重复处理
    )

    print("Training SentencePiece model...")
    spm.SentencePieceTrainer.train(train_command)
    
    print(f"Done! Model saved to {MODEL_PREFIX}.model and {MODEL_PREFIX}.vocab")

    # --- 4. 简单测试 ---
    print("-" * 30)
    print("Testing the trained model:")
    sp = spm.SentencePieceProcessor(model_file=f"{MODEL_PREFIX}.model")
    
    test_text = "こんにちは、世界。IndexTTSへようこそ！"
    norm_text = normalizer.normalize_japanese(test_text) # 必须先 Normalize
    tokens = sp.encode(norm_text, out_type=str)
    ids = sp.encode(norm_text, out_type=int)
    
    print(f"Original: {test_text}")
    print(f"Normalized: {norm_text}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids}")

    # 清理临时文件 (可选)
    # os.remove(TRAIN_FILE)

if __name__ == "__main__":
    train_bpe_model()