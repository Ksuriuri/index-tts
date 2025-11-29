import os
from pathlib import Path
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from omegaconf import OmegaConf
import sentencepiece as spm
from indextts.utils.front import TextNormalizer, TextTokenizer

model_dir = "./checkpoints/IndexTTS-2-vLLM"
cfg_path = os.path.join(model_dir, "config.yaml")
cfg = OmegaConf.load(cfg_path)

# bpe_path = os.path.join(model_dir, cfg.dataset["bpe_model"])
bpe_path = os.path.join(model_dir, "japanese_bpe.model")

# sp = spm.SentencePieceProcessor(model_file=bpe_path)

normalizer = TextNormalizer()
normalizer.load()
tokenizer = TextTokenizer(bpe_path, normalizer)
max_text_tokens_per_segment = 120
quick_streaming_tokens = 0

# 查看是否有语言相关属性（通常需要看训练时的配置）
print(f"词汇表大小: {tokenizer.sp_model.get_piece_size()}")


# 简单测试几种语言
test_sentences = {
    "中文": "你好，今天天气不错",
    "英文": "Hello, the weather is nice today",
    "日文": "こんにちは、今日はいい天気ですね",
    # "韩文": "안녕하세요, 오늘 날씨가 좋네요",
    # "法文": "Bonjour, il fait beau aujourd'hui"
    "数字": "Count 1 2 3, 然后说456 7 8",
    "特殊符号": "!@#$%^&*()_+-=[]{};':\"\\|,.<>/?`~"
}

print("=== 分词测试结果 ===")
for lang, text in test_sentences.items():
    # tokens = sp.encode_as_pieces(text)
    # ids = sp.encode_as_ids(text)
    
    text_tokens_list = tokenizer.tokenize(text)
    text_token_ids = tokenizer.convert_tokens_to_ids(text_tokens_list)

    # print(text_tokens_list)
    # print(segments)
    # print(text_token_ids)
    # print()
    
    # 统计unk标记数量
    # unk_count = tokens.count("<unk>")
    unk_count = text_token_ids.count(2)
    unk_ratio = unk_count / len(text_tokens_list) if text_tokens_list else 0
    
    print(f"\n{lang}: {text}")
    print(f"  Tokens: {text_tokens_list[:50]}{'...' if len(text_tokens_list)>50 else ''} (共{len(text_tokens_list)}个)")
    print(f"  IDs: {text_token_ids[:50]}{'...' if len(text_token_ids)>50 else ''}")
    print(f"  <unk>数量: {unk_count} (占比: {unk_ratio:.1%})")
    
    # 简单判断
    if unk_ratio < 0.1:
        print(f"  ✅ 可能支持{lang}")
    else:
        print(f"  ❌ 可能不支持{lang}")