# python .\tests\langs\print_bpe_token.py

import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from indextts.utils.front import TextNormalizer, TextTokenizer
import unicodedata
import re

def export_token_mapping(tokenizer, output_file="token_mapping.txt"):
    """
    导出所有token的映射表，以方便阅读的形式保存
    """
    import html
    
    def format_token(token):
        """格式化token，使其更易读"""
        # 处理空格
        if token == "▁":
            return "'▁'(空格)"
        elif token.startswith("▁"):
            return f"'{token}' (前缀有空格)"
        
        # 处理特殊字符
        if token in tokenizer.special_tokens_map.values():
            return f"'{token}' [特殊标记]"
        
        # 处理控制字符
        if any(ord(c) < 32 for c in token):
            escaped = "".join(f"\\x{ord(c):02x}" if ord(c) < 32 else c for c in token)
            return f"'{escaped}' (控制字符)"
        
        # 处理标点符号
        if any(unicodedata.category(c).startswith("P") for c in token):
            return f"'{token}' [标点]"
        
        # 处理数字
        if token.isdigit():
            return f"'{token}' [数字]"
        
        # 处理拼音（根据你代码中的范围）
        pinyin_pattern = re.compile(r"[a-zA-Z]+[1-5]")
        if pinyin_pattern.match(token):
            return f"'{token}' [拼音]"
        
        # 其他字符
        return f"'{token}'"
    
    # 按类别分组
    special_tokens = []
    punctuation_tokens = []
    pinyin_tokens = []
    chinese_tokens = []
    english_tokens = []
    other_tokens = []
    space_tokens = []
    
    print(f"正在生成token映射表，词汇表大小: {tokenizer.vocab_size}...")
    
    for token_id in range(tokenizer.vocab_size):
        token = tokenizer.convert_ids_to_tokens(token_id)
        
        if token in tokenizer.special_tokens_map.values():
            special_tokens.append((token_id, token))
        elif token == "▁" or token.startswith("▁"):
            space_tokens.append((token_id, token))
        elif re.match(r"[a-zA-Z]+[1-5]", token):
            pinyin_tokens.append((token_id, token))
        elif any(unicodedata.category(c).startswith("P") for c in token if c != "▁"):
            punctuation_tokens.append((token_id, token))
        # elif any('\u4e00' <= c <= '\u9fff' for c in token):
        elif any(
            '\u3400' <= c <= '\u4dbf'  # 扩展A
            or '\u4e00' <= c <= '\u9fff'  # 基本区
            or '\U00020000' <= c <= '\U0002a6df'  # 扩展B
            or '\U0002a700' <= c <= '\U0002b73f'  # 扩展C
            or '\U0002b740' <= c <= '\U0002b81f'  # 扩展D
            or '\U0002b820' <= c <= '\U0002ceaf'  # 扩展E
            or '\U0002ceb0' <= c <= '\U0002ebef'  # 扩展F
            or '\U00030000' <= c <= '\U0003134f'  # 扩展G
            or '\U00031350' <= c <= '\U000323af'  # 扩展H (Unicode 15.1)
            for c in token
        ):
            chinese_tokens.append((token_id, token))
        elif any('a' <= c.lower() <= 'z' for c in token):
            english_tokens.append((token_id, token))
        else:
            other_tokens.append((token_id, token))
    
    # 写入文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TOKEN MAPPING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"总词汇量: {tokenizer.vocab_size}\n\n")
        
        # 特殊标记
        f.write("【特殊标记】\n")
        f.write("-" * 80 + "\n")
        for token_id, token in special_tokens:
            f.write(f"ID {token_id:5d}: {format_token(token)}\n")
        f.write("\n")
        
        # 空格标记
        f.write("【空格相关标记】\n")
        f.write("-" * 80 + "\n")
        for token_id, token in space_tokens:
            f.write(f"ID {token_id:5d}: {format_token(token)}\n")
        f.write("\n")
        
        # 拼音
        f.write(f"【拼音标记】 (共 {len(pinyin_tokens)} 个)\n")
        f.write("-" * 80 + "\n")
        for token_id, token in pinyin_tokens:
            f.write(f"ID {token_id:5d}: {format_token(token)}\n")
        f.write("\n")
        
        # 标点符号
        f.write(f"【标点符号】 (共 {len(punctuation_tokens)} 个)\n")
        f.write("-" * 80 + "\n")
        for token_id, token in punctuation_tokens:
            f.write(f"ID {token_id:5d}: {format_token(token)}\n")
        f.write("\n")
        
        # 中文字符
        f.write(f"【中文字符/词】 (共 {len(chinese_tokens)} 个)\n")
        f.write("-" * 80 + "\n")
        for i, (token_id, token) in enumerate(chinese_tokens):
            f.write(f"ID {token_id:5d}: {format_token(token)}")
            if (i + 1) % 5 == 0:
                f.write("\n")
        if len(chinese_tokens) % 5 != 0:
            f.write("\n")
        f.write("\n")
        
        # 英文字符
        f.write(f"【英文字符/词】 (共 {len(english_tokens)} 个)\n")
        f.write("-" * 80 + "\n")
        for i, (token_id, token) in enumerate(english_tokens):
            f.write(f"ID {token_id:5d}: {format_token(token)}")
            if (i + 1) % 5 == 0:
                f.write("\n")
        if len(english_tokens) % 5 != 0:
            f.write("\n")
        f.write("\n")
        
        # 其他
        if other_tokens:
            f.write(f"【其他标记】 (共 {len(other_tokens)} 个)\n")
            f.write("-" * 80 + "\n")
            for token_id, token in other_tokens:
                f.write(f"ID {token_id:5d}: {format_token(token)}\n")
            f.write("\n")
        
        # 统计信息
        f.write("=" * 80 + "\n")
        f.write("统计信息:\n")
        f.write(f"  特殊标记:       {len(special_tokens)}\n")
        f.write(f"  空格标记:       {len(space_tokens)}\n")
        f.write(f"  拼音标记:       {len(pinyin_tokens)}\n")
        f.write(f"  标点符号:       {len(punctuation_tokens)}\n")
        f.write(f"  中文字符/词:    {len(chinese_tokens)}\n")
        f.write(f"  英文字符/词:    {len(english_tokens)}\n")
        f.write(f"  其他标记:       {len(other_tokens)}\n")
        f.write("=" * 80 + "\n")
    
    print(f"Token映射表已保存到: {output_file}")
    
    # 同时打印一些关键信息到控制台
    print("\n关键token预览:")
    print("-" * 50)
    for token_id in range(min(20, tokenizer.vocab_size)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        print(f"ID {token_id:5d}: {format_token(token)}")
    print("-" * 50)
    print(f"... 共 {tokenizer.vocab_size} 个tokens")
    print("=" * 50)


if __name__ == "__main__":
    # python .\tests\langs\print_bpe_token.py
    text_normalizer = TextNormalizer()
    
    # 添加导出功能
    print("\n" + "="*60)
    print("正在导出token映射表...")
    print("="*60)
    
    # 确保模型已加载
    tokenizer = TextTokenizer(
        vocab_file=r"checkpoints\IndexTTS-2-vLLM\bpe.model",
        # vocab_file=r"checkpoints\IndexTTS-2-vLLM\japanese_bpe.model",
        normalizer=text_normalizer,
    )
    
    # 导出到文件
    export_token_mapping(tokenizer, "outputs/token_mapping_readable.txt")
    # export_token_mapping(tokenizer, "outputs/token_mapping_readable_jp.txt")
    
    # # 如果你想查看特定范围的token，比如拼音
    # print("\n拼音token预览 (ID 8474-10201):")
    # print("-" * 60)
    # for token_id in range(8474, min(10202, tokenizer.vocab_size)):
    #     token = tokenizer.convert_ids_to_tokens(token_id)
    #     print(f"ID {token_id:5d}: '{token}'")