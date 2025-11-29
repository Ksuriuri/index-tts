import os
import sys
import math  # 新增：用于计算概率

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from indextts.utils.front import TextNormalizer, TextTokenizer
import unicodedata
import re

def export_token_mapping(tokenizer: TextTokenizer, output_file="token_mapping_with_probs.txt"):
    """
    导出所有token的映射表，包含概率信息（如果可用）
    """
    import html
    
    def format_token(token, score=None):
        """格式化token，使其更易读"""
        # 处理空格
        if token == "▁":
            display = "'▁'(空格)"
        elif token.startswith("▁"):
            display = f"'{token}' (前缀有空格)"
        # 处理特殊字符
        elif token in tokenizer.special_tokens_map.values():
            display = f"'{token}' [特殊标记]"
        # 处理控制字符
        elif any(ord(c) < 32 for c in token):
            escaped = "".join(f"\\x{ord(c):02x}" if ord(c) < 32 else c for c in token)
            display = f"'{escaped}' (控制字符)"
        # 处理标点符号
        elif any(unicodedata.category(c).startswith("P") for c in token):
            display = f"'{token}' [标点]"
        # 处理数字
        elif token.isdigit():
            display = f"'{token}' [数字]"
        # 处理拼音
        elif re.match(r"[a-zA-Z]+[1-5]", token):
            display = f"'{token}' [拼音]"
        else:
            display = f"'{token}'"
        
        # 添加概率信息（如果可用）
        if score is not None:
            display += f" | p={score:.6f}"
        
        return display
    
    # 按类别分组（现在同时存储score）
    special_tokens = []
    punctuation_tokens = []
    pinyin_tokens = []
    chinese_tokens = []
    english_tokens = []
    other_tokens = []
    space_tokens = []
    
    print(f"正在生成token映射表，词汇表大小: {tokenizer.vocab_size}...")
    
    # 检查tokenizer是否支持get_score方法
    has_score = hasattr(tokenizer.sp_model, 'GetScore')
    
    if not has_score:
        print("⚠️ 警告：当前tokenizer不支持提取token概率，将只输出ID和文本映射")
        print("这通常是因为：")
        print("  1. 使用的是标准HuggingFace tokenizer（不包含概率信息）")
        print("  2. TextTokenizer未封装get_score方法")
    
    for token_id in range(tokenizer.vocab_size):
        token = tokenizer.convert_ids_to_tokens(token_id)
        
        # 尝试获取token的score（对数概率）
        score = None
        if hasattr(tokenizer.sp_model, 'GetScore'):
            log_prob = tokenizer.sp_model.GetScore(token_id)
            score = math.exp(log_prob)  # 转换为实际概率
        
        # 分类（现在包含score）
        token_info = (token_id, token, score)
        
        if token in tokenizer.special_tokens_map.values():
            special_tokens.append(token_info)
        elif token == "▁" or token.startswith("▁"):
            space_tokens.append(token_info)
        elif re.match(r"[a-zA-Z]+[1-5]", token):
            pinyin_tokens.append(token_info)
        elif any(unicodedata.category(c).startswith("P") for c in token if c != "▁"):
            punctuation_tokens.append(token_info)
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
            chinese_tokens.append(token_info)
        elif any('a' <= c.lower() <= 'z' for c in token):
            english_tokens.append(token_info)
        else:
            other_tokens.append(token_info)
    
    # 写入文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"TOKEN MAPPING REPORT{' WITH PROBABILITIES' if has_score else ''}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"总词汇量: {tokenizer.vocab_size}\n")
        if has_score:
            f.write("概率信息: 已包含（logp=对数概率, p=转换后概率）\n")
        else:
            f.write("概率信息: 暂不可用（tokenizer不支持）\n")
        f.write("\n")
        
        # 所有类别输出时现在都会包含概率信息
        categories = [
            ("【特殊标记】", special_tokens),
            ("【空格相关标记】", space_tokens),
            ("【拼音标记】", pinyin_tokens),
            ("【标点符号】", punctuation_tokens),
            ("【中文字符/词】", chinese_tokens),
            ("【英文字符/词】", english_tokens),
        ]

        items_per_line = 1
        
        sum_probs = 0
        for cat_name, tokens in categories:
            if tokens:
                f.write(f"{cat_name} (共 {len(tokens)} 个)\n")
                f.write("-" * 80 + "\n")
                for i, (token_id, token, score) in enumerate(tokens):
                    if score is not None and score < 1.0:
                        sum_probs += score
                    f.write(f"ID {token_id:5d}: {format_token(token, score)}")
                    if (i + 1) % items_per_line == 0:  # 调整每行列数
                        f.write("\n")
                if len(tokens) % items_per_line != 0:
                    f.write("\n")
                f.write("\n")
        print(f"总概率和: {sum_probs:.6f}")
        
        if other_tokens:
            f.write(f"【其他标记】 (共 {len(other_tokens)} 个)\n")
            f.write("-" * 80 + "\n")
            for i, (token_id, token, score) in enumerate(other_tokens):
                f.write(f"ID {token_id:5d}: {format_token(token, score)}")
                if (i + 1) % items_per_line == 0:
                    f.write("\n")
            if len(other_tokens) % items_per_line != 0:
                f.write("\n")
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
        
        if has_score:
            # 计算一些概率统计
            all_scores = [s for _, _, s in 
                         (special_tokens + space_tokens + pinyin_tokens + 
                          punctuation_tokens + chinese_tokens + english_tokens + other_tokens) 
                         if s is not None]
            if all_scores:
                avg_logp = sum(all_scores) / len(all_scores)
                f.write(f"\n概率统计:\n")
                f.write(f"  平均对数概率:   {avg_logp:.6f}\n")
                f.write(f"  最小/最大logp:  {min(all_scores):.6f} / {max(all_scores):.6f}\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\n✅ Token映射表已保存到: {output_file}")
    
    # # 控制台预览（包含概率）
    # print("\n关键token预览:")
    # print("-" * 70)
    # for token_id in range(min(20, tokenizer.vocab_size)):
    #     token = tokenizer.convert_ids_to_tokens(token_id)
    #     score = None
    #     if hasattr(tokenizer, 'get_score'):
    #         score = tokenizer.get_score(token_id)
    #     elif hasattr(tokenizer, 'sp') and hasattr(tokenizer.sp, 'get_score'):
    #         score = tokenizer.sp.get_score(token_id)
    #     print(f"ID {token_id:5d}: {format_token(token, score)}")
    # print("-" * 70)


if __name__ == "__main__":
    text_normalizer = TextNormalizer()
    
    print("\n" + "="*60)
    print("正在导出token映射表（含概率信息）...")
    print("="*60)
    
    tokenizer = TextTokenizer(
        # vocab_file=r"checkpoints\IndexTTS-2-vLLM\bpe.model",
        # vocab_file=r"checkpoints\IndexTTS-2-vLLM\jp_bpe.model",
        vocab_file=r"checkpoints\IndexTTS-2-vLLM\japanese_bpe.model",
        normalizer=text_normalizer,
    )
    
    # export_token_mapping(tokenizer, "outputs/token_mapping_with_probs.txt")
    # export_token_mapping(tokenizer, "outputs/token_mapping_with_probs_jp.txt")
    export_token_mapping(tokenizer, "outputs/token_mapping_with_probs_jp_git.txt")
    
    # # 如果你想查看特定范围的token
    # print("\n拼音token预览 (ID 8474-10201):")
    # print("-" * 70)
    # for token_id in range(8474, min(10202, tokenizer.vocab_size)):
    #     token = tokenizer.convert_ids_to_tokens(token_id)
    #     score = None
    #     if hasattr(tokenizer, 'get_score'):
    #         score = tokenizer.get_score(token_id)
    #     print(f"ID {token_id:5d}: {format_token(token, score)}")