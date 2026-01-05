import sys
import unicodedata

# 尝试导入 protobuf 定义
# 如果报错，请确保当前目录下有 sentencepiece_model_pb2.py 文件
# try:
# import sentencepiece_model_pb2 as model_pb2
from sentencepiece import sentencepiece_model_pb2 as model_pb2
# except ImportError:
#     print("错误: 找不到 sentencepiece_model_pb2 模块。")
#     print("请下载 sentencepiece_model.proto 并使用 protoc 编译生成该文件，")
#     print("或者直接下载现成的 sentencepiece_model_pb2.py 到当前目录。")
#     sys.exit(1)

def generate_jp_candidates():
    """生成所有待添加的日文候选字符"""
    candidates = []

    # 1. 核心平假名
    hiragana = [chr(i) for i in range(0x3040, 0x30A0) if unicodedata.category(chr(i)) != 'Cn']
    candidates.extend(hiragana)
    print(f"收集平假名: {len(hiragana)} 个")

    # 2. 核心片假名
    katakana = [chr(i) for i in range(0x30A0, 0x3100) if unicodedata.category(chr(i)) != 'Cn']
    candidates.extend(katakana)
    print(f"收集片假名: {len(katakana)} 个")
    
    # 々 \u3005、〜 \u301C、〇 \u3007
    candidates.extend(["\u3005", "\u301C", "\u3007"])

    # 3. JIS 第一水准汉字 (Level 1 Kanji)
    # 范围：16区 - 47区
    jis1_chars = []
    for row in range(16, 48):
        for cell in range(1, 95):
            # 47区到 51 位结束
            if row == 47 and cell > 51:
                break
            b1 = row + 0xA0
            b2 = cell + 0xA0
            try:
                char_unicode = bytes([b1, b2]).decode('euc_jp')
                jis1_chars.append(char_unicode)
            except:
                continue
    candidates.extend(jis1_chars)
    print(f"收集 JIS第一水准汉字: {len(jis1_chars)} 个")

    # 4. JIS 第二水准汉字 (Level 2 Kanji) [新增部分]
    # 范围：48区 - 84区
    jis2_chars = []
    for row in range(48, 85):
        for cell in range(1, 95):
            # 84区到 6 位结束 (最后两个字通常是 熙 后面就没有了)
            if row == 84 and cell > 6:
                break
            
            b1 = row + 0xA0
            b2 = cell + 0xA0
            try:
                char_unicode = bytes([b1, b2]).decode('euc_jp')
                jis2_chars.append(char_unicode)
            except:
                continue
    candidates.extend(jis2_chars)
    print(f"收集 JIS第二水准汉字: {len(jis2_chars)} 个")

    # 数字 0-9，由于中英会被 TextNormalizer 转成对应语言的字词，这里 0-9 为日语专用
    nums = [str(i) for i in range(0, 10)]
    candidates.extend(nums)
    print(f"数字: {len(nums)} 个")
    
    return candidates

def extend_model(input_model_path, output_model_path):
    # 1. 加载现有的 bpe.model
    m = model_pb2.ModelProto()
    with open(input_model_path, "rb") as f:
        m.ParseFromString(f.read())
    
    print(f"\n成功加载模型: {input_model_path}")
    print(f"当前词表大小: {len(m.pieces)}")

    # 2. 获取参考属性 (参考 ID 7 的中文单字)
    # 你的分析中 ID 7 是中文单字，概率(Score)表现为 1.0 (在 proto 中可能是 0.0)
    # 我们将新 token 的属性设置得与现有中文 token 完全一致
    ref_piece = m.pieces[7]
    ref_type = ref_piece.type  # 通常是 NORMAL 或 USER_DEFINED
    ref_score = ref_piece.score
    
    print(f"参考 Token (ID 7): '{ref_piece.piece}'")
    print(f"参考属性 - Type: {ref_type}, Score: {ref_score}")

    # 3. 建立现有词表集合，用于去重
    # 重要：日文汉字与中文汉字由大量重叠 (UniHan)，如果不去重，会出现同一个字有两个 ID 的情况
    existing_vocab = set(p.piece for p in m.pieces)

    # 4. 生成候选并过滤
    candidates = generate_jp_candidates()
    tokens_to_add = []
    
    for token in candidates:
        if token not in existing_vocab:
            tokens_to_add.append(token)
            # 添加到 set 以免候选列表自身有重复 (虽然逻辑上不应该有)
            existing_vocab.add(token)
    
    print(f"\n待添加的唯一 Token 数: {len(tokens_to_add)}")
    print(f"例如: {tokens_to_add[:10]} ...")

    # 5. 追加到模型 pieces 列表末尾
    for token in tokens_to_add:
        new_piece = m.pieces.add()
        new_piece.piece = token
        new_piece.score = ref_score
        new_piece.type = ref_type
    
    # 6. 保存新模型
    with open(output_model_path, "wb") as f:
        f.write(m.SerializeToString())
    
    print(f"\n新模型已保存至: {output_model_path}")
    print(f"新模型词表总大小: {len(m.pieces)}")
    print(f"新增 ID 范围: {len(m.pieces) - len(tokens_to_add)} - {len(m.pieces) - 1}")

if __name__ == "__main__":
    # 确保当前目录下有 bpe.model
    extend_model(
        r"checkpoints/IndexTTS-2-vLLM/bpe.model",
        r"checkpoints/IndexTTS-2-vLLM/jp_bpe.model"
    )