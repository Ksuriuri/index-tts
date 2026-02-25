import sys
import unicodedata

from sentencepiece import sentencepiece_model_pb2 as model_pb2

def generate_es_candidates():
    """生成所有待添加的西班牙文候选字符"""
    candidates = []

    # 1. 西班牙语特殊字符 (Spanish specific characters)
    # 仅保留大写，因为 TextNormalizer 会将文本转为大写
    es_chars = [
        'Á', 'É', 'Í', 'Ó', 'Ú', 'Ü', 'Ñ', '¿', '¡'
    ]
    candidates.extend(es_chars)
    print(f"收集西班牙语特殊字符: {len(es_chars)} 个")

    # 2. 常用西班牙语词根词缀 (Common Spanish prefixes and suffixes)
    # 这些词缀有助于 BPE 模型更好地切分西班牙语单词
    # 全部转为大写
    prefixes = [
        "anti", "auto", "bi", "co", "con", "de", "des", "dis", "en", "ex", 
        "extra", "in", "inter", "mal", "micro", "mono", "multi", "no", "para", 
        "poli", "pos", "post", "pre", "pro", "re", "semi", "sin", "sobre", 
        "sub", "super", "tele", "trans", "tri", "uni", "vice"
    ]
    prefixes = [p.upper() for p in prefixes]

    suffixes = [
        "able", "aceo", "ada", "ador", "al", "ando", "ante", "ar", "ario", "az", 
        "ción", "dad", "dor", "dura", "ear", "edo", "ero", "ez", "eza", "fa", 
        "ficar", "fico", "filo", "fobia", "forme", "gar", "grama", "ia", "ible", 
        "ico", "ido", "iento", "il", "illo", "in", "ino", "ion", "ismo", "ista", 
        "ita", "itis", "itud", "izar", "logia", "ment", "mente", "miento", "ncial", 
        "nd", "oide", "ol", "on", "or", "oso", "sion", "tad", "te", "terapia", 
        "to", "tor", "triz", "tud", "umbre", "ura", "zuela",
        # 新增带特殊字符的后缀
        "sión", "ía", "ería", "eño", "eña", "és", "ín", "ón", "ólogo", "ítico", "güe", "güi"
    ]
    suffixes = [s.upper() for s in suffixes]
    
    # 将词缀加入候选列表
    # 注意：BPE 通常处理子词，这里直接添加这些常见的组合
    # candidates.extend(prefixes)
    candidates.extend(suffixes)
    print(f"收集常用前缀: {len(prefixes)} 个")
    print(f"收集常用后缀: {len(suffixes)} 个")

    # 3. 添加带空格的版本 (Add space-prefixed versions)
    # SentencePiece 使用 \u2581 表示空格。很多词出现在句首或空格后，因此需要带空格的版本。
    # 我们为特殊字符和常用前缀添加带空格的版本。后缀通常在词尾，不需加空格。
    SPACE = "\u2581"
    
    # es_chars_with_space = [f"{SPACE}{c}" for c in es_chars]
    prefixes_with_space = [f"{SPACE}{p}" for p in prefixes]
    
    # candidates.extend(es_chars_with_space)
    candidates.extend(prefixes_with_space)
    # print(f"收集带空格的特殊字符: {len(es_chars_with_space)} 个")
    print(f"收集带空格的前缀: {len(prefixes_with_space)} 个")

    # 4. 数字 0-9
    # 虽然通常基础模型可能包含数字，但为了确保覆盖，这里保留原有的逻辑
    nums = [str(i) for i in range(0, 10)]
    candidates.extend(nums)
    print(f"数字: {len(nums)} 个")
    
    return candidates

def extend_model(input_model_path, output_model_path):
    # 1. 加载现有的 bpe.model
    m = model_pb2.ModelProto()
    try:
        with open(input_model_path, "rb") as f:
            m.ParseFromString(f.read())
    except FileNotFoundError:
        print(f"错误: 找不到输入模型文件: {input_model_path}")
        return
    
    print(f"\n成功加载模型: {input_model_path}")
    print(f"当前词表大小: {len(m.pieces)}")

    # 2. 获取参考属性 (参考 ID 7 的中文单字)
    # 你的分析中 ID 7 是中文单字，概率(Score)表现为 1.0 (在 proto 中可能是 0.0)
    # 我们将新 token 的属性设置得与现有中文 token 完全一致
    if len(m.pieces) > 7:
        ref_piece = m.pieces[7]
        ref_type = ref_piece.type  # 通常是 NORMAL 或 USER_DEFINED
        ref_score = ref_piece.score
        print(f"参考 Token (ID 7): '{ref_piece.piece}'")
        print(f"参考属性 - Type: {ref_type}, Score: {ref_score}")
    else:
        # 如果模型太小，使用默认值
        ref_type = model_pb2.ModelProto.SentencePiece.NORMAL
        ref_score = 0.0
        print("警告: 模型太小，无法获取参考 Token (ID 7)，使用默认属性。")

    # 3. 建立现有词表集合，用于去重
    existing_vocab = set(p.piece for p in m.pieces)

    # 4. 生成候选并过滤
    candidates = generate_es_candidates()
    tokens_to_add = []
    
    for token in candidates:
        if token not in existing_vocab:
            tokens_to_add.append(token)
            # 添加到 set 以免候选列表自身有重复
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
    # 注意：这里假设用户会根据实际情况修改路径，或者保持原有相对路径结构
    extend_model(
        r"checkpoints/IndexTTS-2-vLLM/bpe.model",
        r"checkpoints/IndexTTS-2-vLLM/es_bpe.model"
    )
