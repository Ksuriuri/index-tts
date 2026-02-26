import os
import sys

# 原始路径设置
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from sentencepiece import sentencepiece_model_pb2 as model_pb2

from trainers.bpe.bpe_jp_expand import generate_jp_candidates
from trainers.bpe.bpe_es_expand import generate_es_candidates


def extend_model_jp_es(input_model_path, output_model_path):
    """在同一个模型中先追加日文 token，再追加西班牙文 token。"""
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
    # 如果模型太小，则使用默认值
    if len(m.pieces) > 7:
        ref_piece = m.pieces[7]
        ref_type = ref_piece.type  # 通常是 NORMAL 或 USER_DEFINED
        ref_score = ref_piece.score
        print(f"参考 Token (ID 7): '{ref_piece.piece}'")
        print(f"参考属性 - Type: {ref_type}, Score: {ref_score}")
    else:
        ref_type = model_pb2.ModelProto.SentencePiece.NORMAL
        ref_score = 0.0
        print("警告: 模型太小，无法获取参考 Token (ID 7)，使用默认属性。")

    # 3. 建立现有词表集合，用于去重
    existing_vocab = set(p.piece for p in m.pieces)

    # 4. 生成日文候选并过滤
    jp_candidates = generate_jp_candidates()
    jp_tokens_to_add = []
    for token in jp_candidates:
        if token not in existing_vocab:
            jp_tokens_to_add.append(token)
            existing_vocab.add(token)

    print(f"\n待添加的唯一日文 Token 数: {len(jp_tokens_to_add)}")
    print(f"例如(日文): {jp_tokens_to_add[:10]} ...")

    # 5. 生成西班牙文候选并过滤（在日文之后追加）
    es_candidates = generate_es_candidates()
    es_tokens_to_add = []
    for token in es_candidates:
        if token not in existing_vocab:
            es_tokens_to_add.append(token)
            existing_vocab.add(token)

    print(f"待添加的唯一西班牙文 Token 数: {len(es_tokens_to_add)}")
    print(f"例如(西文): {es_tokens_to_add[:10]} ...")

    # 6. 按顺序追加到模型 pieces 列表末尾
    for token in jp_tokens_to_add:
        new_piece = m.pieces.add()
        new_piece.piece = token
        new_piece.score = ref_score
        new_piece.type = ref_type

    for token in es_tokens_to_add:
        new_piece = m.pieces.add()
        new_piece.piece = token
        new_piece.score = ref_score
        new_piece.type = ref_type

    # 7. 保存新模型
    with open(output_model_path, "wb") as f:
        f.write(m.SerializeToString())

    print(f"\n新模型已保存至: {output_model_path}")
    print(f"新模型词表总大小: {len(m.pieces)}")
    total_new = len(jp_tokens_to_add) + len(es_tokens_to_add)
    print(f"新增 ID 范围: {len(m.pieces) - total_new} - {len(m.pieces) - 1}")
    print(f"其中日文新增 {len(jp_tokens_to_add)} 个，西文新增 {len(es_tokens_to_add)} 个。")


if __name__ == "__main__":
    # 可以通过命令行参数自定义路径，否则使用默认路径
    if len(sys.argv) >= 3:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
    else:
        in_path = r"checkpoints/IndexTTS-2-vLLM/bpe.model"
        out_path = r"checkpoints/IndexTTS-2-vLLM/jp_es_bpe.model"

    extend_model_jp_es(in_path, out_path)

