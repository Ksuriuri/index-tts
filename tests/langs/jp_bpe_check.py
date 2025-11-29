import os
import sys
import datetime

# 尝试添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from collections import Counter
from tqdm import tqdm
from modelscope.msdatasets import MsDataset
# 假设 indextts 已经在你的环境中安装或位于路径中
try:
    from indextts.utils.front import TextNormalizer, TextTokenizer
except ImportError:
    print("提示: 未找到 indextts 库，请确保代码运行在正确的环境中。")
    # 为了防止直接报错退出，这里只做提示，实际运行时如果缺库会报错
    pass

# --- 配置部分 ---
MODELSCOPE_CACHE_DIR = r'./outputs'
MODEL_FILE = r"D:\workspace\index-tts\checkpoints\IndexTTS-2-vLLM\jp_bpe.model"  # 请修改为你的实际路径
UNK_ID = 2
TEST_SAMPLE_COUNT = 10000  # 测试多少条数据
DATASET_ID = 'wikimedia/wikipedia'
SUBSET_NAME = '20231101.ja' 

# 结果保存路径
RESULT_SAVE_PATH = r'./outputs/test_report.txt'
UNK_DETAILS_PATH = r'./outputs/unk_tokens_details.csv'

def run_modelscope_test():
    # --- 1. 加载你的 BPE 模型 ---
    print(f">>> 正在加载 BPE 模型: {MODEL_FILE}")
    if not os.path.exists(MODEL_FILE):
        print(f"错误: 找不到模型文件 {MODEL_FILE}")
        return

    try:
        normalizer = TextNormalizer()
        normalizer.load()
        tokenizer = TextTokenizer(MODEL_FILE, normalizer)
    except Exception as e:
        print(f"模型加载出错: {e}")
        return

    # --- 2. 连接 ModelScope 加载数据 (流式) ---
    print(f">>> 正在连接 ModelScope (国内源)...")
    print(f">>> 缓存目录已设置为: {MODELSCOPE_CACHE_DIR}")
    
    try:
        ds = MsDataset.load(
            DATASET_ID, 
            subset_name=SUBSET_NAME, 
            split='train', 
            use_streaming=True,
            cache_dir=MODELSCOPE_CACHE_DIR
        )
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return

    print(f">>> 开始测试前 {TEST_SAMPLE_COUNT} 条样本...")

    # --- 3. 循环测试 ---
    total_tokens = 0
    total_unks = 0
    unk_counter = Counter()
    processed_count = 0
    
    ds_iter = iter(ds)
    progress_bar = tqdm(total=TEST_SAMPLE_COUNT, desc="Processing", unit="samples")

    while processed_count < TEST_SAMPLE_COUNT:
        try:
            item = next(ds_iter)
        except StopIteration:
            break

        text = item.get('text', '')
        if not text:
            continue

        # 截断过长的文本以提高速度
        text = text[:500]

        try:
            # Tokenize
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            
            # 统计
            batch_unks = ids.count(UNK_ID)
            total_tokens += len(ids)
            total_unks += batch_unks
            
            # 记录具体的 UNK
            if batch_unks > 0:
                for t_str, t_id in zip(tokens, ids):
                    if t_id == UNK_ID:
                        unk_counter[t_str] += 1
            
            processed_count += 1
            progress_bar.update(1)
            
        except Exception as e:
            pass 

    progress_bar.close()

    # --- 4. 结果生成与保存 ---
    if total_tokens == 0:
        print("未处理任何数据。")
        return

    coverage = (1 - (total_unks / total_tokens)) * 100
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 构建报告内容字符串
    lines = []
    lines.append("="*60)
    lines.append(f"ModelScope 测试报告")
    lines.append(f"测试时间:   {current_time}")
    lines.append("="*60)
    lines.append(f"模型路径:   {MODEL_FILE}")
    lines.append(f"数据集:     {DATASET_ID} ({SUBSET_NAME})")
    lines.append(f"样本数量:   {processed_count}")
    lines.append("-" * 60)
    lines.append(f"总 Token:   {total_tokens}")
    lines.append(f"总 UNK:     {total_unks}")
    lines.append(f"覆盖率:     {coverage:.4f}%")
    lines.append("="*60)

    lines.append("\n>>> ☠️  导致 UNK 最多的前 50 个 Token:")
    if not unk_counter:
        lines.append("无 (完美！)")
    else:
        for token, count in unk_counter.most_common(50):
            # 为了防止特殊字符破坏排版，使用 repr
            lines.append(f"  {repr(token):<20} : {count} 次")

    lines.append("\n>>> 🚀 结论:")
    if coverage > 99.0:
        lines.append("✅ 强：这个模型日语能力很棒。")
    elif coverage > 95.0:
        lines.append("🆗 中：大部分能读，但可能有一些特定汉字不行。")
    else:
        lines.append("❌ 弱：这可能不是一个合格的日语模型，或者词表主要针对中文。")

    report_content = "\n".join(lines)

    # 1. 打印到控制台
    print("\n" + report_content)

    # 2. 保存报告到 TXT
    try:
        with open(RESULT_SAVE_PATH, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\n[OK] 测试报告已保存至: {os.path.abspath(RESULT_SAVE_PATH)}")
    except Exception as e:
        print(f"\n[Error] 保存报告失败: {e}")

    # 3. (可选) 保存详细的 UNK 列表到 CSV，方便后续分析
    if unk_counter:
        try:
            with open(UNK_DETAILS_PATH, 'w', encoding='utf-8-sig') as f:
                f.write("Token,Count\n")
                for token, count in unk_counter.most_common():
                    # 处理 CSV 中的逗号和换行
                    clean_token = token.replace('"', '""')
                    f.write(f'"{clean_token}",{count}\n')
            print(f"[OK] 详细 UNK 列表已保存至: {os.path.abspath(UNK_DETAILS_PATH)}")
        except Exception as e:
            print(f"[Error] 保存 UNK 详情失败: {e}")

if __name__ == "__main__":
    run_modelscope_test()