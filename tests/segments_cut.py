import os
import io
import random
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
from tqdm import tqdm

# ================= 配置区域 =================
# TODO: 请替换为你想要测试的 Parquet 文件路径
PARQUET_PATH = "/mnt/data_3t_1/datasets/preprocess/Japanese-Eroge-Voice/part_0002.parquet"
OUTPUT_DIR = "./outputs/seg_audio"      # 音频保存路径
MAX_SAMPLES_TO_SAVE = 20           # 保存多少个切片后停止
MAX_AUDIO_DURATION = 36            # 原始代码中的限制
MIN_SILLENCE_DURATION = 0.5
# ===========================================

def main():
    if not os.path.exists(PARQUET_PATH):
        print(f"错误: 找不到文件 {PARQUET_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42) # 固定种子方便复现

    print(f"正在读取: {PARQUET_PATH}")
    parquet_file = pq.ParquetFile(PARQUET_PATH)
    
    saved_count = 0
    
    # 只需要读取 audio 和 whisper_large_v3 列
    iterator = parquet_file.iter_batches(batch_size=16, columns=['audio', 'whisper_large_v3'])

    for batch in iterator:
        if saved_count >= MAX_SAMPLES_TO_SAVE:
            break

        audio_col = batch['audio']
        asr_col = batch['whisper_large_v3']
        
        for i in range(len(batch)):
            if saved_count >= MAX_SAMPLES_TO_SAVE:
                break

            try:
                # 1. 获取 ASR 数据
                asr_data = asr_col[i].as_py()
                segments = list(asr_data.get('segments', []))
                
                if not segments:
                    continue

                # 2. 核心逻辑：合并 Group (复用你的逻辑)
                merged_groups = []
                current_group = [segments[0]]
                
                for seg_idx in range(1, len(segments)):
                    prev_seg = current_group[-1]
                    curr_seg = segments[seg_idx]
                    
                    # 检查间隔：当前开始时间 - 上一段结束时间
                    silence_gap = curr_seg['start'] - prev_seg['end']
                    
                    if silence_gap >= MIN_SILLENCE_DURATION:
                        merged_groups.append(current_group)
                        current_group = [curr_seg]
                    else:
                        current_group.append(curr_seg)
                
                if current_group:
                    merged_groups.append(current_group)

                if len(merged_groups) < 2:
                    continue

                # 3. 读取音频 (只有在确定要处理时才解码，节省时间)
                array, sampling_rate = sf.read(io.BytesIO(audio_col[i]), dtype='float32')
                
                # 双声道转单声道
                if array.ndim > 1:
                    array = np.mean(array, axis=1)

                total_duration = array.shape[0] / sampling_rate

                # 4. 遍历切分后的组并保存
                for g_idx, group in enumerate(merged_groups):
                    if saved_count >= MAX_SAMPLES_TO_SAVE:
                        break

                    group_start_time = group[0]['start']
                    group_end_time = group[-1]['end']
                    group_text = "".join([s['text'] for s in group]).strip()

                    # 计算采样点索引 (复用你的 Padding 逻辑)
                    pad_seconds_start = random.uniform(0.2, 0.4)
                    start_time_padded = max(0.0, group_start_time - pad_seconds_start)
                    
                    pad_seconds_end = random.uniform(0.3, 0.5)
                    end_time_padded = min(total_duration, group_end_time + pad_seconds_end)

                    start_sample = int(start_time_padded * sampling_rate)
                    end_sample = int(end_time_padded * sampling_rate)

                    # 提取音频切片
                    audio_slice = array[start_sample:end_sample]
                    
                    slice_duration = audio_slice.shape[0] / sampling_rate
                    
                    # 简单过滤
                    if slice_duration <= 0 or slice_duration > MAX_AUDIO_DURATION:
                        continue

                    # 5. 保存到本地
                    file_name = f"sample_{saved_count:03d}_{g_idx}.wav"
                    save_path = os.path.join(OUTPUT_DIR, file_name)
                    
                    sf.write(save_path, audio_slice, sampling_rate)
                    
                    # 同时保存文本方便对照
                    with open(save_path.replace(".wav", ".txt"), "w", encoding="utf-8") as f:
                        f.write(f"Text: {group_text}\n")
                        f.write(f"Original Time: {group_start_time:.2f} - {group_end_time:.2f}\n")
                        f.write(f"Padded Time: {start_time_padded:.2f} - {end_time_padded:.2f}\n")
                        f.write(f"Source Index: {i}, Group: {g_idx}\n")

                    print(f"已保存: {file_name} | 文本: {group_text[:30]}...")
                saved_count += 1

            except Exception as e:
                print(f"处理出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n完成！共保存 {saved_count} 个音频片段到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()