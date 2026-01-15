import os
import io
import random
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
import torch

# --- 配置参数 ---
TEST_PARQUET_PATH = "/mnt/data_3t_1/datasets/preprocess/Japanese-Eroge-Voice/part_0002.parquet" # 请替换为一个真实存在的文件名
SAVE_DIR = "outputs/debug_audio"
NUM_SAMPLES_TO_SAVE = 5
MAX_AUDIO_DURATION = 36

def debug_preprocess_logic():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print(f"Opening: {TEST_PARQUET_PATH}")
    parquet_file = pq.ParquetFile(TEST_PARQUET_PATH)
    
    saved_count = 0
    
    # 只读第一批数据进行测试
    for batch in parquet_file.iter_batches(batch_size=10, columns=['audio', 'whisper_large_v3']):
        audio_col = batch['audio']
        asr_col = batch['whisper_large_v3']

        for i in range(len(batch)):
            if saved_count >= NUM_SAMPLES_TO_SAVE:
                break

            asr_data = asr_col[i].as_py()
            segments = list(asr_data['segments'])
            
            if len(segments) < 2:
                continue

            # --- 核心切分逻辑 (与你原始代码一致) ---
            merged_groups = []
            current_group = [segments[0]]
            for seg_idx in range(1, len(segments)):
                prev_seg = current_group[-1]
                curr_seg = segments[seg_idx]
                silence_gap = curr_seg['start'] - prev_seg['end']
                
                if silence_gap >= 0.5:
                    merged_groups.append(current_group)
                    current_group = [curr_seg]
                else:
                    current_group.append(curr_seg)
            if current_group:
                merged_groups.append(current_group)

            if len(merged_groups) < 2:
                continue

            # 读取完整音频
            full_array, sampling_rate = sf.read(io.BytesIO(audio_col[i]), dtype='float32')
            if full_array.ndim > 1:
                full_array = np.mean(full_array, axis=1)

            # 遍历切分出的组并保存
            for g_idx, group in enumerate(merged_groups):
                if saved_count >= NUM_SAMPLES_TO_SAVE:
                    break
                    
                group_text = "".join([s['text'] for s in group]).strip()
                
                # 随机 Padding 逻辑
                pad_start = random.uniform(0.2, 0.4)
                pad_end = random.uniform(0.3, 0.5)
                
                start_time = max(0.0, group[0]['start'] - pad_start)
                end_time = min(float(full_array.shape[0]) / sampling_rate, group[-1]['end'] + pad_end)
                
                start_sample = int(start_time * sampling_rate)
                end_sample = int(end_time * sampling_rate)
                
                audio_slice = full_array[start_sample:end_sample]
                duration = audio_slice.shape[0] / sampling_rate

                # 保存文件
                file_name = f"sample_{saved_count}_len_{duration:.2f}s.wav"
                save_path = os.path.join(SAVE_DIR, file_name)
                sf.write(save_path, audio_slice, sampling_rate)
                
                print(f"Saved: {file_name}")
                print(f"   Text: {group_text}")
                print(f"   Gap: {pad_start:.2f}s front, {pad_end:.2f}s back")
                
                saved_count += 1

    print(f"\nDone! Please check the '{SAVE_DIR}' folder.")

if __name__ == "__main__":
    debug_preprocess_logic()