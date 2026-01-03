import os
import glob
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from loguru import logger
import math

# --- 配置 ---
# DATASET_NAME = "Galgame-VisualNovel-Reupload"
# DATASET_NAME = "Gacha_games_jp"
# DATASET_NAME = "Emilia_JA"
DATASET_NAME = "Emilia-YODAS_JA"
# DATASET_NAME = "Japanese-Eroge-Voice"
DATASET_DIR = f"/mnt/data_3t_1/datasets/preprocess/{DATASET_NAME}"
BATCH_SIZE = 1024

def simplify_parquet_audio(file_path):
    temp_path = file_path + ".tmp"
    
    try:
        # 1. 使用 ParquetFile 读取元数据
        parquet_file = pq.ParquetFile(file_path)
        original_schema = parquet_file.schema_arrow
        
        # 检查 audio 列是否存在及其类型
        try:
            audio_field_idx = original_schema.get_field_index('audio')
            audio_field = original_schema.field(audio_field_idx)
        except ValueError:
            logger.warning(f"Column 'audio' not found in {file_path}. Skipping.")
            return "skipped"

        # 判断是否已经是 binary 类型
        if pa.types.is_binary(audio_field.type):
            return "already_simplified"

        # 2. 构建新 Schema
        new_fields = []
        for i, field in enumerate(original_schema):
            if i == audio_field_idx:
                # 将原来的类型（通常是 Struct）改为 Binary
                new_fields.append(pa.field('audio', pa.binary()))
            else:
                new_fields.append(field)
        
        new_schema = pa.schema(new_fields)
        
        # 3. 计算总批次用于内部进度条
        total_rows = parquet_file.metadata.num_rows
        total_batches = math.ceil(total_rows / BATCH_SIZE)
        
        # 4. 创建写入器并处理
        writer = pq.ParquetWriter(temp_path, schema=new_schema, compression='snappy')
        
        # 使用内部进度条 (leave=False 保证子进度条完成后自动消失)
        with tqdm(total=total_batches, desc=f"  Processing batches", leave=False) as pbar:
            for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
                new_columns = []
                for i, col in enumerate(batch.columns):
                    if i == audio_field_idx:
                        # 从 StructArray 中提取名为 'bytes' 的子列
                        # 如果你的列结构中字段名不是 'bytes'，请修改此处
                        simplified_audio = col.field('bytes')
                        new_columns.append(simplified_audio)
                    else:
                        new_columns.append(col)
                
                new_batch = pa.RecordBatch.from_arrays(new_columns, schema=new_schema)
                writer.write_batch(new_batch)
                pbar.update(1)
            
        writer.close()
        
        # 5. 替换原文件
        os.replace(temp_path, file_path)
        return "success"

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return "error"

def main():
    # 获取目录下所有的 parquet 文件
    parquet_files = sorted(glob.glob(os.path.join(DATASET_DIR, "part_*.parquet")))
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {DATASET_DIR}")
        return

    logger.info(f"Found {len(parquet_files)} files. Starting simplification...")

    # 外部总进度条
    for f in tqdm(parquet_files, desc="Overall Progress"):
        status = simplify_parquet_audio(f)
        
        file_name = os.path.basename(f)
        if status == "already_simplified":
            logger.info(f"Skipped: {file_name} (Already simplified)")
        elif status == "success":
            logger.success(f"Processed: {file_name}")
        elif status == "error":
            logger.error(f"Failed: {file_name}")

    logger.info("Task completed.")

if __name__ == "__main__":
    main()