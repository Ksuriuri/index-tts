import argparse
import json
import os
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from typing import List, Dict, Any
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)


def count_samples_in_pkl(file_path: Path) -> tuple[Path, int, float]:
    """读取单个pkl文件的长度和修改时间"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            length = len(data)
        mtime = file_path.stat().st_mtime  # 修改时间，用于缓存失效检测
        return file_path, length, mtime
    except Exception as e:
        print(f"[Error] Failed to read {file_path}: {e}")
        return file_path, 0, 0.0


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset index for LazyJapaneseGPTDataset")
    parser.add_argument("--data-dir", default="/mnt/data_3t_2/datasets/indextts_train_data/Galgame-VisualNovel-Reupload", type=str, help="Root directory of .pkl files")
    parser.add_argument("--output", default="/mnt/data_sdd/hhy/index-tts/train_data/train_info_251211.json", type=str, help="Path to save the index JSON file")
    parser.add_argument("--pattern", type=str, default="*-split-pkl-*.pkl", help="Glob pattern to find pkl files")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    # 1. 查找所有pkl文件
    if data_dir.is_file():
        file_paths = [data_dir]
    else:
        file_paths = sorted(list(data_dir.rglob(args.pattern)))
    
    if not file_paths:
        raise FileNotFoundError(f"No .pkl files found in {data_dir} with pattern '{args.pattern}'")

    print(f"[IndexBuilder] Found {len(file_paths)} files. Calculating lengths...")

    # 2. 并行计算文件长度
    file_info = []
    total_samples = 0
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(
            executor.map(count_samples_in_pkl, file_paths),
            total=len(file_paths),
            desc="Processing files"
        ))
    
    # 3. 整理结果
    for file_path, length, mtime in results:
        if length > 0:
            file_info.append({
                "path": str(file_path.absolute()),  # 保存绝对路径
                "length": length,
                "mtime": mtime
            })
            total_samples += length
    
    # 4. 构建索引数据结构
    index_data = {
        "data_dir": str(data_dir.absolute()),
        "total_samples": total_samples,
        "files": file_info,
        "created_at": str(Path().stat().st_mtime),  # 时间戳
    }

    # 5. 保存到JSON文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    print(f"[IndexBuilder] Successfully saved index to {output_path}")
    print(f"[IndexBuilder] Total samples: {total_samples}")
    print(f"[IndexBuilder] Valid files: {len(file_info)} / {len(file_paths)}")


if __name__ == "__main__":
    main()