import os
from huggingface_hub import snapshot_download

LANG = "Spanish"
DATASET_NAME = "ylacombe/google-chilean-spanish"

repo_id = f"{DATASET_NAME}"
repo_type = "dataset"
# 目标文件夹路径
local_dir = f"/mnt/data_3t_1/datasets/raw_data/{LANG}/{DATASET_NAME.split('/')[-1]}"
# 只下载指定目录下的所有文件
# allow_patterns = "es/*"
allow_patterns = None

print(f"开始下载 {repo_id} 的 {allow_patterns} 到 {local_dir} ...")

try:
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        # local_dir_use_symlinks=False 意味着下载实际文件而不是缓存的软链接
        # 这对于数据存储盘通常更好
        local_dir_use_symlinks=False, 
        resume_download=True,
        max_workers=8 # 根据你的网络情况调整并发数
    )
    print("下载完成！")
except Exception as e:
    print(f"下载出错: {e}")