# from huggingface_hub import snapshot_download
# snapshot_download(
#     repo_id='joujiboi/Galgame-VisualNovel-Reupload',
#     repo_type='dataset',
#     local_dir='/mnt/data_3t_1/datasets/raw_data/Galgame-VisualNovel-Reupload',
#     max_workers=4
# )





import os
from huggingface_hub import login, snapshot_download

login(token="AccessToken")

REPO_ID = "amphion/Emilia-Dataset"
TARGET_DIR = "Emilia-YODAS/JA"
# TARGET_DIR = "Emilia/JA"
ALLOW_PATTERN = f"{TARGET_DIR}/*"
LOCAL_DIR = "/mnt/data_3t_1/datasets/raw_data"

try:
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        allow_patterns=ALLOW_PATTERN,
        # resume_download=True,  # 不需要显式设置，默认打开
        max_workers=8,
    )
    print("下载完成！")
except Exception as e:
    print(f"下载过程中出错: {e}")

# 4. 验证结构
target_path = os.path.join(LOCAL_DIR, TARGET_DIR)
if os.path.exists(target_path):
    print(f"文件已成功保存在: {target_path}")
else:
    print("警告：目标文件夹未生成，请检查下载日志。")