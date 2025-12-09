from dataclasses import dataclass
from typing import Dict, Union
import torch
import numpy as np


@dataclass
class ProcessedData:
    # 使用 Union 允许该字段既可以是 Tensor 也可以是 Numpy，避免IDE报错
    text_ids: Union[torch.Tensor, np.ndarray]  # int32
    codes: Union[torch.Tensor, np.ndarray]     # int32
    text_len: int
    code_len: int
    condition: Union[torch.Tensor, np.ndarray] # float32/16
    emo_vec: Union[torch.Tensor, np.ndarray]   # float32/16
    duration: float

    def to_numpy(self):
        """
        将所有 Tensor 字段转换为 Numpy 数组。
        通常在放入 Queue 之前，或者保存 .pkl 文件之前调用。
        """
        # 处理 text_ids
        if isinstance(self.text_ids, torch.Tensor):
            self.text_ids = self.text_ids.detach().cpu().numpy()
        
        # 处理 codes
        if isinstance(self.codes, torch.Tensor):
            self.codes = self.codes.detach().cpu().numpy()
            
        # 处理 condition
        if isinstance(self.condition, torch.Tensor):
            self.condition = self.condition.detach().cpu().numpy()
            
        # 处理 emo_vec
        if isinstance(self.emo_vec, torch.Tensor):
            self.emo_vec = self.emo_vec.detach().cpu().numpy()
            
        return self

    def to_tensor(self, device: str = "cpu"):
        """
        将所有 Numpy 字段转换回 PyTorch Tensor。
        通常在 Dataset 的 __getitem__ 中读取 .pkl 后调用。
        
        Args:
            device (str): 转换后的 Tensor 存放设备，默认为 'cpu'
        """
        # 处理 text_ids (通常用于 Embedding，建议转为 int32 或 int64)
        if isinstance(self.text_ids, np.ndarray):
            self.text_ids = torch.from_numpy(self.text_ids).to(device=device)
            # 如果需要强制类型，可以加 .long() 或 .int()
            # self.text_ids = self.text_ids.long() 

        # 处理 codes
        if isinstance(self.codes, np.ndarray):
            self.codes = torch.from_numpy(self.codes).to(device=device)
            
        # 处理 condition (通常模型需要 float32 或 float16)
        if isinstance(self.condition, np.ndarray):
            self.condition = torch.from_numpy(self.condition).to(device=device)
            
        # 处理 emo_vec
        if isinstance(self.emo_vec, np.ndarray):
            self.emo_vec = torch.from_numpy(self.emo_vec).to(device=device)

        return self


@dataclass
class AudioData:
    """
    meta_info: {
        "src_path": "path/to/source_file",
        "duration": 10,
    }
    """
    processed_data: ProcessedData
    meta_info: Dict[str, str]