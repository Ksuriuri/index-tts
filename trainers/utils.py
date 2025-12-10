from dataclasses import dataclass
from typing import Dict, Union
import torch
import numpy as np


@dataclass
class ProcessedData:
    # 使用 Union 允许该字段既可以是 Tensor 也可以是 Numpy，避免IDE报错
    text_ids: Union[torch.Tensor, np.ndarray]  # int16
    codes: Union[torch.Tensor, np.ndarray]     # int16
    text_len: int
    code_len: int
    condition: Union[torch.Tensor, np.ndarray] # float16
    emo_vec: Union[torch.Tensor, np.ndarray]   # float16
    duration: float

    def to_numpy(self):
        if isinstance(self.text_ids, torch.Tensor):
            self.text_ids = self.text_ids.detach().cpu().numpy()
        
        if isinstance(self.codes, torch.Tensor):
            self.codes = self.codes.detach().cpu().numpy()
            
        if isinstance(self.condition, torch.Tensor):
            self.condition = self.condition.detach().cpu().numpy()
            
        if isinstance(self.emo_vec, torch.Tensor):
            self.emo_vec = self.emo_vec.detach().cpu().numpy()
            
        return self

    def to_tensor(self, device: str = "cpu"):
        if isinstance(self.text_ids, np.ndarray):
            self.text_ids = torch.from_numpy(self.text_ids).to(device=device, dtype=torch.int32)

        if isinstance(self.codes, np.ndarray):
            self.codes = torch.from_numpy(self.codes).to(device=device, dtype=torch.int32)
            
        if isinstance(self.condition, np.ndarray):
            self.condition = torch.from_numpy(self.condition).to(device=device, dtype=torch.float32)
            
        if isinstance(self.emo_vec, np.ndarray):
            self.emo_vec = torch.from_numpy(self.emo_vec).to(device=device, dtype=torch.float32)

        return self

