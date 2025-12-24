from dataclasses import dataclass
from typing import Dict, Union
import torch
import numpy as np


@dataclass
class ProcessedData:
    text_ids: np.ndarray  # int16, [text_len]
    codes: np.ndarray     # int16, [code_len]
    text_len: int
    code_len: int
    condition: np.ndarray # float16, [32, 1280]
    emo_vec: np.ndarray   # float16, [1280]
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

        self.text_ids = self.text_ids.astype(np.int16)
        self.codes = self.codes.astype(np.int16)
        
        self.condition = self.condition.astype(np.float16)
        self.emo_vec = self.emo_vec.astype(np.float16)
            
        return self

    def to_tensor(self, device: str = "cpu"):
        if isinstance(self.text_ids, np.ndarray):
            self.text_ids = torch.from_numpy(self.text_ids).to(device=device, dtype=torch.long)

        if isinstance(self.codes, np.ndarray):
            self.codes = torch.from_numpy(self.codes).to(device=device, dtype=torch.long)
            
        if isinstance(self.condition, np.ndarray):
            self.condition = torch.from_numpy(self.condition).to(device=device, dtype=torch.float32)
            
        if isinstance(self.emo_vec, np.ndarray):
            self.emo_vec = torch.from_numpy(self.emo_vec).to(device=device, dtype=torch.float32)

        return self

