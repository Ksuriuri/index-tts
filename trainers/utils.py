from dataclasses import dataclass
from typing import Dict
import torch


@dataclass
class ProcessedData:
    text_ids: torch.Tensor  # torch.int32, [text_len]
    codes: torch.Tensor  # torch.int32, [code_len]
    text_len: int
    code_len: int
    condition: torch.Tensor  # torch.float32, [32, dim]
    emo_vec: torch.Tensor  # torch.float32, [1, dim]
    duration: float


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