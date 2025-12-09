from dataclasses import dataclass
import os
import sys
import time
import traceback
from typing import List
from loguru import logger

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from omegaconf import OmegaConf
import torch
from torch.multiprocessing import Process, Queue

from transformers import SeamlessM4TFeatureExtractor
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
import safetensors
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer
from trainers.utils import ProcessedData

@dataclass
class DataPreprocessorReqData:
    """
    audio: torch.Tensor, [audio_len], sampling_rate=16000, torch.float32, [-1, 1]  # , share_memory
    """
    text: str
    audio: torch.Tensor
    file_rel_path: str  # 用于追踪数据属于哪个文件


class DataPreprocessor(Process):
    def __init__(
        self,
        model_dir: str,
        input_queue: Queue,
        output_queue: Queue,
        worker_id: int = 0,
        gpu_id: int = 0,
        daemon = True
    ):
        super().__init__(daemon=daemon)
        self.model_dir = model_dir
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.input_queue = input_queue
        self.output_queue = output_queue

    def init_models(self):
        cfg_path = os.path.join(self.model_dir, "config.yaml")
        cfg = OmegaConf.load(cfg_path)

        self.device = torch.device(f"cuda")  # :{self.gpu_id}
        self.dtype = torch.float32

        bpe_path = os.path.join(self.model_dir, "jp_bpe.model")
        normalizer = TextNormalizer()
        normalizer.load()
        self.tokenizer = TextTokenizer(bpe_path, normalizer)

        self.gpt = UnifiedVoice(**cfg.gpt)
        gpt_path = os.path.join(self.model_dir, cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, gpt_path)
        self.gpt = self.gpt.to(self.device)
        self.gpt.eval() # 确保进入eval模式
        logger.info(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] gpt initializing...')

        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            os.path.join(self.model_dir, "w2v-bert-2.0")
        )

        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(self.model_dir, cfg.w2v_stat),
            os.path.join(self.model_dir, "w2v-bert-2.0")
        )
        self.semantic_model = self.semantic_model.to(device=self.device, dtype=self.dtype)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(device=self.device, dtype=self.dtype)
        self.semantic_std = self.semantic_std.to(device=self.device, dtype=self.dtype)
        logger.info(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] semantic_model initializing...')

        self.semantic_codec = build_semantic_codec(cfg.semantic_codec)
        semantic_code_ckpt = os.path.join(self.model_dir, "semantic_codec/model.safetensors")
        safetensors.torch.load_model(self.semantic_codec, semantic_code_ckpt)
        self.semantic_codec = self.semantic_codec.to(device=self.device, dtype=self.dtype)
        self.semantic_codec.eval()
        logger.info(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] semantic_codec initializing...')

        # 关闭所有梯度
        for param in self.gpt.parameters():
            param.requires_grad = False
        for param in self.semantic_model.parameters():
            param.requires_grad = False
        for param in self.semantic_codec.parameters():
            param.requires_grad = False

    def healthy_check(self):
        try:
            fake_audio = torch.zeros(16000, dtype=torch.float32)
            fake_text = "123"
            self.preprocess(fake_text, fake_audio)
        except Exception as e:
            logger.error(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] healthy check error: {e}')
            return False
        return True

    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        init_flag = False
        for _ in range(3):
            try:
                self.init_models()
                if self.healthy_check():
                    logger.info(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] init models success')
                    init_flag = True
                    break
            except Exception as e:
                logger.error(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] init models error: {e}')
        while init_flag:
            input_data: List[DataPreprocessorReqData] = self.input_queue.get()
            if input_data is None:
                break

            processed_datas_req = []
            try:
                texts, audios = [], []
                for input_data_ in input_data:
                    # processed_data = self.preprocess(input_data_.text, input_data_.audio)
                    # processed_datas_req.append((input_data_.file_rel_path, processed_data))
                    texts.append(input_data_.text)
                    audios.append(input_data_.audio)
                processed_datas = self.preprocess_batch(texts, audios)
                for i, processed_data in enumerate(processed_datas):
                    processed_datas_req.append((input_data[i].file_rel_path, processed_data))
                self.output_queue.put(processed_datas_req)
            except Exception as e:
                logger.error(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] preprocess error: {traceback.format_exc()}')
                if self.healthy_check():
                    continue
                else:
                    break

    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    @torch.no_grad()
    def preprocess(
        self,
        text: str,
        audio: torch.Tensor,
    ):
        """
        audio: torch.Tensor, [audio_len], sampling_rate=16000, torch.float32, [-1, 1]
        """  
        duration = audio.shape[0] / 16000
        # Tokenize Text
        text_tokens_list = self.tokenizer.tokenize(text)
        text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        
        if len(text_ids) == 0:
            raise ValueError("text_ids is empty")
            
        text_ids = torch.tensor(text_ids, dtype=torch.int16)  # [text_len]
        text_len = len(text_ids)

        # Extract Features
        # stt = time.time()
        inputs = self.extract_features(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        # logger.error(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] extract_features time: {time.time() - stt:.2f}s')
        
        # Get Speaker Condition Embedding
        # stt = time.time()
        spk_cond_emb = self.get_emb(input_features, attention_mask)
        # logger.error(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] get_emb time: {time.time() - stt:.2f}s')

        # Quantize / Codec
        # stt = time.time()
        cond_lengths = attention_mask.sum(dim=1).long()
        semantic_code, _ = self.semantic_codec.quantize(spk_cond_emb)  # [1, code_len]
        semantic_code = semantic_code.squeeze(0)  # [code_len]
        code_len = semantic_code.shape[0]
        # logger.error(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] quantize time: {time.time() - stt:.2f}s')

        # Get Conditioning & Emotion Vector
        # stt = time.time()
        feat_t = spk_cond_emb.transpose(1, 2)
        cond_lengths_device = cond_lengths.to(spk_cond_emb.device)
        
        conditioning = self.gpt.get_conditioning(feat_t, cond_lengths_device).squeeze(0)  # [32, 1280]
        emo_vec = self.gpt.get_emovec(spk_cond_emb, cond_lengths_device).squeeze(0)  # [1280]
        # logger.error(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] get_conditioning & get_emovec time: {time.time() - stt:.2f}s')

        # Create Object
        processed_data = ProcessedData(
            text_ids=text_ids.to(device="cpu", dtype=torch.int16),
            codes=semantic_code.to(device="cpu", dtype=torch.int16),
            text_len=text_len,
            code_len=code_len,
            condition=conditioning.to(device="cpu", dtype=torch.float16),
            emo_vec=emo_vec.to(device="cpu", dtype=torch.float16),
            duration=duration,
        )
        return processed_data

    @torch.no_grad()
    def preprocess_batch(
        self,
        texts: List[str],
        audios: List[torch.Tensor],
    ):
        """
        audio: torch.Tensor, [audio_len], sampling_rate=16000, torch.float32, [-1, 1]
        """
        batch_size = len(texts)
        durations = [audio.shape[0] / 16000 for audio in audios]

        # Tokenize Text
        text_ids_list = []
        text_lens = []
        for text in texts:
            text_tokens_list = self.tokenizer.tokenize(text)
            text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        
            # if len(text_ids) == 0:
            #     raise ValueError("text_ids is empty")
            
            text_ids = torch.tensor(text_ids, dtype=torch.int16)  # [text_len]
            text_len = len(text_ids)

            text_ids_list.append(text_ids)
            text_lens.append(text_len)

        # Extract Features
        # stt = time.time()
        inputs = self.extract_features(audios, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        # logger.error(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] extract_features time: {time.time() - stt:.2f}s')
        
        # Get Speaker Condition Embedding
        # stt = time.time()
        spk_cond_emb = self.get_emb(input_features, attention_mask)
        # logger.error(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] get_emb time: {time.time() - stt:.2f}s')

        # Quantize / Codec
        # stt = time.time()
        cond_lengths = attention_mask.sum(dim=1).long()
        semantic_code, _ = self.semantic_codec.quantize(spk_cond_emb)  # [b, code_len]

        semantic_codes = []
        code_lens = []
        for b in range(batch_size):
            semantic_code_ = semantic_code[b, :cond_lengths[b]]
            semantic_codes.append(semantic_code_)
            code_lens.append(semantic_code_.shape[0])
        # logger.error(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] quantize time: {time.time() - stt:.2f}s')

        # Get Conditioning & Emotion Vector
        # stt = time.time()
        feat_t = spk_cond_emb.transpose(1, 2)
        cond_lengths_device = cond_lengths.to(spk_cond_emb.device)
        
        conditioning = self.gpt.get_conditioning(feat_t, cond_lengths_device)  # [b, 32, 1280]
        emo_vec = self.gpt.get_emovec(spk_cond_emb, cond_lengths_device)  # [b, 1280]
        # logger.error(f'[worker:{self.worker_id}] [gpu:{self.gpu_id}] get_conditioning & get_emovec time: {time.time() - stt:.2f}s')

        conditioning = conditioning.to(device="cpu", dtype=torch.float16)
        emo_vec = emo_vec.to(device="cpu", dtype=torch.float16)

        # Create Object
        processed_datas = []
        for b in range(batch_size):
            processed_data = ProcessedData(
                text_ids=text_ids_list[b],
                codes=semantic_codes[b].to(device="cpu", dtype=torch.int16),
                text_len=text_lens[b],
                code_len=code_lens[b],
                condition=conditioning[b].clone(),
                emo_vec=emo_vec[b].clone(),
                duration=durations[b],
            )
            processed_datas.append(processed_data.to_numpy())
        return processed_datas
