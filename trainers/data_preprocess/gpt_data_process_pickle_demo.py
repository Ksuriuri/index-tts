import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

import librosa
from omegaconf import OmegaConf
import torch
import torchaudio

from transformers import SeamlessM4TFeatureExtractor
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
import safetensors
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer
from trainers.train_gpt_v2 import Sample

model_dir = "./checkpoints/IndexTTS-2-vLLM"
cfg_path = os.path.join(model_dir, "config.yaml")
cfg = OmegaConf.load(cfg_path)

device = "cuda"
dtype = torch.float32


# ******************* load model *********************
bpe_path = os.path.join(model_dir, "jp_bpe.model")
normalizer = TextNormalizer()
normalizer.load()
tokenizer = TextTokenizer(bpe_path, normalizer)

gpt = UnifiedVoice(**cfg.gpt)
gpt_path = os.path.join(model_dir, cfg.gpt_checkpoint)
load_checkpoint(gpt, gpt_path)
gpt = gpt.to(device)

extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
    # "facebook/w2v-bert-2.0"
    os.path.join(model_dir, "w2v-bert-2.0")
)
semantic_model, semantic_mean, semantic_std = build_semantic_model(
    os.path.join(model_dir, cfg.w2v_stat),
    os.path.join(model_dir, "w2v-bert-2.0")
)
semantic_model = semantic_model.to(device=device, dtype=dtype)
semantic_model.eval()
semantic_mean = semantic_mean.to(device=device, dtype=dtype)
semantic_std = semantic_std.to(device=device, dtype=dtype)

semantic_codec = build_semantic_codec(cfg.semantic_codec)
semantic_code_ckpt = os.path.join(model_dir, "semantic_codec/model.safetensors")
safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
semantic_codec = semantic_codec.to(device=device, dtype=dtype)
semantic_codec.eval()
print('>> semantic_codec weights restored from: {}'.format(semantic_code_ckpt))

def _load_audio(audio_path,max_audio_length_seconds,verbose=False,sr=None):
    if not sr:
        audio, sr = librosa.load(audio_path)
    else:
        audio, _ = librosa.load(audio_path,sr=sr)
    audio = torch.tensor(audio).unsqueeze(0)
    max_audio_samples = int(max_audio_length_seconds * sr)
    if audio.shape[1] > max_audio_samples:
        return None, None
    return audio, sr

@torch.no_grad()
def get_emb(input_features, attention_mask):
    vq_emb = semantic_model(
        input_features=input_features,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    feat = vq_emb.hidden_states[17]  # (B, T, C)
    feat = (feat - semantic_mean) / semantic_std
    return feat


# ************************ infer *********************
input_wav_path = "/home/tanhe/hhy/datasets/raw_data/WutheringWaves_Dataset/jp/白芷/ja_vo_Huanglong_main_1_2_5_46.wav"
text = "弱い順に、水風級巨浪級怒涛級津波級……そして、規格外の鳴式というふうに区分される。"  # from ja_vo_Huanglong_main_1_2_5_46.lab

with torch.no_grad():
    text_tokens_list = tokenizer.tokenize(text)
    text_ids = tokenizer.convert_tokens_to_ids(text_tokens_list)
    text_ids = torch.tensor(text_ids, dtype=torch.int32)  # [text_len]
    text_len = len(text_ids)

    audio, sr = _load_audio(input_wav_path, 36, sr=16000)
    if audio is None:
        raise ValueError("Failed to load audio")
    audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
    inputs = extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"]
    attention_mask = inputs["attention_mask"]
    input_features = input_features.to(device)
    attention_mask = attention_mask.to(device)
    spk_cond_emb = get_emb(input_features, attention_mask)

    cond_lengths = attention_mask.sum(dim=1).long()
    semantic_code, _ = semantic_codec.quantize(spk_cond_emb)  # [1, code_len]
    code_len = semantic_code.shape[1]

    feat_t = spk_cond_emb.transpose(1, 2)
    cond_lengths_device = cond_lengths.to(spk_cond_emb.device)
    conditioning = gpt.get_conditioning(feat_t, cond_lengths_device)  # [1, 32, 1280]
    emo_vec = gpt.get_emovec(spk_cond_emb, cond_lengths_device)  # [1, 1280]

    sample = Sample(
        text_ids=text_ids.to(device="cpu", dtype=torch.int32),
        codes=semantic_code.to(device="cpu", dtype=torch.int32),
        text_len=text_len,
        code_len=code_len,
        condition=conditioning.to(device="cpu", dtype=torch.float32),
        emo_vec=emo_vec.to(device="cpu", dtype=torch.float32),
    )