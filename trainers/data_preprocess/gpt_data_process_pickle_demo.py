import os
import random
import sys
import time
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
    os.path.join(model_dir, "w2v-bert-2.0"),
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
input_wav_path = "/mnt/data_3t_1/datasets/raw_data/WutheringWaves_Dataset/jp/白芷/ja_vo_Huanglong_main_1_2_5_46.wav"
text = "弱い順に、水風級巨浪級怒涛級津波級……そして、規格外の鳴式というふうに区分される。"  # from ja_vo_Huanglong_main_1_2_5_46.lab

# stt = time.time()
# audio, sr = _load_audio(input_wav_path, 36, sr=16000)
# if audio is None:
#     raise ValueError("Failed to load audio")
# audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
# print(f"load audio time: {time.time() - stt}")

batch_size = 16
random.seed(42)
torch.manual_seed(42)

fake_audio = [torch.rand(int(16000 * random.randint(100, 800) / 100)) for _ in range(batch_size)]
audio_16ks = fake_audio

text_tokens_list = tokenizer.tokenize(text)
text_ids = tokenizer.convert_tokens_to_ids(text_tokens_list)
text_ids = torch.tensor(text_ids, dtype=torch.int32)  # [text_len]
text_len = len(text_ids)

with torch.no_grad():

    # stt_total = time.time()
    # for audio_16k in audio_16ks:
    #     stt = time.time()
    #     # for _ in range(1000):
    #     inputs = extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
    #     print(f"extract features time: {time.time() - stt}")

    #     stt = time.time()
    #     input_features = inputs["input_features"]
    #     attention_mask = inputs["attention_mask"]
    #     input_features = input_features.to(device)
    #     attention_mask = attention_mask.to(device)
    #     spk_cond_emb = get_emb(input_features, attention_mask)
    #     print(f"get emb time: {time.time() - stt}, spk_cond_emb: {spk_cond_emb.shape}, input_features: {input_features.shape}")

    #     stt = time.time()
    #     cond_lengths = attention_mask.sum(dim=1).long()
    #     semantic_code, _ = semantic_codec.quantize(spk_cond_emb)  # [1, code_len]
    #     code_len = semantic_code.shape[1]
    #     print(f"quantize time: {time.time() - stt}, semantic_code: {semantic_code.shape}")

    #     stt = time.time()
    #     feat_t = spk_cond_emb.transpose(1, 2)
    #     cond_lengths_device = cond_lengths.to(spk_cond_emb.device)
    #     conditioning = gpt.get_conditioning(feat_t, cond_lengths_device)  # [1, 32, 1280]
    #     emo_vec = gpt.get_emovec(spk_cond_emb, cond_lengths_device)  # [1, 1280]
    #     print(f"get conditioning time: {time.time() - stt}, conditioning: {conditioning.shape}, emo_vec: {emo_vec.shape}")
    #     print()

    for _ in range(2):
        stt_total = time.time()
        stt = time.time()
        # for _ in range(1000):
        inputs = extract_features(audio_16ks, sampling_rate=16000, return_tensors="pt")
        print(f"extract features time: {time.time() - stt}")

        stt = time.time()
        input_features = inputs["input_features"]
        attention_mask = inputs["attention_mask"]
        input_features = input_features.to(device)
        attention_mask = attention_mask.to(device)
        spk_cond_emb = get_emb(input_features, attention_mask)
        print(f"get emb time: {time.time() - stt}, spk_cond_emb: {spk_cond_emb.shape}, input_features: {input_features.shape}")

        for b in range(batch_size):
            spk_cond_emb_ = spk_cond_emb[b: b+1, :attention_mask[b].sum()]
        stt = time.time()
        cond_lengths = attention_mask.sum(dim=1).long()
        semantic_code, _ = semantic_codec.quantize(spk_cond_emb)  # [1, code_len]
        code_len = semantic_code.shape[1]
        print(f"quantize time: {time.time() - stt}, semantic_code: {semantic_code.shape}")

        stt = time.time()
        feat_t = spk_cond_emb.transpose(1, 2)
        cond_lengths_device = cond_lengths.to(spk_cond_emb.device)
        conditioning = gpt.get_conditioning(feat_t, cond_lengths_device)  # [1, 32, 1280]
        emo_vec = gpt.get_emovec(spk_cond_emb, cond_lengths_device)  # [1, 1280]
        print(f"get conditioning time: {time.time() - stt}, conditioning: {conditioning.shape}, emo_vec: {emo_vec.shape}")
        print()

print(f"total time: {time.time() - stt_total}")