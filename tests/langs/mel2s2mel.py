import os
from pathlib import Path
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

import librosa
from omegaconf import OmegaConf
import torch
import torchaudio

from transformers import SeamlessM4TFeatureExtractor
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
import safetensors
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.audio import mel_spectrogram

model_dir = "./checkpoints/IndexTTS-2-vLLM"
cfg_path = os.path.join(model_dir, "config.yaml")
cfg = OmegaConf.load(cfg_path)

device = "cuda"
dtype = torch.float16
use_cuda_kernel = False
sampling_rate = 22050


# ******************* load model *********************
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

s2mel_path = os.path.join(model_dir, cfg.s2mel_checkpoint)
s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
s2mel, _, _, _ = load_checkpoint2(
    s2mel,
    None,
    s2mel_path,
    load_only_params=True,
    ignore_modules=[],
    is_distributed=False,
)
s2mel = s2mel.to(device=device, dtype=dtype)
s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
s2mel.eval()
print(f">> s2mel weights restored from: {s2mel_path}")

# load campplus_model
# campplus_ckpt_path = hf_hub_download(
#     "funasr/campplus", filename="campplus_cn_common.bin", cache_dir=os.path.join(model_dir, "campplus")
# )
campplus_ckpt_path = os.path.join(model_dir, "campplus/campplus_cn_common.bin")
campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
campplus_model = campplus_model.to(device=device, dtype=dtype)
campplus_model.eval()
print(f">> campplus_model weights restored from: {campplus_ckpt_path}")


bigvgan_name = cfg.vocoder.name
bigvgan = bigvgan.BigVGAN.from_pretrained(os.path.join(model_dir, "bigvgan"), use_cuda_kernel=use_cuda_kernel)
bigvgan = bigvgan.to(device=device, dtype=dtype)
bigvgan.remove_weight_norm()
bigvgan.eval()
print(">> bigvgan weights restored from:", bigvgan_name)

mel_fn_args = {
    "n_fft": cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
    "win_size": cfg.s2mel['preprocess_params']['spect_params']['win_length'],
    "hop_size": cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
    "num_mels": cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
    "sampling_rate": cfg.s2mel["preprocess_params"]["sr"],
    "fmin": cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
    "fmax": None if cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
    "center": False
}
mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

def _load_and_cut_audio(audio_path,max_audio_length_seconds,verbose=False,sr=None):
    if not sr:
        audio, sr = librosa.load(audio_path)
    else:
        audio, _ = librosa.load(audio_path,sr=sr)
    audio = torch.tensor(audio).unsqueeze(0)
    max_audio_samples = int(max_audio_length_seconds * sr)

    if audio.shape[1] > max_audio_samples:
        if verbose:
            print(f"Audio too long ({audio.shape[1]} samples), truncating to {max_audio_samples} samples")
        audio = audio[:, :max_audio_samples]
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
input_wav_path = r"assets\audio\vo_BZLQ001_6_hutao_07.wav"
# input_wav_path = r"assets\audio\vo_klee_friendship_02.wav"
output_path = rf"outputs\{Path(input_wav_path).stem}_bigvgan_out.wav"
diffusion_steps = 25
inference_cfg_rate = 0.7

with torch.no_grad():
    with torch.amp.autocast(device_type=device, dtype=dtype):
        audio,sr = _load_and_cut_audio(input_wav_path,15)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
        inputs = extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"]
        attention_mask = inputs["attention_mask"]
        input_features = input_features.to(device)
        attention_mask = attention_mask.to(device)
        spk_cond_emb = get_emb(input_features, attention_mask)

        _, S_ref = semantic_codec.quantize(spk_cond_emb)
        ref_mel = mel_fn(audio_22k.to(device).float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
        feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device),
                                                    num_mel_bins=80,
                                                    dither=0,
                                                    sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)  # feat2另外一个滤波器能量组特征[922, 80]
        style = campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]

        # print(f"S_ref: {S_ref.shape}, ref_target_lengths: {ref_target_lengths}")
        prompt_condition = s2mel.models['length_regulator'](S_ref,
                                                            ylens=ref_target_lengths,
                                                            n_quantizers=3,
                                                            f0=None)[0]
        
        cat_condition = torch.cat([prompt_condition, prompt_condition], dim=1)
        vc_target = s2mel.models['cfm'].inference(cat_condition,
                                                torch.LongTensor([cat_condition.size(1)]).to(device),
                                                ref_mel, style, None, diffusion_steps,
                                                inference_cfg_rate=inference_cfg_rate)
        vc_target = vc_target[:, :, ref_mel.size(-1):]
        wav = bigvgan(vc_target.float()).squeeze().unsqueeze(0)

        wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
        wav = wav.cpu()  # to cpu

        # 直接保存音频到指定路径中
        if os.path.dirname(output_path) != "":
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)