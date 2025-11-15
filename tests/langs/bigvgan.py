import os
from pathlib import Path
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

import librosa
from omegaconf import OmegaConf
import torch
import torchaudio

from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.audio import mel_spectrogram

model_dir = "./checkpoints/IndexTTS-2-vLLM"
cfg_path = os.path.join(model_dir, "config.yaml")
cfg = OmegaConf.load(cfg_path)

device = "cuda"
use_cuda_kernel = False
sampling_rate = 22050

bigvgan_name = cfg.vocoder.name
bigvgan = bigvgan.BigVGAN.from_pretrained(os.path.join(model_dir, "bigvgan"), use_cuda_kernel=use_cuda_kernel)
bigvgan = bigvgan.to(device)
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


# infer
input_wav_path = r"assets\audio\vo_BZLQ001_6_hutao_07.wav"
output_path = rf"outputs\{Path(input_wav_path).stem}_bigvgan_out.wav"

with torch.no_grad():
    audio,sr = _load_and_cut_audio(input_wav_path,15)
    audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
    ref_mel = mel_fn(audio_22k.to(device).float())

    wav = bigvgan(ref_mel.float()).squeeze().unsqueeze(0)

    wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
    wav = wav.cpu()  # to cpu

    # 直接保存音频到指定路径中
    if os.path.dirname(output_path) != "":
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)