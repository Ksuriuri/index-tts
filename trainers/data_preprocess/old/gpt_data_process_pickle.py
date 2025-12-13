import os
import random
import sys
import pickle
import glob
from tqdm import tqdm

# 原始路径设置
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
from trainers.utils import ProcessedData

# ******************* 配置路径 *********************
DATASET_ROOT = "/home/tanhe/hhy/datasets/raw_data/WutheringWaves_Dataset/jp"
OUTPUT_DIR = "./train_data"
val_num = 128
model_dir = "./checkpoints/IndexTTS-2-vLLM"
random.seed(42)

# ******************* Load Model (保持原样) *********************
cfg_path = os.path.join(model_dir, "config.yaml")
cfg = OmegaConf.load(cfg_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

print(f"Loading models to {device}...")

bpe_path = os.path.join(model_dir, "jp_bpe.model")
normalizer = TextNormalizer()
normalizer.load()
tokenizer = TextTokenizer(bpe_path, normalizer)

gpt = UnifiedVoice(**cfg.gpt)
gpt_path = os.path.join(model_dir, cfg.gpt_checkpoint)
load_checkpoint(gpt, gpt_path)
gpt = gpt.to(device)
gpt.eval() # 确保进入eval模式

extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
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

# ******************* Helper Functions *********************

def _load_audio(audio_path, max_audio_length_seconds, verbose=False, sr=None):
    try:
        if not sr:
            audio, sr = librosa.load(audio_path)
        else:
            audio, _ = librosa.load(audio_path, sr=sr)
        
        audio = torch.tensor(audio).unsqueeze(0)
        max_audio_samples = int(max_audio_length_seconds * sr)
        
        if audio.shape[1] > max_audio_samples:
            if verbose: print(f"Skipping {audio_path}: Audio too long.")
            return None, None
        return audio, sr
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return None, None

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

# ******************* Main Processing Loop *********************

def process_all_files():
    all_samples = []
    
    # 获取所有wav文件路径
    # 结构: root/character_name/xxx.wav
    search_pattern = os.path.join(DATASET_ROOT, "**", "*.wav")
    wav_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(wav_files)} audio files. Starting processing...")
    
    # 使用tqdm显示进度
    for wav_path in tqdm(wav_files, desc="Processing Audio"):
        # 构建对应的 lab 文件路径
        lab_path = os.path.splitext(wav_path)[0] + ".lab"
        
        if not os.path.exists(lab_path):
            print(f"Warning: Missing label file for {wav_path}")
            continue
            
        try:
            # 1. 读取文本
            with open(lab_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                print(f"Warning: Empty text in {lab_path}")
                continue

            with torch.no_grad():
                # 2. Tokenize Text
                text_tokens_list = tokenizer.tokenize(text)
                text_ids = tokenizer.convert_tokens_to_ids(text_tokens_list)
                
                if len(text_ids) == 0:
                    print(f"Warning: No tokens generated for {lab_path}")
                    continue
                    
                text_ids = torch.tensor(text_ids, dtype=torch.int32)  # [text_len]
                text_len = len(text_ids)

                # 3. Process Audio
                # 限制最大时长36秒，采样率强制16000
                audio, sr = _load_audio(wav_path, 36, sr=16000)
                if audio is None:
                    continue
                
                # audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
                audio_16k = audio

                # 4. Extract Features
                inputs = extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
                input_features = inputs["input_features"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                # 5. Get Speaker Condition Embedding
                spk_cond_emb = get_emb(input_features, attention_mask)

                # 6. Quantize / Codec
                cond_lengths = attention_mask.sum(dim=1).long()
                semantic_code, _ = semantic_codec.quantize(spk_cond_emb)  # [1, code_len]
                code_len = semantic_code.shape[1]

                # 7. Get Conditioning & Emotion Vector
                feat_t = spk_cond_emb.transpose(1, 2)
                cond_lengths_device = cond_lengths.to(spk_cond_emb.device)
                
                conditioning = gpt.get_conditioning(feat_t, cond_lengths_device)  # [1, 32, 1280]
                emo_vec = gpt.get_emovec(spk_cond_emb, cond_lengths_device)  # [1, 1280]

                # 8. Create Sample Object
                # 注意：所有tensor都转回CPU，否则显存会爆炸，且pickle保存时不需要在GPU上
                sample = ProcessedData(
                    text_ids=text_ids.to(device="cpu", dtype=torch.int32),
                    codes=semantic_code.to(device="cpu", dtype=torch.int32),
                    text_len=text_len,
                    code_len=code_len,
                    condition=conditioning.to(device="cpu", dtype=torch.float32),
                    emo_vec=emo_vec.to(device="cpu", dtype=torch.float32),
                )
                
                all_samples.append(sample)

        except Exception as e:
            print(f"Error processing {wav_path}: {str(e)}")
            continue

    print(f"\nProcessing complete. Successfully processed {len(all_samples)} samples.")
    
    # ******************* Save to Pickle *********************
    print(f"Saving to {OUTPUT_DIR}...")
    try:
        random.shuffle(all_samples)
        with open(os.path.join(OUTPUT_DIR, "train_samples.pkl"), 'wb') as f:
            pickle.dump(all_samples[:-val_num], f)
        with open(os.path.join(OUTPUT_DIR, "val_samples.pkl"), 'wb') as f:
            pickle.dump(all_samples[-val_num:], f)
        print("Done.")
    except Exception as e:
        print(f"Error saving pickle file: {e}")

if __name__ == "__main__":
    process_all_files()