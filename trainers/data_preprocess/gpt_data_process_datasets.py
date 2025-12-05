import os
import random
import sys
import glob
import numpy as np # 新增 numpy
from tqdm import tqdm
from datasets import Dataset, Features, Sequence, Value, Array2D, Array3D # 新增 datasets 库

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
# from trainers.train_gpt_v2 import Sample # 不再需要 Sample 类

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
gpt.eval()

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

# ******************* Generator Function *********************

def data_generator():
    """
    生成器函数，每次 yield 一个处理好的样本字典。
    """
    # 获取所有wav文件路径
    search_pattern = os.path.join(DATASET_ROOT, "**", "*.wav")
    wav_files = glob.glob(search_pattern, recursive=True)
    
    # 可以在这里打乱文件顺序，或者后续在 dataset 中 shuffle
    random.shuffle(wav_files)
    
    print(f"Found {len(wav_files)} audio files. Starting processing...")
    
    for wav_path in tqdm(wav_files, desc="Processing Audio"):
        lab_path = os.path.splitext(wav_path)[0] + ".lab"
        
        if not os.path.exists(lab_path):
            continue
            
        try:
            # 1. 读取文本
            with open(lab_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                continue

            with torch.no_grad():
                # 2. Tokenize Text
                text_tokens_list = tokenizer.tokenize(text)
                text_ids = tokenizer.convert_tokens_to_ids(text_tokens_list)
                
                if len(text_ids) == 0:
                    continue
                    
                text_ids = torch.tensor(text_ids, dtype=torch.int32)  # [text_len]
                text_len = len(text_ids)

                # 3. Process Audio
                audio, sr = _load_audio(wav_path, 36, sr=16000)
                if audio is None:
                    continue
                
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

                # 8. Yield Dictionary (Convert to Numpy)
                # Datasets 库存储 numpy array 效率最高
                yield {
                    "text_ids": text_ids.numpy().astype(np.int64),
                    "codes": semantic_code.squeeze(0).cpu().numpy().astype(np.int64), # 降维 [code_len]
                    "text_len": int(text_len),
                    "code_len": int(code_len),
                    "condition": conditioning.squeeze(0).cpu().numpy().astype(np.float32), # [32, 1280]
                    "emo_vec": emo_vec.squeeze(0).cpu().numpy().astype(np.float32),        # [1280]
                }

        except Exception as e:
            print(f"Error processing {wav_path}: {str(e)}")
            continue

# ******************* Main Processing Loop *********************

def process_and_save():
    # 定义 Features (可选，显式定义可以确保数据类型正确)
    # 这一步不是强制的，但对于大型数据集是最佳实践
    features = Features({
        "text_ids": Sequence(Value("int64")),
        "codes": Sequence(Value("int64")),
        "text_len": Value("int64"),
        "code_len": Value("int64"),
        "condition": Array2D(shape=(32, 1280), dtype="float32"),
        "emo_vec": Sequence(Value("float32"), length=1280), # 或者 Array1D
    })

    print("Initializing Dataset from generator...")
    # 使用 from_generator 创建数据集
    # 这一步会边生成边缓存到本地缓存目录，不会炸内存
    ds = Dataset.from_generator(data_generator, features=features)
    
    print(f"Total samples processed: {len(ds)}")
    
    if len(ds) == 0:
        print("Error: No samples processed.")
        return

    # Shuffle and Split
    print("Shuffling and splitting dataset...")
    ds = ds.shuffle(seed=42)
    
    # train_test_split
    # test_size 可以是比例(float)或数量(int)
    dataset_dict = ds.train_test_split(test_size=val_num)
    
    train_ds = dataset_dict["train"]
    val_ds = dataset_dict["test"]
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    
    # Save to disk (Apache Arrow format)
    # 这将生成文件夹，包含 data-00000-of-xxxxx.arrow 等文件
    train_path = os.path.join(OUTPUT_DIR, "train")
    val_path = os.path.join(OUTPUT_DIR, "val")
    
    print(f"Saving train dataset to {train_path}...")
    train_ds.save_to_disk(train_path)
    
    print(f"Saving val dataset to {val_path}...")
    val_ds.save_to_disk(val_path)
    
    print("All done!")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    process_and_save()