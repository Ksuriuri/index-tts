import os
import sys
import argparse
import torch
from tqdm import tqdm

# 添加路径以确保能导入 indextts (与 webui.py 保持一致)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

from indextts.infer_v2 import IndexTTS2

def load_scp_as_dict(path, root_dir=None):
    """读取 scp/text 文件为字典: {id: content}"""
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                key, value = parts
                # 如果是 wav 路径且是相对路径，尝试拼接 root_dir
                if root_dir and path.endswith('.scp') and not value.startswith('/'):
                    full_path = os.path.join(root_dir, value)
                    if os.path.exists(full_path):
                        value = full_path
                    # 还有一种情况，CV3-Eval 的 scp 里的路径可能是基于 data 目录的相对路径
                    # 这里做一个简单的容错处理，如果找不到文件，请手动修改这里
                data[key] = value
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./checkpoints/IndexTTS-2-vLLM", help="模型路径")
    parser.add_argument("--data_dir", type=str, default="/mnt/data_sdd/hhy/CV3-Eval/data/zero_shot/ja", 
                        help="CV3-Eval ja 数据目录")
    parser.add_argument("--output_dir", type=str, default="/mnt/data_sdd/hhy/CV3-Eval/model_data/zero_shot/ja", help="生成的音频存放目录")
    parser.add_argument("--cv3_root", type=str, default="/mnt/data_sdd/hhy/CV3-Eval", help="CV3-Eval 项目根目录，用于补全 scp 中的相对路径")
    
    args = parser.parse_args()

    # 1. 初始化模型
    print(f"Loading model from {args.model_dir}...")
    # 检查 CUDA
    use_deepspeed = False # 简单起见，批量推理通常不需要 deepspeed 除非显存极度吃紧，可自行开启
    use_fp16 = False
    
    tts = IndexTTS2(
        model_dir=args.model_dir,
        cfg_path=os.path.join(args.model_dir, "config.yaml"),
        use_fp16=use_fp16,
        use_deepspeed=use_deepspeed,
        use_cuda_kernel=False 
    )

    # 2. 准备数据
    text_path = os.path.join(args.data_dir, "text")
    prompt_scp_path = os.path.join(args.data_dir, "prompt_wav.scp")
    
    print(f"Reading text from {text_path}")
    texts = load_scp_as_dict(text_path)
    
    print(f"Reading prompts from {prompt_scp_path}")
    # 注意：prompt_wav.scp 里面的路径可能是相对路径，需要 args.cv3_root 来补全
    # 如果 prompt_wav.scp 里写的是 absolute path，这里 root_dir 传什么都不影响
    prompts = load_scp_as_dict(prompt_scp_path, root_dir=args.cv3_root)

    # 3. 准备输出目录
    # 必须建立一个 wavs 子目录以符合评测脚本的 find 逻辑
    save_dir = os.path.join(args.output_dir, "wavs") 
    os.makedirs(save_dir, exist_ok=True)

    # 4. 开始生成
    print(f"Start generating {len(texts)} utterances...")
    
    # 过滤出共同的 ID (防止数据不齐)
    common_ids = set(texts.keys()) & set(prompts.keys())
    sorted_ids = sorted(list(common_ids))
    
    if len(sorted_ids) < len(texts):
        print(f"Warning: Text file has {len(texts)} lines but matched {len(sorted_ids)} with prompt scp.")

    for utt_id in tqdm(sorted_ids):
        text = texts[utt_id]
        prompt_wav = prompts[utt_id]
        
        output_path = os.path.join(save_dir, f"{utt_id}.wav")
        if os.path.exists(output_path):
            print(f"Skipping {utt_id}: output file already exists.")
            continue
        
        # 检查 prompt 文件是否存在
        if not os.path.exists(prompt_wav):
            # 尝试通过 data_dir 修正 (CV3-Eval 有时 scp 里的路径比较奇怪)
            # 假设 scp 内容是: data/zero_shot/ja/waveform/xxx.wav
            # 而 data_dir 是: .../CV3-Eval/data/zero_shot/ja
            # 我们需要回退 3 层找到 root
            possible_path = os.path.abspath(os.path.join(args.data_dir, "../../../", prompt_wav))
            if os.path.exists(possible_path):
                prompt_wav = possible_path
            else:
                print(f"ERROR: Prompt file not found: {prompt_wav} for id {utt_id}")
                continue

        try:
            # 调用推理
            # 这里的参数对应 webui.py 中的 gen_single 逻辑
            tts.infer(
                spk_audio_prompt=prompt_wav,
                text=text,
                output_path=output_path,
                emo_audio_prompt=None, # Zero-shot 默认用 prompt 控制音色和风格
                emo_alpha=1.0,
                verbose=False,
                max_text_tokens_per_segment=120, # 默认分句长度
                # do_sample=True,
                # top_p=args.top_p,
                # top_k=0, # WebUI default logic checks if >0, usually 0 or None means ignore
                # temperature=args.temperature,
                # length_penalty=0.0,
                # num_beams=1,
                # repetition_penalty=args.repetition_penalty,
                # max_mel_tokens=1500
            )
        except Exception as e:
            print(f"Failed to generate {utt_id}: {e}")

    print(f"Generation finished. Saved to {save_dir}")

if __name__ == "__main__":
    main()