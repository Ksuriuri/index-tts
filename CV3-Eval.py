import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# 添加路径以确保能导入 indextts
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

from indextts.infer_v2 import IndexTTS2

def load_scp_as_dict(path, root_dir=None):
    """读取 scp/text 文件为字典: {id: content}"""
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                key, value = parts
                if root_dir and path.endswith('.scp') and not value.startswith('/'):
                    full_path = os.path.join(root_dir, value)
                    if os.path.exists(full_path):
                        value = full_path
                data[key] = value
    return data

def worker(gpu_id, task_ids, texts, prompts, args):
    """
    单个 GPU 的工作进程
    """
    # 强制该进程只看到指定的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")

    print(f"Process [GPU {gpu_id}]: Loading model...")
    # 在每个进程内部初始化模型，确保模型加载到当前进程的 GPU 上
    tts = IndexTTS2(
        model_dir=args.model_dir,
        cfg_path=os.path.join(args.model_dir, "config.yaml"),
        use_fp16=args.use_fp16,
        use_deepspeed=False, # 推理通常不开启
        use_cuda_kernel=False 
    )

    save_dir = os.path.join(args.output_dir, "wavs")
    
    # 使用 tqdm 显示每个进程的进度 (加上 position 参数防止进度条重叠)
    pbar = tqdm(task_ids, desc=f"GPU {gpu_id}", position=gpu_id)
    
    for utt_id in pbar:
        text = texts[utt_id]
        prompt_wav = prompts[utt_id]
        output_path = os.path.join(save_dir, f"{utt_id}.wav")

        if os.path.exists(output_path):
            continue

        # 路径容错逻辑
        if not os.path.exists(prompt_wav):
            possible_path = os.path.abspath(os.path.join(args.data_dir, "../../../", prompt_wav))
            if os.path.exists(possible_path):
                prompt_wav = possible_path
            else:
                continue

        try:
            tts.infer(
                spk_audio_prompt=prompt_wav,
                text=text,
                output_path=output_path,
                emo_audio_prompt=None,
                emo_alpha=1.0,
                verbose=False,
                max_text_tokens_per_segment=120
            )
        except Exception as e:
            print(f"GPU {gpu_id} Error {utt_id}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./checkpoints/IndexTTS-2-vLLM")
    parser.add_argument("--data_dir", type=str, default="/mnt/data_sdd/hhy/CV3-Eval/data/zero_shot/ja")
    parser.add_argument("--output_dir", type=str, default="/mnt/data_sdd/hhy/CV3-Eval/model_data/zero_shot/ja")
    parser.add_argument("--cv3_root", type=str, default="/mnt/data_sdd/hhy/CV3-Eval")
    parser.add_argument("--gpus", type=str, default="0", help="使用的 GPU 列表，例如 '0,1,2,3'")
    parser.add_argument("--use_fp16", action="store_true", help="是否开启 fp16 推理")
    
    args = parser.parse_args()

    # 1. 设置使用的 GPU
    gpu_list = [int(i) for i in args.gpus.split(",")]
    num_gpus = len(gpu_list)
    print(f"Using GPUs: {gpu_list}")

    # 2. 读取数据
    text_path = os.path.join(args.data_dir, "text")
    prompt_scp_path = os.path.join(args.data_dir, "prompt_wav.scp")
    texts = load_scp_as_dict(text_path)
    prompts = load_scp_as_dict(prompt_scp_path, root_dir=args.cv3_root)

    common_ids = sorted(list(set(texts.keys()) & set(prompts.keys())))
    print(f"Total matched utterances: {len(common_ids)}")

    # 3. 准备输出目录
    save_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(save_dir, exist_ok=True)

    # 4. 数据切分
    # 将任务平分给各个 GPU
    chunks = [common_ids[i::num_gpus] for i in range(num_gpus)]

    # 5. 启动多进程
    mp.set_start_method('spawn', force=True) # CUDA 必须使用 spawn
    processes = []
    for i in range(num_gpus):
        p = mp.Process(
            target=worker, 
            args=(gpu_list[i], chunks[i], texts, prompts, args)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"All processes finished. Results saved to {save_dir}")

if __name__ == "__main__":
    main()