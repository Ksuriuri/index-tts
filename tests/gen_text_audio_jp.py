import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from indextts.infer_v2 import IndexTTS2

# ================= 数据定义 =================

# SUB_NAME = ""
# SUFFIX = ".mp3"

SUB_NAME = "_denoised"
SUFFIX = ".wav"

AUDIO_FILES = [
    "2", "3", "4", "5", "6", 
    "7", "8", "9", "10", "11",
    "12", "13", "14", "15", "16"
]

AUDIO_FILES = [f"{i}{SUB_NAME}{SUFFIX}" for i in AUDIO_FILES]

NORM_TEXTS = [
    "本日はお集まりいただき、誠にありがとうございます。",
    "今日は少し昔の話をします。",
    "それでは、内容を説明します。",
    "今日は番組をお聞きいただき、ありがとうございます。",
    "今日も番組を聞いてくれてありがとう。",
    "今日も元気にお届けします。",
    "ここからが本番だ。",
    "今日は落ち着いたテーマでお話しします。",
    "リラックスして聞いてください。",
    "状況は理解した。",
    "本日もありがとうございます。",
    "最後までお聞きいただき感謝します。",
    "本日は最後までお聞きいただき、ありがとうございます。",
    "今日も元気に配信スタート！",
    "今日の話、最後まで聞いてね。"
]

CHARAC_TEXTS = [
    "長年、教育の現場に立ってきた立場から、落ち着いて物事を考える大切さをお伝えしたいと思います。",
    "私が若かった昭和の時代には、今とは全く違う価値観がありました。",
    "経験を積んだからこそ、表面的な結論ではなく本質を見る必要があります。",
    "テンポよく、分かりやすく、ポイントだけをお話ししていきます。",
    "僕の体験をベースに、正直に話します。",
    "ノリよく、感情そのままで話していきます。",
    "自分の信念を、言葉にしてぶつける。",
    "日常の中で感じたことを、丁寧に共有します。",
    "等身大の言葉で話したいと思います。",
    "感情を抑え、理性的に対処する。",
    "これまでの経験を、静かに共有したいと思います。",
    "人生の締めくくりとして、伝えたい言葉があります。",
    "長い人生の中で学んだことを、静かに皆さんと共有したいと思います。",
    "テンポよく、楽しく、今気になってることを話すよ！",
    "自分の気持ちを、そのまま言葉にして話してみる。"
]

CORNER_CASES = [
    "あ、あの……えーと……その……時間を大切に……し、し、しましょう……",
    "人生とは三十年、四十年、五十年……一瞬で、しかし永遠のように、流れていくもの、なのかもしれません。",
    "2025年3月31日23時59分59秒、CPU使用率99.8%、メモリ32GB中31.6GB消費、システムは――応答不能。",
    "え、ちょ、まっ、待って、待って――今何も見えないよ",
    "API叩いて、JSON返って、404出て、いや待って401！？もう何も信じない。",
    "マジで！？え、無理、無理無理無理、ちょっと待って脳が追いつかない！",
    "りょ、了解不能、思考回路ショートォォ！",
    "昨日、今日、明日、過去、現在、未来が全部一緒に押し寄せてきた。",
    "「私」「わたし」「ワタシ」って、結局どれが本当の私なの？",
    "感情値0.01%、怒り99.99%、理性――強制終了。",
    "……沈黙。五秒。十秒。誰も、何も、話さない。",
    "生まれて、学んで、働いて、失って、そして――まだ、終わらない。",
    "……昔のことは、覚えているのに……さっきの話は……もう、思い出せません……",
    "え！？なに！？まじで！？ちょ、早すぎて頭ついてこないんだけど！？",
    "なんで？どうして？理由は？説明して？でも納得できない、まだ。"
]

# ===========================================

def worker(gpu_id, tasks, args):
    """
    单个 GPU 的工作进程
    tasks: list of dict {'prompt_path': str, 'text': str, 'output_path': str}
    """
    # 强制该进程只看到指定的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"Process [GPU {gpu_id}]: Loading model...")
    
    # 初始化模型
    try:
        tts = IndexTTS2(
            model_dir=args.model_dir,
            cfg_path=os.path.join(args.model_dir, "config.yaml"),
            use_fp16=args.use_fp16,
            use_deepspeed=False, 
            use_cuda_kernel=False 
        )
    except Exception as e:
        print(f"Failed to load model on GPU {gpu_id}: {e}")
        return

    # 使用 tqdm 显示进度
    pbar = tqdm(tasks, desc=f"GPU {gpu_id}", position=gpu_id)
    
    for task in pbar:
        prompt_path = task['prompt_path']
        text = task['text']
        output_path = task['output_path']

        # 检查输出是否存在
        if os.path.exists(output_path):
            continue

        # 检查 prompt 是否存在
        if not os.path.exists(prompt_path):
            print(f"[Warning] Prompt file not found: {prompt_path}")
            continue

        try:
            tts.infer(
                spk_audio_prompt=prompt_path,
                text=text,
                output_path=output_path,
                emo_audio_prompt=None,
                emo_alpha=1.0,
                verbose=False,
                max_text_tokens_per_segment=120
            )
        except Exception as e:
            print(f"GPU {gpu_id} Error processing {os.path.basename(output_path)}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./checkpoints/IndexTTS-2-vLLM")
    # 输入输出路径硬编码在逻辑中，但保留参数以便微调
    parser.add_argument("--raw_audio_dir", type=str, default=f"./outputs/test_audio/raw_audio{SUB_NAME}")
    parser.add_argument("--gen_audio_dir", type=str, default=f"./outputs/test_audio/gen_audio{SUB_NAME}")
    parser.add_argument("--gpus", type=str, default="0", help="使用的 GPU 列表，例如 '0,1,2,3'")
    parser.add_argument("--use_fp16", action="store_true", help="是否开启 fp16 推理")
    
    args = parser.parse_args()

    # 1. 设置使用的 GPU
    gpu_list = [int(i) for i in args.gpus.split(",")]
    num_gpus = len(gpu_list)
    print(f"Using GPUs: {gpu_list}")

    # 2. 准备输出目录
    os.makedirs(args.gen_audio_dir, exist_ok=True)

    # 3. 构建任务列表
    all_tasks = []
    
    # 确保数据长度对齐
    assert len(AUDIO_FILES) == len(NORM_TEXTS) == len(CHARAC_TEXTS) == len(CORNER_CASES), \
        "Error: The lengths of audio files and text lists do not match."

    for i, audio_file in enumerate(AUDIO_FILES):
        base_name = os.path.splitext(audio_file)[0]
        prompt_full_path = os.path.join(args.raw_audio_dir, audio_file)

        # 任务 1: Norm Text
        all_tasks.append({
            "prompt_path": prompt_full_path,
            "text": NORM_TEXTS[i],
            "output_path": os.path.join(args.gen_audio_dir, f"{base_name}_norm_text.wav")
        })

        # 任务 2: Charac Text
        all_tasks.append({
            "prompt_path": prompt_full_path,
            "text": CHARAC_TEXTS[i],
            "output_path": os.path.join(args.gen_audio_dir, f"{base_name}_charac_text.wav")
        })

        # 任务 3: Corner Case
        all_tasks.append({
            "prompt_path": prompt_full_path,
            "text": CORNER_CASES[i],
            "output_path": os.path.join(args.gen_audio_dir, f"{base_name}_corner_case.wav")
        })

    print(f"Total tasks generated: {len(all_tasks)}")

    # 4. 任务分发
    # 将任务平分给各个 GPU
    chunks = [all_tasks[i::num_gpus] for i in range(num_gpus)]

    # 5. 启动多进程
    mp.set_start_method('spawn', force=True) # CUDA 必须使用 spawn
    processes = []
    
    for i in range(num_gpus):
        p = mp.Process(
            target=worker, 
            args=(gpu_list[i], chunks[i], args)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"All processes finished. Results saved to {args.gen_audio_dir}")

if __name__ == "__main__":
    main()