import os
import sys
import pathlib
import json
import argparse
import hashlib
from tqdm import tqdm
from video import VideoData
from utils import load_json, save_json

# 是否默认使用全部帧进行评测，若为 False，则默认采样16帧，因为我的电脑跑全部帧会爆内存
DEFAULT_ALL_FRAMES = True

# --model_input_dir和--model_output_dir参数，即被评测模型的输入和输出文件夹路径
# 评测脚本会遍历输出文件夹中的视频文件，并根据对应的输入文件夹中的同名图片、音频、文本文件，进行评测
parser = argparse.ArgumentParser(description="Evaluate video generation models.")
parser.add_argument('--model_input_dir', type=str, required=True,
                    help='Path to the input directory of the evaluated model. This directory should contain the reference images, audio, and text files corresponding to the generated videos.')
parser.add_argument('--model_output_dir', type=str, required=True,
                    help='Path to the output directory of the evaluated model. This directory should contain the generated video files.')

# results_dir参数，指定评测结果的保存路径
parser.add_argument('--results_dir', type=str, default='./evaluation_results',
                    help='Path to the directory where evaluation results will be saved.')

# --evaluate_subjects参数，指定需要运行的评测主题，多个主题用逗号分隔
# 例如：--evaluate_subjects dino_consistency,audio_visual_synchrony
parser.add_argument('--evaluate_subjects', type=str, required=True,
                    help='Comma-separated list of evaluation subjects to run, e.g., "dino_consistency,audio_visual_synchrony".')



# --device参数，指定运行评测时使用的设备，默认为'cuda'
parser.add_argument('--device', type=str, default='cuda',
                    help='Device to use for evaluation, e.g., "cuda" or "cpu". Default is "cuda".')
# --batch_size参数，指定评测时的批处理大小，默认为16
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size to use during evaluation. Default is 16.')

if DEFAULT_ALL_FRAMES:
    # --sampling 参数，指定评测时的视频采样数量，默认为0，表示使用全部帧
    parser.add_argument('--sampling', type=int, default=0,
                        help='Number of frames to sample from each video during evaluation. Default is 0 (use all frames).')
else:
    # --sampling 参数，指定评测时的视频采样数量，默认为16，0表示使用全部帧
    parser.add_argument('--sampling', type=int, default=16,
                        help='Number of frames to sample from each video during evaluation. Default is 16.')

# --model_args参数，传递给评测主题模型的额外参数，格式为JSON字符串，如果有多个评测主题，可以为每个主题传递不同的参数，用分号分隔
parser.add_argument('--model_args', type=str, default='{}',
                    help='Additional arguments for the evaluation subject models in JSON format.')

# --filter 参数，指定在评测前只过滤出文件名包含该字符串的文件进行评测，默认为空表示不过滤
parser.add_argument('--filter', type=str, default='',
                    help='Filter to only evaluate videos whose filenames contain this string. Default is empty (no filter).')

# --group_id 参数，指定当前评测的组号（从0开始），需要与--group_total配合使用
parser.add_argument('--group_id', type=int, default=None,
                    help='Group ID for hash-based file partitioning (0-indexed). Must be used with --group_total.')

# --group_total 参数，指定总共分成多少组，需要与--group_id配合使用
parser.add_argument('--group_total', type=int, default=None,
                    help='Total number of groups for hash-based file partitioning. Must be used with --group_id.')

args = parser.parse_args()

def main():
    model_input_dir = pathlib.Path(args.model_input_dir)
    model_output_dir = pathlib.Path(args.model_output_dir)
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    evaluate_subjects = [subj.strip() for subj in args.evaluate_subjects.split(',')]
    device = args.device
    batch_size = args.batch_size
    sampling = args.sampling
    model_args_list = [json.loads(arg) for arg in args.model_args.split(';')]

    # 验证分组参数
    if (args.group_id is not None) != (args.group_total is not None):
        raise ValueError("--group_id and --group_total must be used together.")
    
    if args.group_id is not None:
        if args.group_id < 0 or args.group_id >= args.group_total:
            raise ValueError(f"--group_id must be between 0 and {args.group_total - 1}.")
        if args.group_total <= 0:
            raise ValueError("--group_total must be greater than 0.")
        print(f"Using hash-based file partitioning: group {args.group_id} of {args.group_total}")


    # 构建VideoData列表
    data_list = []
    for video_file in model_output_dir.glob('*.mp4'):
        video_filename = video_file.stem
        if args.filter and args.filter not in video_filename:
            continue
        
        # 如果启用了分组功能，根据文件名哈希取模判断是否属于当前组
        if args.group_id is not None:
            # 使用 hashlib.md5 确保在不同进程和机器上哈希值一致
            file_hash = int(hashlib.md5(video_filename.encode('utf-8')).hexdigest(), 16)
            group = file_hash % args.group_total
            if group != args.group_id:
                continue
        
        # 假设对应的参考图像、音频和文本文件与视频文件同名但扩展名不同
        image_file = model_input_dir / f"{video_filename}.png"
        audio_file = model_input_dir / f"{video_filename}.wav"
        text_file = model_input_dir / f"{video_filename}.json"
        video_data = VideoData(
            video_path=str(video_file),
            audio_path=str(audio_file) if image_file.exists() else None,
            text_path=str(text_file) if text_file.exists() else None,
            image_path=str(image_file) if image_file.exists() else None
        )
        data_list.append(video_data)
    # data_list = data_list[:3]
    # 逐个评测主题运行评测
    for subject, model_args in zip(evaluate_subjects, model_args_list):
        print(f"Running evaluation for subject: {subject}")
        subject_module = __import__(f"subjects.{subject}", fromlist=['evaluate'])
        data_list = subject_module.evaluate(
            data_list,
            device=device,
            batch_size=batch_size,
            model_args=model_args,
            sampling=sampling
        )

    # 保存评测结果
    results = [data.to_dict() for data in data_list]
    # 如果使用了分组功能，在文件名中包含组信息
    if args.group_id is not None:
        results_path = results_dir / f"evaluation_results_{'-'.join(evaluate_subjects)}_group{args.group_id}of{args.group_total}.json"
    else:
        results_path = results_dir / f"evaluation_results_{'-'.join(evaluate_subjects)}.json"
    save_json(results, str(results_path))
    print(f"Evaluation results saved to {results_path}")
    print(f"Total videos evaluated: {len(data_list)}")

if __name__ == "__main__":
    main()
    # Example usage:
    # python evaluate.py --model_input_dir /mnt/f/temp/video-eval-mock/inputs --model_output_dir /mnt/f/temp/video-eval-mock/results --evaluate_subjects dino_consistency --model_args '{"model_name": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"}'