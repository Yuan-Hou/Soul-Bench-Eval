"""
并行执行视频评测任务

该脚本可以自动将评测任务分成多个组，并行执行，最后自动合并结果。
支持自动将多张 GPU 均匀分配给不同的任务组。

使用方法:
    # 使用 8 张 GPU，分成 8 组并行执行
    python parallel_evaluate.py \\
        --model_input_dir ./inputs \\
        --model_output_dir ./outputs \\
        --evaluate_subjects dino_consistency \\
        --group_total 8 \\
        --num_gpus 8
    
    # 使用 4 张 GPU，分成 8 组并行执行（每张 GPU 运行 2 个任务）
    python parallel_evaluate.py \\
        --model_input_dir ./inputs \\
        --model_output_dir ./outputs \\
        --evaluate_subjects dino_consistency \\
        --group_total 8 \\
        --num_gpus 4 \\
        --parallelism 4
    
    # 指定使用哪些 GPU
    python parallel_evaluate.py \\
        --model_input_dir ./inputs \\
        --model_output_dir ./outputs \\
        --evaluate_subjects dino_consistency \\
        --group_total 8 \\
        --gpu_ids 0,1,2,3,4,5,6,7
"""

import os
import sys
import pathlib
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import json


def run_evaluation_group(
    group_id: int,
    group_total: int,
    model_input_dir: str,
    model_output_dir: str,
    results_dir: str,
    evaluate_subjects: str,
    gpu_id: int,
    batch_size: int = 16,
    sampling: int = 0,
    model_args: str = '{}',
    filter_str: str = '',
    extra_args: List[str] = None
) -> Tuple[int, bool, str]:
    """
    运行单个评测组
    
    Args:
        group_id: 组号
        group_total: 总组数
        model_input_dir: 输入目录
        model_output_dir: 输出目录
        results_dir: 结果目录
        evaluate_subjects: 评测主题
        gpu_id: GPU ID
        batch_size: 批处理大小
        sampling: 采样帧数
        model_args: 模型参数
        filter_str: 文件名过滤
        extra_args: 额外参数列表
    
    Returns:
        (group_id, success, message): 组号、是否成功、消息
    """
    cmd = [
        'python', 'evaluate.py',
        '--model_input_dir', model_input_dir,
        '--model_output_dir', model_output_dir,
        '--results_dir', results_dir,
        '--evaluate_subjects', evaluate_subjects,
        '--device', f'cuda:{gpu_id}',
        '--batch_size', str(batch_size),
        '--sampling', str(sampling),
        '--model_args', model_args,
        '--group_id', str(group_id),
        '--group_total', str(group_total),
    ]
    
    if filter_str:
        cmd.extend(['--filter', filter_str])
    
    if extra_args:
        cmd.extend(extra_args)
    
    # 设置环境变量，指定使用的 GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    log_file = pathlib.Path(results_dir) / f'group{group_id}of{group_total}.log'
    
    try:
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"GPU: {gpu_id}\n")
            f.write(f"{'='*60}\n\n")
            f.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1
            )
            
            # 实时写入日志
            for line in process.stdout:
                f.write(line)
                f.flush()
            
            process.wait()
            
            if process.returncode == 0:
                return group_id, True, f"Group {group_id} completed successfully"
            else:
                return group_id, False, f"Group {group_id} failed with return code {process.returncode}"
    
    except Exception as e:
        return group_id, False, f"Group {group_id} failed with exception: {str(e)}"


def merge_results(results_dir: str, subjects: str, group_total: int, output_file: str = None) -> bool:
    """
    调用 merge_results.py 合并结果
    
    Args:
        results_dir: 结果目录
        subjects: 评测主题
        group_total: 总组数
        output_file: 输出文件路径（可选）
    
    Returns:
        是否成功
    """
    cmd = [
        'python', 'merge_results.py',
        '--results_dir', results_dir,
        '--subjects', subjects,
        '--group_total', str(group_total)
    ]
    
    if output_file:
        cmd.extend(['--output', output_file])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Merge failed: {e}")
        print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Parallel evaluation of video generation models with automatic GPU allocation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use 8 GPUs for 8 groups (one group per GPU)
  python parallel_evaluate.py --model_input_dir ./inputs --model_output_dir ./outputs \\
      --evaluate_subjects dino_consistency --group_total 8 --num_gpus 8
  
  # Use 4 GPUs for 8 groups with parallelism of 4 (2 groups per GPU)
  python parallel_evaluate.py --model_input_dir ./inputs --model_output_dir ./outputs \\
      --evaluate_subjects dino_consistency --group_total 8 --num_gpus 4 --parallelism 4
  
  # Specify which GPUs to use
  python parallel_evaluate.py --model_input_dir ./inputs --model_output_dir ./outputs \\
      --evaluate_subjects dino_consistency --group_total 8 --gpu_ids 0,2,4,6
        """
    )
    
    # 必需参数
    parser.add_argument('--model_input_dir', type=str, required=True,
                        help='Path to the input directory of the evaluated model.')
    parser.add_argument('--model_output_dir', type=str, required=True,
                        help='Path to the output directory of the evaluated model.')
    parser.add_argument('--evaluate_subjects', type=str, required=True,
                        help='Comma-separated list of evaluation subjects.')
    
    # 分组参数
    parser.add_argument('--group_total', type=int, required=True,
                        help='Total number of groups to split the evaluation into.')
    
    # GPU 配置
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument('--num_gpus', type=int, default=8,
                          help='Number of GPUs to use (default: 8). GPUs will be numbered 0 to num_gpus-1.')
    gpu_group.add_argument('--gpu_ids', type=str,
                          help='Comma-separated list of GPU IDs to use, e.g., "0,1,2,3,4,5,6,7".')
    
    # 并行度
    parser.add_argument('--parallelism', type=int, default=None,
                        help='Maximum number of parallel tasks. Default is equal to number of GPUs.')
    
    # 评测参数
    parser.add_argument('--results_dir', type=str, default='./evaluation_results',
                        help='Path to the directory where evaluation results will be saved.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation (default: 16).')
    parser.add_argument('--sampling', type=int, default=0,
                        help='Number of frames to sample (default: 0, use all frames).')
    parser.add_argument('--model_args', type=str, default='{}',
                        help='Additional model arguments in JSON format.')
    parser.add_argument('--filter', type=str, default='',
                        help='Filter to only evaluate videos whose filenames contain this string.')
    
    # 合并结果
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for merged results.')
    parser.add_argument('--skip_merge', action='store_true',
                        help='Skip merging results after all groups complete.')
    
    # 其他选项
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing them.')
    
    args = parser.parse_args()
    
    # 解析 GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    else:
        gpu_ids = list(range(args.num_gpus))
    
    num_gpus = len(gpu_ids)
    
    # 设置并行度
    if args.parallelism is None:
        parallelism = num_gpus
    else:
        parallelism = args.parallelism
    
    # 验证参数
    if args.group_total <= 0:
        print("Error: --group_total must be greater than 0")
        sys.exit(1)
    
    if parallelism <= 0:
        print("Error: --parallelism must be greater than 0")
        sys.exit(1)
    
    # 创建结果目录
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"Parallel Evaluation Configuration:")
    print(f"{'='*60}")
    print(f"Model Input Dir:     {args.model_input_dir}")
    print(f"Model Output Dir:    {args.model_output_dir}")
    print(f"Results Dir:         {args.results_dir}")
    print(f"Evaluate Subjects:   {args.evaluate_subjects}")
    print(f"Total Groups:        {args.group_total}")
    print(f"Available GPUs:      {gpu_ids}")
    print(f"Number of GPUs:      {num_gpus}")
    print(f"Parallelism:         {parallelism}")
    print(f"Batch Size:          {args.batch_size}")
    print(f"Sampling:            {args.sampling}")
    print(f"{'='*60}\n")
    
    if args.dry_run:
        print("DRY RUN MODE - Commands that would be executed:")
        for group_id in range(args.group_total):
            gpu_id = gpu_ids[group_id % num_gpus]
            print(f"\nGroup {group_id} (GPU {gpu_id}):")
            cmd = [
                'python', 'evaluate.py',
                '--model_input_dir', args.model_input_dir,
                '--model_output_dir', args.model_output_dir,
                '--results_dir', args.results_dir,
                '--evaluate_subjects', args.evaluate_subjects,
                '--device', f'cuda:{gpu_id}',
                '--batch_size', str(args.batch_size),
                '--sampling', str(args.sampling),
                '--model_args', args.model_args,
                '--group_id', str(group_id),
                '--group_total', str(args.group_total),
            ]
            if args.filter:
                cmd.extend(['--filter', args.filter])
            print('  ' + ' '.join(cmd))
        print("\nDry run complete. No tasks were executed.")
        return
    
    # 执行并行评测
    start_time = time.time()
    completed_groups = []
    failed_groups = []
    
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        # 提交所有任务
        future_to_group = {}
        for group_id in range(args.group_total):
            # 循环分配 GPU
            gpu_id = gpu_ids[group_id % num_gpus]
            
            future = executor.submit(
                run_evaluation_group,
                group_id=group_id,
                group_total=args.group_total,
                model_input_dir=args.model_input_dir,
                model_output_dir=args.model_output_dir,
                results_dir=args.results_dir,
                evaluate_subjects=args.evaluate_subjects,
                gpu_id=gpu_id,
                batch_size=args.batch_size,
                sampling=args.sampling,
                model_args=args.model_args,
                filter_str=args.filter
            )
            future_to_group[future] = group_id
            print(f"Submitted group {group_id} to GPU {gpu_id}")
        
        print(f"\nAll {args.group_total} groups submitted. Waiting for completion...\n")
        
        # 收集结果
        for future in as_completed(future_to_group):
            group_id, success, message = future.result()
            
            if success:
                completed_groups.append(group_id)
                print(f"✓ {message}")
            else:
                failed_groups.append(group_id)
                print(f"✗ {message}")
            
            print(f"  Progress: {len(completed_groups) + len(failed_groups)}/{args.group_total} "
                  f"(Success: {len(completed_groups)}, Failed: {len(failed_groups)})\n")
    
    elapsed_time = time.time() - start_time
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"Evaluation Summary:")
    print(f"{'='*60}")
    print(f"Total groups:        {args.group_total}")
    print(f"Completed:           {len(completed_groups)}")
    print(f"Failed:              {len(failed_groups)}")
    print(f"Time elapsed:        {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    if failed_groups:
        print(f"\nFailed groups: {sorted(failed_groups)}")
        print("Check log files for details.")
    
    # 合并结果
    if not args.skip_merge and len(completed_groups) > 0:
        print(f"\n{'='*60}")
        print("Merging results...")
        print(f"{'='*60}\n")
        
        merge_success = merge_results(
            results_dir=args.results_dir,
            subjects=args.evaluate_subjects,
            group_total=args.group_total,
            output_file=args.output
        )
        
        if merge_success:
            print("\n✓ All results merged successfully!")
        else:
            print("\n✗ Failed to merge results. You can try merging manually with merge_results.py")
            sys.exit(1)
    elif args.skip_merge:
        print("\nSkipping merge (--skip_merge specified)")
    else:
        print("\nNo results to merge (all groups failed)")
        sys.exit(1)
    
    if failed_groups:
        print(f"\n⚠ Warning: {len(failed_groups)} group(s) failed. Check log files.")
        sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print("✓ All tasks completed successfully!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
