#!/usr/bin/env python3
"""
合并多个分组的评测结果文件

该脚本用于合并通过 --group_id 和 --group_total 参数分组评测产生的多个 JSON 结果文件。
可以自动检测并合并同一评测主题的所有分组结果，也可以手动指定要合并的文件。

使用方法:
    # 自动模式：自动检测并合并指定目录下的所有分组结果
    python merge_results.py --results_dir ./evaluation_results --subjects dino_consistency --group_total 4
    
    # 手动模式：指定要合并的文件列表
    python merge_results.py --input_files result1.json result2.json result3.json --output merged.json
"""

import os
import sys
import pathlib
import json
import argparse
from typing import List, Dict, Any
from utils import load_json, save_json


def merge_evaluation_results(input_files: List[str], output_file: str, verbose: bool = True) -> None:
    """
    合并多个评测结果文件
    
    Args:
        input_files: 输入文件路径列表
        output_file: 输出文件路径
        verbose: 是否打印详细信息
    """
    if not input_files:
        raise ValueError("No input files provided.")
    
    # 检查所有输入文件是否存在
    missing_files = [f for f in input_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"The following files do not exist: {missing_files}")
    
    # 加载所有结果文件
    all_results = []
    video_ids_seen = set()
    duplicate_count = 0
    
    for file_path in input_files:
        if verbose:
            print(f"Loading {file_path}...")
        
        results = load_json(file_path)
        
        if not isinstance(results, list):
            raise ValueError(f"File {file_path} does not contain a list of results.")
        
        # 检查重复的视频（基于视频路径）
        for result in results:
            video_path = result.get('video_path', '')
            if video_path in video_ids_seen:
                duplicate_count += 1
                if verbose:
                    print(f"  Warning: Duplicate video found: {video_path}")
            else:
                video_ids_seen.add(video_path)
                all_results.append(result)
        
        if verbose:
            print(f"  Loaded {len(results)} results from {file_path}")
    
    # 按视频路径排序，使结果更有序
    all_results.sort(key=lambda x: x.get('video_path', ''))
    
    # 保存合并后的结果
    save_json(all_results, output_file)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Merge completed successfully!")
        print(f"Total input files: {len(input_files)}")
        print(f"Total unique videos: {len(all_results)}")
        print(f"Duplicate videos skipped: {duplicate_count}")
        print(f"Merged results saved to: {output_file}")
        print(f"{'='*60}")


def auto_detect_and_merge(results_dir: str, subjects: str, group_total: int, 
                          output_file: str = None, verbose: bool = True) -> None:
    """
    自动检测并合并指定目录下的分组结果
    
    Args:
        results_dir: 结果文件所在目录
        subjects: 评测主题（多个主题用逗号分隔）
        group_total: 总分组数
        output_file: 输出文件路径（可选）
        verbose: 是否打印详细信息
    """
    results_dir = pathlib.Path(results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    
    # 构建预期的文件名模式
    subjects_str = '-'.join([s.strip() for s in subjects.split(',')])
    
    # 查找所有分组结果文件
    input_files = []
    missing_groups = []
    
    for group_id in range(group_total):
        expected_filename = f"evaluation_results_{subjects_str}_group{group_id}of{group_total}.json"
        file_path = results_dir / expected_filename
        
        if file_path.exists():
            input_files.append(str(file_path))
            if verbose:
                print(f"Found group {group_id}: {file_path}")
        else:
            missing_groups.append(group_id)
            if verbose:
                print(f"Warning: Missing group {group_id}: {file_path}")
    
    if not input_files:
        raise FileNotFoundError(f"No group result files found in {results_dir}")
    
    if missing_groups:
        print(f"\nWarning: {len(missing_groups)} group(s) missing: {missing_groups}")
        response = input("Continue with available files? (y/n): ")
        if response.lower() != 'y':
            print("Merge cancelled.")
            return
    
    # 确定输出文件路径
    if output_file is None:
        output_file = str(results_dir / f"evaluation_results_{subjects_str}.json")
    
    # 执行合并
    print(f"\nMerging {len(input_files)} files...")
    merge_evaluation_results(input_files, output_file, verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Merge evaluation results from multiple group files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto mode: Automatically detect and merge all group results
  python merge_results.py --results_dir ./evaluation_results --subjects dino_consistency --group_total 4
  
  # Manual mode: Specify input files explicitly
  python merge_results.py --input_files result1.json result2.json result3.json --output merged.json
  
  # Auto mode with custom output file
  python merge_results.py --results_dir ./evaluation_results --subjects dino_consistency,video_quality --group_total 8 --output final_results.json
        """
    )
    
    # 创建互斥组：自动模式或手动模式
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    # 自动模式参数
    mode_group.add_argument('--results_dir', type=str,
                           help='Path to the directory containing group result files (auto mode).')
    
    # 手动模式参数
    mode_group.add_argument('--input_files', type=str, nargs='+',
                           help='List of input JSON files to merge (manual mode).')
    
    # 自动模式所需的额外参数
    parser.add_argument('--subjects', type=str,
                       help='Comma-separated list of evaluation subjects (required for auto mode).')
    
    parser.add_argument('--group_total', type=int,
                       help='Total number of groups (required for auto mode).')
    
    # 通用参数
    parser.add_argument('--output', type=str,
                       help='Output file path for merged results. If not specified, will be auto-generated in auto mode.')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output.')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    try:
        if args.results_dir:
            # 自动模式
            if not args.subjects or not args.group_total:
                parser.error("--subjects and --group_total are required when using --results_dir")
            
            auto_detect_and_merge(
                results_dir=args.results_dir,
                subjects=args.subjects,
                group_total=args.group_total,
                output_file=args.output,
                verbose=verbose
            )
        else:
            # 手动模式
            if not args.output:
                parser.error("--output is required when using --input_files")
            
            merge_evaluation_results(
                input_files=args.input_files,
                output_file=args.output,
                verbose=verbose
            )
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
