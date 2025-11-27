#!/usr/bin/env python3
"""
计算评估结果JSON文件中所有数值指标的平均值
对于qwen_vl_vllm的结果，使用正则表达式匹配"Score: "后面的数字
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Union
from collections import defaultdict


def extract_qwen_score(response_text: str) -> Union[float, None]:
    """
    从qwen_vl_vllm的response中提取Score
    匹配格式: "Score: <数字>"
    """
    pattern = r'Score:\s*(\d+(?:\.\d+)?)'
    match = re.search(pattern, response_text)
    if match:
        return float(match.group(1))
    print("无法提取Score")
    # raise ValueError()
    return None


def collect_numeric_values(data: Any, path: str = "", values_dict: Dict[str, List[float]] = None, parent_key: str = "") -> Dict[str, List[float]]:
    """
    递归遍历数据结构，收集所有数值类型的值
    
    Args:
        data: 要遍历的数据
        path: 当前路径（用于追踪键的层次结构）
        values_dict: 存储数值的字典
        parent_key: 父级键名，用于特殊处理
    
    Returns:
        包含所有数值指标及其值列表的字典
    """
    if values_dict is None:
        values_dict = defaultdict(list)
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            # 特殊处理qwen_vl_vllm的response字段
            if key == "qwen_vl_vllm" and isinstance(value, dict) and "response" in value:
                response = value.get("response", "")
                score = extract_qwen_score(response)
                if score is not None:
                    metric_path = f"{current_path}.score"
                    values_dict[metric_path].append(score)
                # 继续处理其他字段（如果有的话）
                for sub_key, sub_value in value.items():
                    if sub_key != "response":
                        collect_numeric_values(sub_value, f"{current_path}.{sub_key}", values_dict, key)
            else:
                collect_numeric_values(value, current_path, values_dict, key)
    
    elif isinstance(data, list):
        # 如果是列表，检查是否为纯数值列表
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in data):
            # 这是一个数值列表，记录每个值
            for value in data:
                values_dict[path].append(float(value))
        else:
            # 递归处理列表中的每个元素
            for i, item in enumerate(data):
                collect_numeric_values(item, path, values_dict, parent_key)
    
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        # 找到数值，添加到对应的指标列表中
        # 对于av_offset字段，取绝对值
        value = float(data)
        if path.endswith('av_offset'):
            value = abs(value)
        values_dict[path].append(value)
    
    return values_dict


def calculate_averages(json_file: str, verbose: bool = False) -> Dict[str, float]:
    """
    计算JSON文件中所有数值指标的平均值
    
    Args:
        json_file: JSON文件路径
        verbose: 是否显示详细信息
    
    Returns:
        包含所有指标平均值的字典
    """
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 收集所有数值
    values_dict = collect_numeric_values(data)
    
    # 计算平均值
    averages = {}
    for metric_path, values in values_dict.items():
        if values:
            avg = sum(values) / len(values)
            averages[metric_path] = avg
            
            if verbose:
                print(f"\n指标: {metric_path}")
                print(f"  样本数: {len(values)}")
                print(f"  平均值: {avg:.6f}")
                print(f"  最小值: {min(values):.6f}")
                print(f"  最大值: {max(values):.6f}")
    
    return averages


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='计算评估结果JSON文件中所有数值指标的平均值',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算单个文件的平均值
  python calculate_average.py evaluation_results/evaluation_results.json
  
  # 显示详细信息（包括最小值、最大值等）
  python calculate_average.py evaluation_results/evaluation_results.json -v
  
  # 计算qwen评估结果的平均值
  python calculate_average.py evaluation_results_qwen/evaluation_results_qwen_vl_vllm.json
  
  # 输出到JSON文件
  python calculate_average.py evaluation_results/evaluation_results.json -o averages.json
        """
    )
    
    parser.add_argument('json_file', help='评估结果JSON文件路径')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    parser.add_argument('-o', '--output', help='输出结果到指定的JSON文件')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"错误: 文件不存在: {args.json_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"正在处理文件: {args.json_file}")
    print("=" * 80)
    
    # 计算平均值
    averages = calculate_averages(args.json_file, verbose=args.verbose)
    
    # 输出结果摘要
    if not args.verbose:
        print("\n所有指标的平均值:")
        print("-" * 80)
        for metric_path, avg_value in sorted(averages.items()):
            print(f"{metric_path}: {avg_value:.6f}")
    
    print("\n" + "=" * 80)
    print(f"共计算了 {len(averages)} 个指标的平均值")
    
    # 如果指定了输出文件，保存结果
    if args.output:
        output_data = {
            "source_file": str(json_path.absolute()),
            "metrics": {k: round(v, 6) for k, v in averages.items()},
            "total_metrics": len(averages)
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
