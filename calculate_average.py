#!/usr/bin/env python3
"""
Calculate the average of all numeric metrics in the evaluation results JSON file
For qwen_vl_vllm results, use regex to match the number after "Score: "
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Union
from collections import defaultdict


def extract_qwen_score(response_text: str) -> Union[float, None]:
    """
    Extract Score from qwen_vl_vllm response
    Match format: "Score: <number>"
    """
    pattern = r'Score:\s*(\d+(?:\.\d+)?)'
    match = re.search(pattern, response_text)
    if match:
        return float(match.group(1))
    print("Unable to extract Score")
    # raise ValueError()
    return None


def collect_numeric_values(data: Any, path: str = "", values_dict: Dict[str, List[float]] = None, parent_key: str = "") -> Dict[str, List[float]]:
    """
    Recursively traverse the data structure to collect all numeric values
    
    Args:
        data: Data to traverse
        path: Current path (used to track key hierarchy)
        values_dict: Dictionary to store values
        parent_key: Parent key name, used for special handling
    
    Returns:
        Dictionary containing all numeric metrics and their value lists
    """
    if values_dict is None:
        values_dict = defaultdict(list)
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            # Special handling for qwen_vl_vllm response field
            if key == "qwen_vl_vllm" and isinstance(value, dict) and "response" in value:
                response = value.get("response", "")
                score = extract_qwen_score(response)
                if score is not None:
                    metric_path = f"{current_path}.score"
                    values_dict[metric_path].append(score)
                # Continue processing other fields (if any)
                for sub_key, sub_value in value.items():
                    if sub_key != "response":
                        collect_numeric_values(sub_value, f"{current_path}.{sub_key}", values_dict, key)
            else:
                collect_numeric_values(value, current_path, values_dict, key)
    
    elif isinstance(data, list):
        # If it is a list, check if it is a pure numeric list
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in data):
            # This is a numeric list, record each value
            for value in data:
                values_dict[path].append(float(value))
        else:
            # Recursively process each element in the list
            for i, item in enumerate(data):
                collect_numeric_values(item, path, values_dict, parent_key)
    
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        # Found a numeric value, add to the corresponding metric list
        # For av_offset field, take the absolute value
        value = float(data)
        if path.endswith('av_offset'):
            value = abs(value)
        values_dict[path].append(value)
    
    return values_dict


def calculate_averages(json_file: str, verbose: bool = False) -> Dict[str, float]:
    """
    Calculate the average of all numeric metrics in the JSON file
    
    Args:
        json_file: JSON file path
        verbose: Whether to show detailed information
    
    Returns:
        Dictionary containing average values of all metrics
    """
    # Read JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Collect all numeric values
    values_dict = collect_numeric_values(data)
    
    # Calculate averages
    averages = {}
    for metric_path, values in values_dict.items():
        if values:
            avg = sum(values) / len(values)
            averages[metric_path] = avg
            
            if verbose:
                print(f"\nMetric: {metric_path}")
                print(f"  Sample count: {len(values)}")
                print(f"  Average: {avg:.6f}")
                print(f"  Min: {min(values):.6f}")
                print(f"  Max: {max(values):.6f}")
    
    return averages


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate the average of all numeric metrics in the evaluation results JSON file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate average for a single file
  python calculate_average.py evaluation_results/evaluation_results.json
  
  # Show detailed information (including min, max, etc.)
  python calculate_average.py evaluation_results/evaluation_results.json -v
  
  # Calculate average for qwen evaluation results
  python calculate_average.py evaluation_results_qwen/evaluation_results_qwen_vl_vllm.json
  
  # Output results to the specified JSON file
  python calculate_average.py evaluation_results/evaluation_results.json -o averages.json
        """
    )
    
    parser.add_argument('json_file', help='Path to evaluation results JSON file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed information')
    parser.add_argument('-o', '--output', help='Output results to the specified JSON file')
    
    args = parser.parse_args()
    
    # Check if file exists
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File not found: {args.json_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing file: {args.json_file}")
    print("=" * 80)
    
    # Calculate averages
    averages = calculate_averages(args.json_file, verbose=args.verbose)
    
    # Output result summary
    if not args.verbose:
        print("\nAverage of all metrics:")
        print("-" * 80)
        for metric_path, avg_value in sorted(averages.items()):
            print(f"{metric_path}: {avg_value:.6f}")
    
    print("\n" + "=" * 80)
    print(f"Calculated averages for {len(averages)} metrics")
    
    # If output file is specified, save results
    if args.output:
        output_data = {
            "source_file": str(json_path.absolute()),
            "metrics": {k: round(v, 6) for k, v in averages.items()},
            "total_metrics": len(averages)
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
