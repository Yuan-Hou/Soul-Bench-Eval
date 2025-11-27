#!/usr/bin/env python3
"""Test script for av_align evaluation subject."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from video import VideoData
from subjects.av_align import evaluate

# Test with a single video
test_video_path = "/mnt/f/workspace/Video-Eval/input_data/results/sonic/0001_human_sing_en_male.mp4"
test_audio_path = "/mnt/f/workspace/Video-Eval/input_data/benchmark_v4/0001_human_sing_en_male.wav"

if __name__ == "__main__":
    print("Testing AV-Align evaluation...")
    print(f"Video: {test_video_path}")
    print(f"Audio: {test_audio_path}")
    
    # Create VideoData object
    video_data = VideoData(
        video_path=test_video_path,
        audio_path=test_audio_path,
    )
    
    # Run evaluation
    results = evaluate(
        data_list=[video_data],
        device="cuda",
        batch_size=1,
        model_args={},
        sampling=None,
    )
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    for result in results:
        print(f"\nVideo: {result.video_path}")
        av_align_result = result.results.get("av_align", {})
        if "error" in av_align_result:
            print(f"  Error: {av_align_result['error']}")
        else:
            print(f"  IoU Score: {av_align_result.get('iou_score', 'N/A')}")
            print(f"  Audio Peaks: {av_align_result.get('num_audio_peaks', 'N/A')}")
            print(f"  Video Peaks: {av_align_result.get('num_video_peaks', 'N/A')}")
            print(f"  FPS: {av_align_result.get('fps', 'N/A')}")
    print("="*60)
