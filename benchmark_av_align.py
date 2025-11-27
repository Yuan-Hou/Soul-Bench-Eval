#!/usr/bin/env python3
"""Performance comparison script for av_align optimization."""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from video import VideoData
from subjects.av_align import evaluate

# Test with a single video
test_video_path = "/mnt/f/workspace/Video-Eval/input_data/results/sonic/0001_human_sing_en_male.mp4"
test_audio_path = "/mnt/f/workspace/Video-Eval/input_data/benchmark_v4/0001_human_sing_en_male.wav"

def run_benchmark(downsample: int, description: str):
    """Run benchmark with specific downsample factor."""
    print(f"\n{'='*60}")
    print(f"Testing: {description} (downsample={downsample})")
    print('='*60)
    
    video_data = VideoData(
        video_path=test_video_path,
        audio_path=test_audio_path,
    )
    
    start_time = time.time()
    
    results = evaluate(
        data_list=[video_data],
        device="cuda",
        batch_size=1,
        model_args={"downsample": downsample},
        sampling=None,
    )
    
    elapsed_time = time.time() - start_time
    
    # Print results
    av_align_result = results[0].results.get("av_align", {})
    
    if "error" in av_align_result:
        print(f"  ❌ Error: {av_align_result['error']}")
    else:
        print(f"  ✅ Completed in {elapsed_time:.2f} seconds")
        print(f"  IoU Score: {av_align_result.get('iou_score', 'N/A'):.4f}")
        print(f"  Audio Peaks: {av_align_result.get('num_audio_peaks', 'N/A')}")
        print(f"  Video Peaks: {av_align_result.get('num_video_peaks', 'N/A')}")
        print(f"  FPS: {av_align_result.get('fps', 'N/A'):.2f}")
        print(f"  Downsample Factor: {av_align_result.get('downsample_factor', 'N/A')}")
    
    return elapsed_time, av_align_result

if __name__ == "__main__":
    print("="*60)
    print("AV-Align Performance Benchmark")
    print("="*60)
    print(f"Video: {test_video_path}")
    print(f"Audio: {test_audio_path}")
    
    benchmarks = [
        (1, "Maximum Accuracy (no downsampling)"),
        (2, "Balanced Mode (default, 2x faster)"),
        (4, "Fast Mode (4x faster)"),
    ]
    
    results = []
    for downsample, description in benchmarks:
        try:
            elapsed, result = run_benchmark(downsample, description)
            results.append((downsample, description, elapsed, result))
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append((downsample, description, None, None))
    
    # Summary
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    
    base_time = None
    for downsample, description, elapsed, result in results:
        if elapsed is None:
            print(f"Downsample {downsample}x: FAILED")
            continue
            
        if base_time is None:
            base_time = elapsed
            speedup = 1.0
        else:
            speedup = base_time / elapsed
        
        iou_score = result.get('iou_score', 0) if result else 0
        
        print(f"Downsample {downsample}x: {elapsed:.2f}s | "
              f"Speedup: {speedup:.2f}x | IoU: {iou_score:.4f}")
    
    print("="*60)
    print("\n✨ Recommendation: Use downsample=2 for best balance of speed and accuracy")
